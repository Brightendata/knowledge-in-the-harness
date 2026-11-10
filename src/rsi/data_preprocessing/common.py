"""What the airline and retail generators share: a request and its reference, the checks every task passes, the target
distribution read from the knowledge-point tags of the official tasks (data/test_annotation.json), user behaviours, and
the choice of V and the training batches."""
import json
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from tau2.data_model.tasks import (Action, Description, EvaluationCriteria, RewardType, StructuredUserInstructions, Task,
                                   UserScenario)
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.environment import get_environment as airline_environment
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import get_environment as retail_environment

ROOT = Path(__file__).resolve().parents[3]
ANNOTATION = ROOT / "data/test_annotation.json"
# tags of the operations a task asks for, whose number per task the strata count
OPS = {
    "airline": {"book", "mod_flights", "cabin_change", "bags_add", "bags_no_remove", "pax_modify", "pax_no_count",
                "cancel_ok", "cancel_refuse", "comp_refuse", "ins_no_add", "deny_other", "same_od", "basic_no_change"},
    "retail": {"cancel", "mod_items", "mod_addr", "mod_pay", "user_addr", "return", "exchange", "deny", "same_type",
               "cancel_partial_no", "catalog", "order_info"},
}
OUTCOMES = ("out_transfer", "out_partial", "out_nowrite", "out_write")
WRITES = {
    "airline": {"book_reservation", "cancel_reservation", "update_reservation_flights", "update_reservation_baggages",
                "update_reservation_passengers", "send_certificate"},
    "retail": {"cancel_pending_order", "modify_pending_order_items", "modify_pending_order_address",
               "modify_pending_order_payment", "return_delivered_order_items", "exchange_delivered_order_items",
               "modify_user_address"},
}
ENVIRONMENT = {"airline": lambda dump: airline_environment(db=FlightDB.model_validate(dump)),
               "retail": lambda dump: retail_environment(db=RetailDB.model_validate(dump))}
V_CUSTOMERS = 0.3  # share of customers whose tasks are kept for V
CHUNK = 64  # candidates built ahead, whose checks run in parallel before they are recorded in turn


def outputs(domain):
    """The candidates, the chosen tasks, the splits and the report of a domain."""
    return (ROOT / f"data/processed/{domain}_generated.jsonl", ROOT / f"data/processed/{domain}_final.jsonl",
            ROOT / f"data/splits/{domain}.json", ROOT / f"data/reports/{domain}_generate.json")


@dataclass
class Part:
    """One request of a task."""
    spec: dict  # what the scenario states, for the domain's tests
    purpose: str
    request: str  # sentences for reason_for_call, starting with the request itself
    lines: list  # sentences for task_instructions
    calls: list = field(default_factory=list)  # reference writes
    forbidden: dict | None = None  # the call that would grant what the agent must refuse or the user declines
    info: list = field(default_factory=list)  # strings the agent must state
    ask: str = ""  # the sentence asking for them
    by_policy: bool = True  # whether the forbidden call breaks the policy, rather than a condition of the user's
    known: list = field(default_factory=list)  # sentences for known_info
    tags: set = field(default_factory=set)  # knowledge points, as in data/test_annotation.json
    # A refused request no tool could grant (adding insurance, splitting a payment) scores the same whatever the agent
    # does, so it only goes with a request that writes.
    needs_company: bool = False
    transfer: bool = False  # the agent should end by transferring the user


def outcome(task, domain):
    """Reference write calls as a set, flights reduced to (number, date), to tell tasks with the same outcome apart."""
    calls = []
    for a in task.evaluation_criteria.actions or []:
        if a.name in WRITES[domain]:
            g = dict(a.arguments)
            if "flights" in g:
                g["flights"] = [(f["flight_number"], f["date"]) for f in g["flights"]]
            calls.append((a.name, json.dumps(g, sort_keys=True)))
    return tuple(sorted(calls))


def run(domain, dump, calls):
    """The database after the calls, or None if one of them fails."""
    env = ENVIRONMENT[domain](dump)
    try:
        for c in calls:
            env.make_tool_call(tool_name=c["name"], **c["arguments"])
    except Exception:
        return None
    return env.tools.db.model_dump()


def instead(parts, k):
    """The reference calls with part k's forbidden call in place of that part's own calls."""
    return [c for i, p in enumerate(parts) for c in ([p.forbidden] if i == k else p.calls)]


def verified(domain, dump, parts):
    """The reference replays without error; making a forbidden call in place of its part's reference calls would
    succeed and leave a different database; a request no tool could grant comes with one that writes; and with several
    requests that write, their calls give the same database in reverse order and each is needed."""
    calls = [c for p in parts for c in p.calls]
    if any(p.needs_company for p in parts) and not calls:
        return False
    end = run(domain, dump, calls)
    if end is None:
        return False
    for k, p in enumerate(parts):
        if p.forbidden and run(domain, dump, instead(parts, k)) in (None, end):
            return False
    writing = [p.calls for p in parts if p.calls]
    if len(writing) > 1:
        if run(domain, dump, [c for w in reversed(writing) for c in w]) != end:
            return False
        if any(run(domain, dump, [c for j, w in enumerate(writing) if j != i for c in w]) == end
               for i in range(len(writing))):
            return False
    return True


def make_task(domain, task_id, parts, reason, known, unknown, lines, info):
    """A tau2 task: the user's scenario and, as the reference, the requests' calls and the information to state."""
    calls = [c for p in parts for c in p.calls]
    return Task(
        id=task_id,
        description=Description(purpose="; ".join(p.purpose for p in parts)),
        user_scenario=UserScenario(instructions=StructuredUserInstructions(
            domain=domain, reason_for_call=reason, known_info=known, unknown_info=unknown,
            task_instructions="\n".join(lines) or ".")),
        evaluation_criteria=EvaluationCriteria(
            actions=[Action(action_id=f"{task_id}_{k}", name=c["name"], arguments=c["arguments"])
                     for k, c in enumerate(calls)],
            communicate_info=info, reward_basis=[RewardType.DB, RewardType.COMMUNICATE]))


_worker = {}


def _start(domain, dump):
    _worker.update(domain=domain, dump=dump)


def _verified(parts):
    return verified(_worker["domain"], _worker["dump"], parts)


def drop(parts, k, info_tags):
    """The parts with request k dropped by the user when asked to confirm it: its one call becomes the call the agent
    must not make."""
    p = parts[k]
    dropped = replace(p, spec={**p.spec, "dropped": True}, calls=[], forbidden=p.calls[0], by_policy=False, info=[],
                      ask="", tags=(p.tags - info_tags) | {"change_mind"},
                      purpose=p.purpose + "; the user drops it when asked to confirm")
    return parts[:k] + [dropped] + parts[k + 1:]


def leaks(a, b):
    """Whether what request a has the agent state turns up in request b's calls or words, which the agent reads out or
    repeats anyway."""
    said = json.dumps([c["arguments"] for c in b.calls + ([b.forbidden] if b.forbidden else [])])
    said = " ".join([said, b.request, *b.lines]).lower()
    return any(x.lower() in said for x in a.info)


class Made:
    """The tasks made so far, as (family, task, parts, tags, customer, group), and how many of them each customer and
    each reservation or order is in. objects(part): the reservations or orders a request is about; caps: (most tasks
    per customer, per reservation or order); droppable(part): whether the user may drop the request when asked to
    confirm it, which a task of several writing requests does at drop_rate. verified() runs in a pool of processes:
    prefetch() starts the checks of the next candidates, which are then recorded one by one as before, so the outcome
    does not depend on the parallelism."""

    def __init__(self, domain, db, taken, assemble, objects, caps, droppable, drop_rate, info_tags):
        self.domain, self.db, self.dump, self.taken, self.assemble = domain, db, db.model_dump(), taken, assemble
        self.objects, self.caps, self.droppable, self.drop_rate, self.info_tags = (objects, caps, droppable, drop_rate,
                                                                                   info_tags)
        self.tasks, self.uses, self.per_user = [], Counter(), Counter()
        self.pool = ProcessPoolExecutor(os.cpu_count(), initializer=_start, initargs=(domain, self.dump))
        self.checks = {}  # what verified() reads of the parts -> its result

    def prefetch(self, part_lists):
        for parts in part_lists:
            key = json.dumps([[p.calls, p.forbidden, p.needs_company] for p in parts])
            if key not in self.checks:
                self.checks[key] = self.pool.submit(_verified, parts)

    def verified(self, parts):
        self.prefetch([parts])
        return self.checks[json.dumps([[p.calls, p.forbidden, p.needs_company] for p in parts])].result()

    def free(self, user_id, parts=()):
        return self.per_user[user_id] < self.caps[0] and all(self.uses[x] < self.caps[1] for p in parts
                                                             for x in self.objects(p))

    def record(self, family, ids, user_id, part_lists, group=None):
        """Assemble and keep the tasks, all of them or none: each must pass verified() and have an outcome no evaluation
        task or earlier task has. A variant group shares its seed, so its tasks differ only where their requests do."""
        out = []
        for task_id, parts in zip(ids, part_lists):
            if not self.verified(parts):
                return 0
            rng = random.Random(group or task_id)
            mind = None
            single = [k for k, p in enumerate(parts) if len(p.calls) == 1 and self.droppable(p)]
            if group is None and sum(bool(p.calls) for p in parts) > 1 and single and rng.random() < self.drop_rate:
                mind = rng.choice(single)
                parts = drop(parts, mind, self.info_tags)
            task, tags = self.assemble(self.db, task_id, user_id, parts, rng, mind)
            said = str(task.user_scenario.instructions).lower()
            if any(x.lower() in said for x in task.evaluation_criteria.communicate_info):  # the user would say it
                return 0
            o = outcome(task, self.domain)
            if o and (o in self.taken or any(o == outcome(t, self.domain) for t, _, _ in out)):
                return 0
            out.append((task, tags, parts))
        for task, tags, parts in out:
            self.taken.add(outcome(task, self.domain))
            self.tasks.append((family, task, parts, tags, user_id, group or task.id))
            self.uses.update(x for p in parts for x in self.objects(p))
            self.per_user[user_id] += 1
        return 1

    def singles(self, family, quota, candidates, name, owner, builds):
        """Up to quota tasks of one request each, a candidate (reservation, order or customer) at a time: with several
        builds, a variant group of one task per build."""
        count = 0
        for start in range(0, len(candidates), CHUNK):
            chunk = []
            for x in candidates[start:start + CHUNK]:
                base = f"generated_{family}_{name(x)}"
                parts = [b(self.db, x, random.Random(base)) for b in builds]
                if not any(p is None or p.needs_company for p in parts):
                    chunk.append((x, base, parts))
            self.prefetch([[p] for x, _, parts in chunk if self.free(owner(x), parts) for p in parts])
            for x, base, parts in chunk:
                if count == quota:
                    return
                if not self.free(owner(x), parts):
                    continue
                ids = [base] if len(builds) == 1 else [f"{base}_v{k}" for k in range(len(builds))]
                count += self.record(family, ids, owner(x), [[p] for p in parts], base if len(builds) > 1 else None)
            if count == quota:
                return


def writes_tag(n):
    """A pseudo-tag for the number of reference writes, so that the choice also matches how many a task has."""
    return f"writes_{min(n, 3)}"


def stratum(tags, domain):
    """(outcome, number of kinds of operation up to 3, whether a fallback is involved)."""
    outcome_ = next(o for o in OUTCOMES if o in tags)
    return outcome_, min(len(tags & OPS[domain]), 3), "fallback" in tags


# tags of a task's shape, or of the branch of a policy rule it takes (the domains' BRANCHES), rather than a knowledge point
PSEUDO = ("writes_", "out_", "owned_", "shape_", "br_")
# tags of how the user talks or how the task is written, drawn on their own: their pairs with knowledge points say
# nothing about how the knowledge points combine
LOOSE = {"emotion", "offtopic", "reactive", "order", "pressure", "no_transfer", "agent_decides", "search_direct",
         "search_onestop", "multi", "find_res", "find_order"}


def joint(tags):
    """The tags and, for a task with two or more reference writes, each knowledge point again as "w2:<tag>", so that the
    choice also matches which knowledge points come with several writes (in the official retail tasks, addresses taken
    from other orders and status mix-ups mostly do; requests over all of a customer's orders mostly do not)."""
    if not tags & {"writes_2", "writes_3"}:
        return set(tags)
    return set(tags) | {f"w2:{t}" for t in tags if not t.startswith(PSEUDO)}


def pairs(tags):
    """Each pair of a task's knowledge points as "pair:<a>+<b>"."""
    points = sorted(t for t in tags if not t.startswith(PSEUDO) and t not in LOOSE)
    return {f"pair:{a}+{b}" for k, a in enumerate(points) for b in points[k + 1:]}


def shape(tags):
    """A pseudo-tag for one request or several with how many writes (up to 3): in the official tasks, several requests
    mostly come with several writes, which the two counts apart would not keep."""
    writes = next(t for t in tags if t.startswith("writes_"))
    return f"shape_{'multi' if 'multi' in tags else 'one'}_{writes[len('writes_'):]}"


def owned(n, domain):
    """A pseudo-tag for how many orders (retail) or reservations (airline) the customer has: the more there are, the
    more the agent has to look through."""
    top = 5 if domain == "retail" else 7
    return f"owned_{min(n, top)}"


def profile(domain, owners=None):
    """Share of the official tasks with each tag (and number of reference writes, each tag of the tasks with several
    writes, each pair of knowledge points found in two or more official stories, and with owners, task id -> owned
    tag, how many orders or reservations their customers have), and in each stratum."""
    from rsi.data_preprocessing.alignment import stories  # it imports this module
    official = {t["id"]: t for t in json.loads((ROOT / f"tau2-bench/data/tau2/domains/{domain}/tasks.json").read_text())}
    n = {k: sum(a["name"] in WRITES[domain] for a in t["evaluation_criteria"].get("actions") or [])
         for k, t in official.items()}
    annotation = json.loads(ANNOTATION.read_text())[domain]
    story = dict(zip(annotation, stories([official[k] for k in annotation])))
    seen = {}  # pair -> the stories it is in
    for k, t in annotation.items():
        for p in pairs(set(t)):
            seen.setdefault(p, set()).add(story[k])
    tasks = [joint(set(t) | {writes_tag(n[k])}) | {p for p in pairs(set(t)) if len(seen[p]) > 1}
             | {shape(set(t) | {writes_tag(n[k])})} | ({owners[k]} if owners else set()) for k, t in annotation.items()]
    tags = Counter(t for x in tasks for t in x)
    strata = Counter(stratum(x, domain) for x in tasks)
    return {t: c / len(tasks) for t, c in tags.items()}, {s: c / len(tasks) for s, c in strata.items()}


def select(pool, domain, sizes, minima, cap=3, seed=20261026, owners=None, branches=None):
    """V and the training batches, in the order of sizes. pool: task id -> {"tags", "customer", "group", "v"}, where v
    marks the customers set aside for V; sizes: split -> number of tasks; minima: split -> (least tasks per tag of the
    official tasks, least per tag only the generated tasks have, or None to leave such tasks out); cap: most tasks of
    one customer in a split; owners: official task id -> owned tag of its customer, matched by the owned tags in the
    pool's tags; branches: branch tag -> (least in V, least in a training batch), floors only. The first split is V: its
    tasks come from the customers set aside for it, carry only tags of the official tasks (and branch tags), and about
    a third of them belong to variant groups; the training batches take no task of those customers.

    Greedy: whole variant groups are added one at a time, each time the one that brings the counts closest to the
    split's targets, by sum(weight x (count - target)^2 / (target + 1)) over the tags of the official tasks (target
    max(least, share x size)), their pairing with several writes (joint), the pairs of knowledge points of two or more
    official stories, the strata, the number of writes, the shape and the owned tags (share x size; the last three and
    "multi" weigh 3, the pairs 2, the rest 1), plus a cost of 50 per task a tag lacks below its least, so that the floors
    are met first. Then chosen groups are swapped for free ones of the same size while that brings the counts closer.
    The squared cost keeps a tag that comes cheaply with many tasks from running far past its share; the pairs keep
    knowledge points that the official tasks combine from being met only apart. V is filled first; the training batches
    then take turns, the one with the fewest tasks choosing next, so that scarce tasks spread evenly over them. Returns
    the splits and, per split, each target with the count reached."""
    share, strata = profile(domain, owners)

    def features(tags):
        return joint(tags) | {p for p in pairs(tags) if p in share} | {shape(tags)}

    groups = {}
    for tid, p in sorted(pool.items()):
        groups.setdefault(p["group"], []).append(tid)
    units = sorted(groups)
    base = sorted(set().union(*(pool[t]["tags"] for t in pool)))
    tags = sorted(set().union(*(features(pool[t]["tags"]) for t in pool))
                  | {t for t in share if t.startswith(("w2:", "pair:"))})
    feats = tags + [str(s) for s in sorted(strata)] + ["grouped"]
    col = {f: k for k, f in enumerate(feats)}
    X = np.zeros((len(units), len(feats)))
    for i, u in enumerate(units):
        for tid in groups[u]:
            for t in features(pool[tid]["tags"]):
                X[i, col[t]] += 1
            s = str(stratum(pool[tid]["tags"], domain))
            if s in col:
                X[i, col[s]] += 1
        X[i, col["grouped"]] = len(groups[u]) if len(groups[u]) > 1 else 0
    size = np.array([len(groups[u]) for u in units])
    customers = sorted({pool[groups[u][0]]["customer"] for u in units})
    owner = np.array([customers.index(pool[groups[u][0]]["customer"]) for u in units])
    order = np.random.default_rng(seed).permutation(len(units)) * 1e-9  # fixed tie-break
    for_v = np.array([pool[groups[u][0]]["v"] for u in units])
    names = list(sizes)
    target, weight, floor, ok = {}, {}, {}, {}
    branches = branches or {}
    for k, name in enumerate(names):
        n, (least, extra) = sizes[name], minima[name]
        target[name], weight[name], floor[name] = np.zeros(len(feats)), np.ones(len(feats)), np.zeros(len(feats))
        for t in tags:
            if t in branches:
                floor[name][col[t]], weight[name][col[t]] = branches[t][min(k, 1)], 0
            elif t.startswith(("w2:", "pair:")):
                target[name][col[t]] = share.get(t, 0) * n
                if t.startswith("pair:"):  # at 1 the single tags crowd out pairs the candidates have plenty of
                    weight[name][col[t]] = 2
            elif t.startswith(("owned_", "shape_")):
                target[name][col[t]] = share.get(t, 0) * n
                weight[name][col[t]] = 3
            elif t in share:
                target[name][col[t]] = max(least, share[t] * n)
                floor[name][col[t]] = least
                if t.startswith("writes_") or t == "multi":  # the shape of a task counts for more than one knowledge point
                    weight[name][col[t]] = 3
            else:  # a cell the official tasks do not have: only its least counts
                floor[name][col[t]], weight[name][col[t]] = extra or 0, 0
        for s, x in strata.items():
            target[name][col[str(s)]] = x * n
            weight[name][col[str(s)]] = 3
        if k == 0:
            target[name][col["grouped"]] = n / 3
        else:
            weight[name][col["grouped"]] = 0
        ok[name] = for_v if k == 0 else ~for_v
        if extra is None:
            ok[name] = ok[name] & (X[:, [col[t] for t in base if t not in share and t not in branches]].sum(1) == 0)

    def cost(name, c):
        return ((c - target[name]) ** 2 / (target[name] + 1) * weight[name]).sum(-1) + \
            50 * np.maximum(floor[name] - c, 0).sum(-1)

    free = np.ones(len(units), bool)
    count = {name: np.zeros(len(feats)) for name in names}
    load = {name: np.zeros(len(customers)) for name in names}  # tasks per customer
    chosen = {name: [] for name in names}
    total = dict.fromkeys(names, 0)
    for turn in (names[:1], names[1:]):
        while True:
            room = {m: free & ok[m] & (size <= sizes[m] - total[m]) & (load[m][owner] + size <= cap) for m in turn}
            open_ = [m for m in turn if total[m] < sizes[m] and room[m].any()]
            if not open_:
                break
            name = min(open_, key=lambda m: total[m] / sizes[m])
            gain = cost(name, count[name]) - cost(name, count[name] + X) - order
            gain[~room[name]] = -np.inf
            i = int(np.argmax(gain))
            chosen[name].append(i)
            count[name] += X[i]
            load[name][owner[i]] += size[i]
            total[name] += size[i]
            free[i] = False
        for name in turn:  # then swap a chosen unit for a free one of the same size while that lowers the cost
            for _ in range(5):
                swapped = False
                for k, i in enumerate(chosen[name]):
                    rest = count[name] - X[i]
                    load[name][owner[i]] -= size[i]
                    after = cost(name, rest + X) + order
                    after[~(free & ok[name] & (size == size[i]) & (load[name][owner] + size <= cap))] = np.inf
                    j = int(np.argmin(after))
                    if after[j] < cost(name, count[name]) - 1e-6:
                        chosen[name][k], count[name] = j, rest + X[j]
                        free[i], free[j] = True, False
                        swapped = True
                        i = j
                    load[name][owner[i]] += size[i]
                if not swapped:
                    break
    splits = {name: [tid for i in chosen[name] for tid in groups[units[i]]] for name in names}
    report = {name: {f: [round(float(target[name][col[f]]), 1), int(count[name][col[f]])] for f in feats}
              for name in names}
    return splits, report


def write(domain, db_path, made, scarce=(), owned_of=None, owners=None, branches=None):
    """Set V_CUSTOMERS of the customers aside for V, half of those in scarce (the few a kind of official task needs)
    among them, choose V (200 tasks) and three training batches (300 each), and write outputs(domain). owned_of:
    customer -> owned tag, added to the tags of their tasks; owners: official task id -> owned tag of its customer;
    branches: as in select."""
    generated, final, splits_path, report_path = outputs(domain)
    for path in (generated, splits_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    customers = sorted({u for *_, u, _ in made})
    rng = random.Random("validation")
    rare = [u for u in customers if u in scarce]
    for_v = set(rng.sample(rare, len(rare) // 2))
    rest = [u for u in customers if u not in scarce]
    for_v |= set(rng.sample(rest, round(V_CUSTOMERS * len(customers)) - len(for_v)))
    records = {t.id: {"task": t.model_dump(mode="json"), "source": "generated", "source_id": t.id, "db": db_path,
                      "tags": tags + ([owned_of(u)] if owned_of else []), "customer": u, "group": g}
               for _, t, _, tags, u, g in made}
    sizes = {"validation": 200, "round1": 300, "round2": 300, "round3": 300}
    minima = {"validation": (3, None), **{f"round{k}": (12, 5) for k in (1, 2, 3)}}
    splits, reached = select({i: {"tags": set(r["tags"]), "customer": r["customer"], "group": r["group"],
                                  "v": r["customer"] in for_v} for i, r in records.items()}, domain, sizes, minima,
                             owners=owners, branches=branches)
    with open(generated, "w") as f:
        for r in records.values():
            f.write(json.dumps(r) + "\n")
    with open(final, "w") as f:
        for ids in splits.values():
            for i in ids:
                f.write(json.dumps(records[i]) + "\n")
    splits_path.write_text(json.dumps(splits, indent=1) + "\n")
    report_path.write_text(json.dumps({
        "candidates": dict(Counter(family for family, *_ in made)),
        "candidate_tags": dict(Counter(t for r in records.values() for t in r["tags"])),
        "selection": reached,
        "specs": {t.id: [{**p.spec, "calls": p.calls, "forbidden": p.forbidden, "by_policy": p.by_policy}
                         for p in parts] for _, t, parts, *_ in made},
    }, indent=1))
    print(json.dumps(Counter(family for family, *_ in made), indent=1))
    print("candidates", len(made), {k: len(v) for k, v in splits.items()})


# ---------------------------------------------------------------- user behaviours
# Each changes how the user talks, never what the reference expects.

TRAITS = (("busy", "relaxing"), ("cautious", "curious"), ("confident", "insecure"), ("creative", "logical"),
          ("dependent", "independent"), ("direct", "polite"), ("flexible", "rigid"), ("happy", "sad"),
          ("impatient", "patient"), ("messy", "organized"), ("optimistic", "pessimistic"), ("outgoing", "shy"))
REACTIVE = ("You only respond to what the agent asks and add nothing of your own.",
            "Only answer what the agent asks; do not volunteer anything else.",
            "Give each piece of information only when the agent asks for it.")
EMOTION = ("You are upset and short on patience.", "You are stressed and want this sorted out quickly.",
           "You are frustrated, but you cooperate with the agent.", "You are anxious and keep your answers short.")
# a transfer ends the call: with other requests, the one that leads to it comes last
TRANSFER_LAST = ("Bring this up only after your other requests have been dealt with.",
                 "Leave this for last, once everything else has been handled.")
NO_TRANSFER = ("Under no circumstances do you want to be transferred to another agent.",
               "You refuse to be transferred to a human agent.",
               "If the agent offers to transfer you, decline and ask it to help you itself.")
OFFTOPIC = ("Before getting to your request, ask the agent to tell you a joke; if it declines, move on.",
            "End each of your messages with 'Thanks for bearing with me.'",
            "You like to chat: add a short remark about your day to some of your messages.",
            "Write casually, with an occasional typo.")


def traits(rng):
    """A persona line with two to five traits, never both of a pair, or None for about a third of the users."""
    k = rng.choice((0, 0, 2, 3, 4, 5))
    return f"You are {', '.join(rng.choice(pair) for pair in rng.sample(TRAITS, k))}." if k else None


def behaviours(rng, rates):
    """The behaviours a task's user shows, each drawn with its rate: name -> probability."""
    return [b for b, p in sorted(rates.items()) if rng.random() < p]


def line(rng, behaviour):
    return rng.choice({"reactive": REACTIVE, "emotion": EMOTION, "no_transfer": NO_TRANSFER,
                       "offtopic": OFFTOPIC}[behaviour])
