"""Evidence pack of one failed conversation, built by script: the agent's database writes against the reference ones,
the database differences, which skills it loaded and when, where it erred, and the conversation. The labels only
describe differences from the reference; why the agent failed is for the editor to say.
"""
import json
from functools import cache

from pydantic import ValidationError
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.domains.airline.data_model import FlightInfo, Passenger, Payment
from tau2.registry import registry

from rsi.benchmarks.run import environment

# write tool -> category (skill) of the domain's plugin
SKILL = {
    "cancel_pending_order": "cancel", "modify_pending_order_address": "modify", "modify_pending_order_payment": "modify",
    "modify_pending_order_items": "modify", "return_delivered_order_items": "return",
    "exchange_delivered_order_items": "exchange", "modify_user_address": "user-address",
    "book_reservation": "book", "update_reservation_flights": "modify", "update_reservation_baggages": "modify",
    "update_reservation_passengers": "modify", "cancel_reservation": "cancel", "send_certificate": "compensation",
}
# what a write acts on: an order, a reservation, else the user (modify_user_address, booking, certificates)
TARGET = ("order_id", "reservation_id", "user_id")
# the models tau2's airline write tools build from the dicts they are given, as in FlightInfo(**flight)
MODELS = {"flights": FlightInfo, "passengers": Passenger, "payment_methods": Payment}


def target(args):
    return next((args[k] for k in TARGET if k in args), None)


@cache
def tools():
    """Both domains' write tools by name."""
    return {n: t for d in ("retail", "airline") for n, t in registry.get_env_constructor(d)().tools.get_tools().items()
            if n in SKILL}


def used(name, args):
    """A write's arguments as its tool reads them. The tools build pydantic models from the dicts they get, so keys the
    models lack drop out and numbers written as strings become numbers: a flight given with its origin or price, or an
    amount of "560", is the same write."""
    try:
        args = tools()[name].params.model_validate(args).model_dump(mode="json")
        return {k: [MODELS[k].model_validate(x).model_dump(mode="json") for x in v] if k in MODELS else v
                for k, v in args.items()}
    except ValidationError:
        return args


def writes(sim):
    """(turn, tool, arguments as the tool read them) of every write the environment accepted, in order."""
    ok = {m.id for m in sim.messages if isinstance(m, ToolMessage) and not m.error}
    return [(m.turn_idx, c.name, used(c.name, c.arguments)) for m in sim.messages if isinstance(m, AssistantMessage)
            for c in m.tool_calls or [] if c.name in SKILL and c.id in ok]


def diff(task, sim):
    """The agent's writes against the reference: paired on the same tool and target first, then on the same tool in
    order; what is left is missing or extra. Agent-side mismatches come by turn, missing writes last in reference order."""
    gold = [(a.name, used(a.name, a.arguments)) for a in task.evaluation_criteria.actions or [] if a.name in SKILL]
    done = writes(sim)
    free = list(range(len(done)))
    out, left = [], []
    for name, args in gold:
        j = next((j for j in free if done[j][1] == name and target(done[j][2]) == target(args)), None)
        if j is None:
            left.append((name, args))
            continue
        free.remove(j)
        if done[j][2] != args:
            out.append({"turn": done[j][0], "label": "wrong arguments", "tool": name, "reference": args,
                        "agent": done[j][2]})
    for name, args in left:
        j = next((j for j in free if done[j][1] == name), None)
        if j is None:
            out.append({"turn": None, "label": "missing", "tool": name, "reference": args})
        else:
            free.remove(j)
            out.append({"turn": done[j][0], "label": "wrong target", "tool": name, "reference": args,
                        "agent": done[j][2]})
    out += [{"turn": done[j][0], "label": "extra", "tool": done[j][1], "agent": done[j][2]} for j in free]
    return sorted(out, key=lambda d: (d["turn"] is None, d["turn"] or 0))


def category(task, sim):
    """Category a failed conversation is grouped under: that of the first write that differs from the reference; with
    all writes right, that of the reference's first write; else 'general'."""
    d = diff(task, sim)
    if d:
        return SKILL[d[0]["tool"]]
    gold = [a.name for a in task.evaluation_criteria.actions or [] if a.name in SKILL]
    return SKILL[gold[0]] if gold else "general"


def loads(sim):
    """(turn, skill name) of every load_skill call, in order."""
    return [(m.turn_idx, load["name"]) for m in sim.messages if isinstance(m, AssistantMessage) and m.raw_data
            for load in m.raw_data.get("skill_loads", [])]


def thinking(message):
    try:
        return (message.raw_data["choices"][0]["message"].get("reasoning_content") or "").strip()
    except (TypeError, KeyError, IndexError):
        return ""


def field_diff(a, b, path=""):
    """(path, a, b) of every field where two JSON-like values differ. Lists are compared whole: an order's items
    listed in another order would otherwise show as many unrelated differences."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [x for k in sorted(set(a) | set(b), key=str) for x in field_diff(a.get(k), b.get(k), f"{path}.{k}")]
    return [] if a == b else [(path, a, b)]


def db_diff(domain, record, task, sim):
    """Field differences of the records where the agent's database differs from the reference one."""
    agent, gold = environment(domain, record), environment(domain, record)
    agent.set_state(initialization_data=None, initialization_actions=None, message_history=list(sim.messages),
                    strict=False)
    for a in task.evaluation_criteria.actions or []:
        try:
            gold.make_tool_call(tool_name=a.name, requestor=a.requestor, **a.arguments)
        except Exception:  # the evaluator ignores a failing reference action the same way
            pass
    g, p = gold.tools.db.model_dump(), agent.tools.db.model_dump()
    out = []
    for table in sorted(g):
        if isinstance(g[table], dict):
            for key in sorted(set(g[table]) | set(p[table]), key=str):
                out += [f"{table}[{key}]{path}: reference {json.dumps(u, default=str)}, "
                        f"agent {json.dumps(v, default=str)}"
                        for path, u, v in field_diff(g[table].get(key), p[table].get(key))]
    return out


def error_turns(sim, mismatches, unsaid):
    """Turns where the agent went wrong: the first wrong or extra write; for missing writes the transfer, else the
    agent's last message; for information not stated, the agent's first replies after its last write."""
    agent = [m for m in sim.messages if isinstance(m, AssistantMessage)]
    turns = [next(d["turn"] for d in mismatches if d["turn"] is not None)] if any(
        d["turn"] is not None for d in mismatches) else []
    if any(d["label"] == "missing" for d in mismatches):
        transfer = [m.turn_idx for m in agent for c in m.tool_calls or [] if c.name == "transfer_to_human_agents"]
        turns.append(transfer[0] if transfer else agent[-1].turn_idx)
    if unsaid:
        last = max((w[0] for w in writes(sim)), default=-1)
        turns += [m.turn_idx for m in agent if m.turn_idx > last and m.has_text_content()][:3]
    return sorted(set(turns))


def conversation(sim, think=()):
    """(turn, line) of the whole conversation without thinking, except the agent's thinking at the given turns.
    load_skill calls are shown as calls only: the plugin is given once, at the top of the analysis call."""
    out = []
    for m in sim.messages:
        t = m.turn_idx
        if isinstance(m, UserMessage):
            out.append((t, f"user: {m.content}"))
        elif isinstance(m, ToolMessage):
            out.append((t, f"tool{' (error)' if m.error else ''}: {m.content}"))
        elif isinstance(m, AssistantMessage):
            for load in (m.raw_data or {}).get("skill_loads", []):
                out.append((t, f"agent calls load_skill({json.dumps({'skill_name': load['name']})})"))
            if t in think and thinking(m):
                out.append((t, f"agent thinking: {thinking(m)}"))
            if m.has_text_content():
                hidden = " (sent together with a tool call, so the user never saw it)" if m.tool_calls else ""
                out.append((t, f"agent{hidden}: {m.content.strip()}"))
            for c in m.tool_calls or []:
                out.append((t, f"agent calls {c.name}({json.dumps(c.arguments)})"))
    return out


def pack(domain, record, task, sim, successes):
    """Evidence pack of one failed conversation; successes: (successful trials, trials) of this task in this run."""
    mismatches = diff(task, sim)
    unsaid = [c.info for c in sim.reward_info.communicate_checks or [] if not c.met]
    errors = error_turns(sim, mismatches, unsaid)
    near = {t + i for t in errors for i in range(-2, 3)}
    transferred = any(c.name == "transfer_to_human_agents" for m in sim.messages if isinstance(m, AssistantMessage)
                      for c in m.tool_calls or [])
    if any(d["label"] == "missing" for d in mismatches) and not transferred:
        # a write that never happened has no turn: its error turn is the agent's last message, so the thinking at
        # every turn is kept, to show where the agent let the write go
        near = {m.turn_idx for m in sim.messages}
    loaded = loads(sim)
    routing = []
    for turn, tool, _ in writes(sim):
        skill = f"{domain}:{SKILL[tool]}"
        before = {n for t, n in loaded if t <= turn}
        routing.append(f"turn {turn}, {tool} (skill {skill}): entry {domain} loaded before it: "
                       f"{'yes' if domain in before else 'no'}; {skill} loaded before it: "
                       f"{'yes' if skill in before else 'no'}")
    for d in mismatches:
        if d["label"] == "missing":
            skill = f"{domain}:{SKILL[d['tool']]}"
            routing.append(f"missing {d['tool']} (skill {skill}): {skill} loaded in the conversation: "
                           f"{'yes' if any(n == skill for _, n in loaded) else 'no'}")
    return {
        "task_id": task.id, "trial": sim.trial,
        "outcome": {"reward": sim.reward_info.reward, "db_match": bool(sim.reward_info.db_check and
                                                                       sim.reward_info.db_check.db_match),
                    "unsaid": unsaid, "successes": list(successes)},
        "reference": [(a.name, used(a.name, a.arguments)) for a in task.evaluation_criteria.actions or []
                      if a.name in SKILL],
        "writes": mismatches,
        "db_diff": db_diff(domain, record, task, sim),
        "loads": loaded,
        "routing": routing,
        "other_domain_loads": [n for _, n in loaded if n.split(":")[0] != domain],
        "error_turns": errors,
        "conversation": conversation(sim, near),
        "error_thinking_turns": sorted(near),
    }


def write_line(d):
    where = f"turn {d['turn']}: " if d["turn"] is not None else ""
    ref = f"; reference arguments {json.dumps(d['reference'])}" if "reference" in d else ""
    got = f"; agent arguments {json.dumps(d['agent'])}" if "agent" in d else ""
    return f"- {where}{d['label']}: {d['tool']}{got}{ref}"


def text(p, contrast=None, level=0):
    """The pack as the analysis call reads it. level trims, in order: 1 drops the successful run shown for contrast,
    2 keeps thinking only at the error turns, 3 keeps only the 4 messages around each error turn."""
    o = p["outcome"]
    lines = [f"### Failed run: task {p['task_id']}, trial {p['trial']}",
             f"Reward {o['reward']:g}. Database matches the reference: {'yes' if o['db_match'] else 'no'}."]
    if o["unsaid"]:
        lines.append("Information the agent had to tell the user but never stated: "
                     + ", ".join(json.dumps(u) for u in o["unsaid"]) + ".")
    lines.append(f"This task succeeded in {o['successes'][0]} of {o['successes'][1]} runs of the current plugin.")
    # the reference in order: one write can change what a later one may do, as a status change does
    lines += ["", "#### Database writes compared with the reference", "Reference writes, in order:"]
    lines += [f"{i}. {name}({json.dumps(args)})" for i, (name, args) in enumerate(p["reference"], 1)] or ["none"]
    lines.append("Where the agent's writes differ:")
    lines += [write_line(d) for d in p["writes"]] or ["- all writes match the reference"]
    lines += ["", "#### Database differences (reference vs agent)"] + [f"- {x}" for x in p["db_diff"] or ["none"]]
    lines += ["", "#### Skill loading"]
    lines.append("Loaded: " + (", ".join(f"turn {t} {n}" for t, n in p["loads"]) or "nothing"))
    lines += [f"- {r}" for r in p["routing"]]
    lines.append("Skills of another domain loaded: " + (", ".join(p["other_domain_loads"]) or "none"))
    lines += ["", "#### Where the agent went wrong",
              "Turns: " + (", ".join(map(str, p["error_turns"])) or "not located"), "", "#### Conversation"]
    keep = set(p["error_turns"]) if level >= 2 else set(p["error_thinking_turns"])
    window = {t + i for t in p["error_turns"] for i in range(-4, 5)} if level >= 3 and p["error_turns"] else None
    skipped = False
    for t, line in p["conversation"]:
        if line.startswith("agent thinking:") and t not in keep:
            continue
        if window is not None and t not in window:
            if not skipped:
                lines.append("[...]")
            skipped = True
            continue
        skipped = False
        lines.append(f"[{t}] {line}")
    if contrast and level < 1:
        lines += ["", f"#### A successful run of the same task (trial {contrast['trial']}), for contrast"]
        lines += [f"[{t}] {line}" for t, line in contrast["conversation"]]
    return "\n".join(lines)
