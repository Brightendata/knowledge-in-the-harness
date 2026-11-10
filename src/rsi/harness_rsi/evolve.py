"""Plugin evolution of one domain in one round. The failure pool of the collection is walked once, in steps. Each step
reruns the current plugin on its failed and control tasks, has the editor M(0) analyse each failed task, merges the
proposals into at most L rule edits, and keeps the edits only if the candidate plugin, run on the same tasks with the
same seed, succeeds at least m times more than the current plugin did. A rule that made a task fail in that rerun (see
harmful) comes out first; the rest of the step is kept if the candidate passed anyway, else run again on its own. An
Attempt reads artifacts/logs/r<k>/collect/<domain>.json and writes artifacts/logs/r<k>/harness_rsi/<domain>/, or
<domain>_retry/ for the retry after a failed gate.
"""
import json
import random
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from math import ceil

from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import Results
from tau2.metrics.agent_metrics import is_successful

from rsi.agent.plugin import apply, documents, manifest, rules
from rsi.benchmarks.run import OTHER, POOL, ROOT, config, environment, records, run, run_seed, start_system
from rsi.harness_rsi import editor, evidence

# plugins/ of an attempt: the start, then each candidate in order, kept or not (plugins/versions.json says which)
START = "v00"


def now():
    return datetime.now().isoformat(timespec="seconds")


def failed(s):
    return s.reward_info is not None and not is_successful(s.reward_info.reward)


def succeeded(s):
    return s.reward_info is not None and is_successful(s.reward_info.reward)


def by_task(results):
    out = defaultdict(list)
    for s in sorted(results.simulations, key=lambda s: (s.task_id, s.trial)):
        out[s.task_id].append(s)
    return out


def group(failing, per_step, cap, rng):
    """Failing task ids of each step. failing: category -> task ids. Each category is shuffled and cut into near-equal
    pieces of at most cap tasks. The pieces of the categories cut into the most pieces are placed first, as they have
    the fewest steps left to go to, larger pieces first; each goes to the step with the fewest tasks that has none of
    its category yet. A step so holds at most cap tasks of a category, about per_step in all."""
    n = max([ceil(sum(map(len, failing.values())) / per_step)] + [ceil(len(v) / cap) for v in failing.values()])
    pieces = []
    for cat in sorted(failing):
        ids = sorted(failing[cat])
        rng.shuffle(ids)
        k = ceil(len(ids) / cap)
        pieces += [(cat, ids[i::k]) for i in range(k)]
    count = {cat: ceil(len(v) / cap) for cat, v in failing.items()}
    steps, cats = [[] for _ in range(n)], [set() for _ in range(n)]
    for cat, piece in sorted(pieces, key=lambda p: (-count[p[0]], -len(p[1]))):
        i = min((i for i in range(n) if cat not in cats[i]), key=lambda i: len(steps[i]))
        steps[i] += piece
        cats[i].add(cat)
    return steps


def controls(good, steps, per_step):
    """per_step control task ids for each of `steps` steps, drawn without replacement across steps and again from the
    start once all were used: a model that fails more leaves fewer all-correct tasks than the steps need. A step never
    gets the same task twice."""
    c = min(per_step, len(good))
    cycle = good * ceil(steps * c / len(good)) if c else []
    return [cycle[i * c:(i + 1) * c] for i in range(steps)]


def plan(collected, tasks, ev, seed):
    """Steps of (failing task ids, control task ids) from the collection. A task is failing when a trial failed (by
    the category of its first failed trial), a control when every trial succeeded; a task tau2 gave up on in some
    trials and that never failed otherwise is neither."""
    failing, good = defaultdict(list), []
    for tid, sims in sorted(by_task(collected).items()):
        bad = [s for s in sims if failed(s)]
        if bad:
            failing[evidence.category(tasks[tid], bad[0])].append(tid)
        elif all(succeeded(s) for s in sims):
            good.append(tid)
    rng = random.Random(seed)
    steps = group(failing, ev["failing_per_step"], ev["category_cap"], rng)
    rng.shuffle(good)
    return list(zip(steps, controls(good, len(steps), ev["controls_per_step"])))


def materials(domain, recs, tasks, current, failing, excluded):
    """One unit per failed task the current plugin failed again: its first failed run, plus a second failed run whose
    write mismatches differ when it never succeeded, or its first successful run for contrast."""
    runs, units = by_task(current), []
    for tid in failing:
        if tid in excluded:
            continue
        sims = runs[tid]
        bad = [s for s in sims if failed(s)]
        if not bad:
            continue
        good = next((s for s in sims if succeeded(s)), None)
        chosen = [bad[0]]
        if good is None:
            def sig(s):
                return [(d["label"], d["tool"]) for d in evidence.diff(tasks[tid], s)]
            second = next((s for s in bad[1:] if sig(s) != sig(bad[0])), None)
            chosen += [second] if second else []
        tally = (sum(succeeded(s) for s in sims), len(sims))
        units.append({"task_id": tid, "packs": [evidence.pack(domain, recs[tid], tasks[tid], s, tally) for s in chosen],
                      "contrast": {"trial": good.trial, "conversation": evidence.conversation(good)} if good else None})
    return units


def support_ids(support, proposals):
    """The task ids an edit's support names. The merge may cite a proposal by the analysis id that heads it, or a task
    whose id joins several parts with '+' by one of its parts, which counts once."""
    task_of = {a["id"]: a["task_id"] for a in proposals}
    task_of |= {part: a["task_id"] for a in proposals for part in a["task_id"].split("+")}
    return sorted({task_of.get(s, s) for s in support})


def blocked_edits(record):
    """The edits of a merge record that the editor's check blocked, or that the rerun took out (see harmful), with the
    problems found."""
    return [{"edit": d["edit"], "problems": d["problems"]}
            for d in record["dropped_by_checks"] + record.get("removed_after_retest", []) if "problems" in d]


def cited(sim, rule_id):
    """Whether the agent's thinking in a conversation, in its load_skill calls too, names the rule ('modify-3' of
    'airline:modify-3' counts, 'modify-30' does not)."""
    name = re.compile(r"(?<![\w-])" + re.escape(rule_id.split(":")[-1]) + r"(?!\w)")
    for m in sim.messages:
        if isinstance(m, AssistantMessage) and m.raw_data:
            for raw in [m.raw_data] + [load["response"] or {} for load in m.raw_data.get("skill_loads", [])]:
                try:
                    if name.search(raw["choices"][0]["message"].get("reasoning_content") or ""):
                        return True
                except (TypeError, KeyError, IndexError):
                    pass
    return False


def harmful(current, candidate, rule_ids, drop, excluded):
    """Rule id -> the tasks it made fail: tasks whose successes fell by at least drop from the current plugin's rerun to
    the candidate's, where the agent's thinking names the rule in at least two of the failed candidate runs. Naming a
    rule is not causing a failure: the agent recites the rules of a task's skill whether or not it then fails, and a
    task falls by a few runs by chance alone, so only a large fall counts."""
    cur, out = by_task(current), defaultdict(list)
    for tid, sims in by_task(candidate).items():
        a, b = sum(map(succeeded, cur[tid])), sum(map(succeeded, sims))
        if tid in excluded or a - b < drop:
            continue
        bad = [s for s in sims if not succeeded(s)]
        for rid in rule_ids:
            n = sum(cited(s, rid) for s in bad)
            if n >= 2:
                out[rid].append({"task": tid, "current": a, "candidate": b, "failed": len(bad), "naming": n})
    return dict(out)


def origin(logs, domain, rule_id, text):
    """The edit that wrote a rule's current text, in any round's evolution under logs (artifacts/logs), and the
    analyses of the tasks behind it; None when no record has it."""
    want = " ".join(text.split())
    for path in logs.glob(f"r*/harness_rsi/{domain}*/steps/s*/merge.json"):
        for e in json.loads(path.read_text())["edits"]:
            if e.get("written") == rule_id and " ".join(e["text"].split()) == want:
                with open(path.parent / "analysis.jsonl") as f:
                    return e, [a for a in map(json.loads, f) if a["task_id"] in e["support"] and a.get("output")]
    return None


def undo(edits, plugin_dir):
    """Edits that take the given edits back out of a version made from plugin_dir: a rule they added is deleted, a rule
    they replaced gets its text in plugin_dir back."""
    old = {i: t for path in documents(plugin_dir).values() for i, t in rules(path)}
    return [{"op": "DELETE", "rule_id": e["written"]} if e["op"] == "ADD" else
            {"op": "REPLACE", "rule_id": e["written"], "text": old[e["written"]]} for e in edits]


def analysis_input(policy, tools, view, blocked, unit, budget):
    """The analysis call's input, trimmed level by level (evidence.text) until it fits the token budget, the evidence in
    it, and the text of its failed runs alone."""
    for level in range(4):
        packs = "\n\n".join(evidence.text(p, unit["contrast"] if i == 0 else None, level)
                            for i, p in enumerate(unit["packs"]))
        user = editor.ANALYST_INPUT.format(policy=policy, tools=tools, plugin=view, blocked=blocked, evidence=packs)
        if len(user) / 3 <= budget:  # 3 characters per token, a cautious estimate that needs no tokenizer
            break
    return user, packs, "\n\n".join(evidence.text(p, None, level) for p in unit["packs"])


class Attempt:
    """One attempt of plugin evolution for a domain in a round, and the state its directory records."""

    def __init__(self, domain, round_, retry, logs=None):
        """logs: the round's log directory, artifacts/logs/r<k> unless given (scripts/smoke_harness.py uses its own)."""
        self.domain, self.round, self.attempt = domain, round_, int(retry)
        self.cfg = config()
        self.ev = self.cfg["evolve"]
        logs = logs or ROOT / f"artifacts/logs/r{round_}"
        self.out = logs / "harness_rsi" / f"{domain}{'_retry' if retry else ''}"
        self.collection = logs / "collect" / f"{domain}.json"
        self.collected = Results.load(self.collection)
        self.tasks = {t.id: t for t in self.collected.tasks}
        self.recs = {r["task"]["id"]: r for r in records(domain, f"round{round_}")}
        system = start_system(round_, logs.parent)
        self.start, self.other = system[domain]["plugin"], system[OTHER[domain]]["plugin"]
        # the agent's model, M(k-1); the editor stays the base model ([editor])
        self.model = system[domain]["model"]
        self.steps = plan(self.collected, self.tasks, self.ev, self.ev["shuffle_seed"][self.attempt])
        env = environment(domain, next(iter(self.recs.values())))
        self.policy, self.tools = env.get_policy(), editor.tools_text(env.get_tools())
        dbs = {ROOT / r["db"] for line in open(POOL[domain]) for r in [json.loads(line)]}
        self.keys = editor.db_keys(sorted(dbs))
        self.policy_amounts = editor.amounts(self.policy)

    def emit(self, event):
        with open(self.out / "events.jsonl", "a") as f:
            f.write(json.dumps({"time": now(), **event}) + "\n")

    def setup(self):
        self.out.mkdir(parents=True, exist_ok=True)
        if not (self.out / "plugins" / START).exists():
            shutil.copytree(ROOT / self.start, self.out / "plugins" / START)
        if not (self.out / "config.json").exists():
            (self.out / "config.json").write_text(json.dumps({
                "round": self.round, "domain": self.domain, "attempt": self.attempt,
                "tasks": f"data/splits/{self.domain}.json:round{self.round}",
                "collection": str(self.collection.relative_to(ROOT)),
                "start_plugin": self.start, "other_plugin": self.other, "model": self.model,
                "steps": [{"step": s, "failing": f, "controls": c, "seed": run_seed(self.round, 1, self.attempt, s)}
                          for s, (f, c) in enumerate(self.steps, 1)],
                # the settings set per attempt hold this attempt's value; the machine's ([server], [run]) are left out
                "settings": {**{k: v for k, v in self.cfg.items() if k not in ("server", "run")}, "evolve": {
                    k: v[self.attempt] if k in ("max_edits", "min_support", "shuffle_seed") else v
                    for k, v in self.ev.items()}}}, indent=2) + "\n")

    def state(self):
        """Events of finished steps (an unfinished step's are dropped, and the step is redone), the current version,
        the number of candidates so far, and the edits the check blocked, in step order."""
        path = self.out / "events.jsonl"
        events = [json.loads(line) for line in open(path)] if path.exists() else []
        done = {e["step"] for e in events if e["event"] == "retest" or (e["event"] == "merge" and not e["edits"])}
        events = [e for e in events if "step" not in e or e["step"] in done]
        path.write_text("".join(json.dumps(e) + "\n" for e in events))
        current, buffer = START, []
        for e in events:
            if e["event"] == "merge":
                record = json.loads((self.out / "steps" / f"s{e['step']:02d}" / "merge.json").read_text())
                if blocked := blocked_edits(record):
                    buffer.append({"step": e["step"], "blocked": blocked})
            elif e["event"] == "retest" and e["kept"]:
                current = e["candidate"]
        versions = sum(1 for e in events if (e["event"] == "merge" and e.get("candidate"))
                       or (e["event"] == "harm" and e.get("version")))
        return events, done, current, versions, buffer

    def system(self, version):
        return [str((self.out / "plugins" / version).relative_to(ROOT)), self.other]

    def index(self):
        """plugins/versions.json from events.jsonl: for each version, the step that made it and the version it was made
        from, the steps whose edits it holds on top of the start, the rules taken out of it after its rerun (without),
        and whether its retest kept it."""
        made, steps = {START: {"start": self.start, "holds": []}}, {}
        for e in map(json.loads, open(self.out / "events.jsonl")):
            if e["event"] == "step":
                steps[e["step"]] = e["plugin"]
            elif e["event"] == "merge" and e.get("candidate"):
                parent = steps[e["step"]]
                made[e["candidate"]] = {"step": e["step"], "from": parent,
                                        "holds": made[parent]["holds"] + [f"s{e['step']:02d}"], "summary": e["summary"]}
            elif e["event"] == "harm" and e.get("version"):
                made[e["version"]] = {**made[e["candidate"]], "without": e["removed"]}
            elif e["event"] == "retest":
                made[e["candidate"]].update(kept=e["kept"], successes=e["successes"])
                if "pruned" in e:
                    made[e["candidate"]]["rerun"] = e["rerun"]
                    made[e["pruned"]["candidate"]].update(kept=False, successes=e["pruned"]["successes"])
        (self.out / "plugins/versions.json").write_text(json.dumps(made, indent=2) + "\n")

    def analyse(self, units, current, buffer, step, seed, step_dir):
        plugin = self.out / "plugins" / current
        view, blocked, reply = editor.plugin_view(plugin), editor.blocked_text(buffer), editor.analysis_schema(plugin)
        evid, analyses = [], []
        for n, u in enumerate(units, 1):
            ids = [f"e{len(evid) + i + 1:02d}" for i in range(len(u["packs"]))]
            evid += [{"id": i, **p} for i, p in zip(ids, u["packs"])]
            analyses.append({"id": f"a{n:02d}", "task_id": u["task_id"], "evidence": ids, "unit": u})
        with open(step_dir / "evidence.jsonl", "w") as f:
            f.writelines(json.dumps(e) + "\n" for e in evid)

        def one(a):
            a["start"] = now()
            user, packs, failed = analysis_input(self.policy, self.tools, view, blocked, a["unit"],
                                                 self.cfg["editor"]["input_tokens"])
            # a proposal whose quotes do not match the evidence is analysed again: the editor often misquotes a right
            # diagnosis by a few words
            for k in range(self.cfg["editor"]["quote_retries"] + 1):
                a["output"] = editor.ask(self.cfg, editor.ANALYST, user, reply,
                                         seed * 100 + int(a["id"][1:]) + 1000 * k, a["id"] + (f"r{k}" if k else ""),
                                         step_dir / "editor_raw.jsonl")
                a["dropped"] = ("no valid reply" if a["output"] is None
                                else editor.quote_problem(a["output"], packs, failed))
                if a["output"] is None or a["dropped"] is None:
                    break
            a["end"] = now()
            return a

        with ThreadPoolExecutor(self.cfg["editor"]["workers"]) as pool:
            analyses = list(pool.map(one, analyses))
        with open(step_dir / "analysis.jsonl", "w") as f:
            f.writelines(json.dumps({k: v for k, v in a.items() if k != "unit"}) + "\n" for a in analyses)
        return analyses

    def merge(self, analyses, current, buffer, seed, step_dir):
        """Merged edits that passed the code checks and the editor's check, the ones that did not, and the merge's
        summary."""
        proposals = [a for a in analyses if a["output"] and a["output"]["op"] != "NONE" and not a["dropped"]]
        record = {"analyses": [a["id"] for a in proposals], "start": now()}
        output = None
        if proposals:
            plugin = self.out / "plugins" / current
            user = editor.MERGE_INPUT.format(policy=self.policy, tools=self.tools, plugin=editor.plugin_view(plugin),
                                             blocked=editor.blocked_text(buffer), n=len(proposals),
                                             proposals=editor.proposals_text(proposals))
            budget, support = self.ev["max_edits"][self.attempt], self.ev["min_support"][self.attempt]
            output = editor.ask(self.cfg, editor.merge_system(budget, support), user, editor.merge_schema(plugin),
                                seed * 100, "merge", step_dir / "editor_raw.jsonl")
        kept, dropped, touched = [], [], set()
        analyzed = {a["task_id"] for a in analyses}
        unclear = {a["task_id"] for a in proposals if a["output"]["relation"] == "unclear"}
        for e in (output or {}).get("edits", []):
            e = {**e, "support": support_ids(e["support"], proposals)}
            problem = editor.edit_problem(e, self.out / "plugins" / current, analyzed, unclear,
                                          self.ev["min_support"][self.attempt], self.keys, self.policy_amounts, touched)
            if problem is None and len(kept) == self.ev["max_edits"][self.attempt]:
                problem = "over the edit budget"
            if problem:
                dropped.append({"edit": e, "reason": problem})
            else:
                kept.append(e)
                touched.add(e["rule_id"])
        # the editor checks each edit that passed the code checks against the policy, before any is tested
        if kept:
            plugin = self.out / "plugins" / current
            jobs = [(i, j) for i in range(len(kept)) for j in range(editor.CHECKS)]

            def check(job):
                i, j = job
                return editor.check(self.cfg, self.policy, self.tools, plugin, kept[i], proposals,
                                    seed * 100 + 50 + 10 * j + i, f"check{i + 1}.{j + 1}", step_dir / "editor_raw.jsonl")

            with ThreadPoolExecutor(len(jobs)) as pool:
                replies = list(pool.map(check, jobs))
            checks = [replies[i * editor.CHECKS:(i + 1) * editor.CHECKS] for i in range(len(kept))]
            dropped += [{"edit": e, "reason": "check " + " | ".join(f"{v['problem']}: {v['example']}" for v in c),
                         "problems": [v["problem"] for v in c]}
                        for e, c in zip(kept, checks) if editor.blocked(c, e["op"])]
            passed = [(e, c) for e, c in zip(kept, checks) if not editor.blocked(c, e["op"])]
            rejected = self.review(passed, proposals, plugin, seed, step_dir)
            dropped += rejected
            kept = [e for e, _ in passed if not any(r["edit"] is e for r in rejected)]
        record.update(end=now(), output=output, edits=kept, dropped_by_checks=dropped,
                      summary=(output or {}).get("summary", ""))
        return record, proposals

    def review(self, passed, proposals, plugin, seed, step_dir):
        """The checked REPLACEs that keep under [evolve] review_below of their rule's words are reviewed against the
        old text (editor.REVIEW), editor.REVIEWS times each; returns the ones the reviews do not keep, as dropped
        edits. They are not listed to later steps as blocked: a rewrite judged wrongly may come back."""
        old = {i: t for path in documents(plugin).values() for i, t in rules(path)}
        rewrites = [(e, c) for e, c in passed if e["op"] == "REPLACE"
                    and editor.kept_share(old[e["rule_id"]], e["text"]) < self.ev["review_below"]]
        if not rewrites:
            return []
        logs = self.out.parent.parent.parent

        def one(job):
            i, j = job
            e, c = rewrites[i]
            runs = "\n\n".join(evidence.text(a["unit"]["packs"][0], None, 3) for a in proposals
                               if a["task_id"] in set(e["support"]))
            return editor.review(self.cfg, self.policy, self.tools, plugin, e, proposals,
                                 origin(logs, self.domain, e["rule_id"], old[e["rule_id"]]), runs, c,
                                 seed * 100 + 70 + 10 * j + i, f"review{i + 1}.{j + 1}", step_dir / "editor_raw.jsonl")

        jobs = [(i, j) for i in range(len(rewrites)) for j in range(editor.REVIEWS)]
        with ThreadPoolExecutor(len(jobs)) as pool:
            replies = list(pool.map(one, jobs))
        out = []
        for i, (e, _) in enumerate(rewrites):
            verdicts = replies[i * editor.REVIEWS:(i + 1) * editor.REVIEWS]
            if not editor.kept_by_review(verdicts):
                out.append({"edit": e, "reason": "review: only the old text right in: " + " | ".join(
                    "no valid reply" if v is None else "; ".join(c["case"] for c in v["cases"] if c["right"] == "old")
                    for v in verdicts), "review": verdicts})
        return out

    def step(self, s, failing, controls, current, versions, buffer):
        step_dir = self.out / "steps" / f"s{s:02d}"
        shutil.rmtree(step_dir, ignore_errors=True)
        step_dir.mkdir(parents=True)
        self.emit({"event": "step", "step": s, "plugin": current, "failing": failing, "regression": controls})
        seed = run_seed(self.round, 1, self.attempt, s)
        recs = [self.recs[i] for i in failing + controls]
        model, user, trials = self.model, self.cfg["user"]["train"], self.ev["trials"]
        cur = run(self.domain, recs, self.system(current), model, user, trials, seed, step_dir / "current.json")
        units = materials(self.domain, self.recs, self.tasks, cur, failing, set(self.ev["excluded"]))
        analyses = self.analyse(units, current, buffer, s, seed, step_dir)
        record, proposals = self.merge(analyses, current, buffer, seed, step_dir)
        version = None
        if record["edits"]:
            version = f"v{versions + 1:02d}"
            shutil.rmtree(self.out / "plugins" / version, ignore_errors=True)
            shutil.copytree(self.out / "plugins" / current, self.out / "plugins" / version)
            # rule numbers go on from the highest any earlier version used, a rejected one included
            used = {}
            for p in (self.out / "plugins").iterdir():
                if p.is_dir() and p.name != version:
                    for name, n in manifest(p).get("last_rule", {}).items():
                        used[name] = max(used.get(name, 0), n)
            for e, rule_id in zip(record["edits"], apply(self.out / "plugins" / version, record["edits"], used)):
                e["written"] = rule_id
        (step_dir / "merge.json").write_text(json.dumps(record, indent=2) + "\n")
        dropped = sum(bool(a["dropped"]) and a["output"] is not None and a["output"]["op"] != "NONE"
                      for a in analyses) + len(record["dropped_by_checks"])
        blocked = blocked_edits(record)
        self.emit({"event": "merge", "step": s, "analyzed": len(analyses), "proposals": len(proposals),
                   "dropped_by_checks": dropped, "blocked_by_check": len(blocked), "edits": len(record["edits"]),
                   **({"candidate": version} if version else {}), "summary": record["summary"]})
        if blocked:
            buffer.append({"step": s, "blocked": blocked})
        if version is None:
            return current, versions
        cand = run(self.domain, recs, self.system(version), model, user, trials, seed, step_dir / "candidate.json")
        a, b = (sum(succeeded(x) for x in r.simulations) for r in (cur, cand))
        retest = {"event": "retest", "step": s, "candidate": version, "runs": len(cand.simulations),
                  "successes": {"current": a, "candidate": b}}
        harm = harmful(cur, cand, [e["written"] for e in record["edits"] if e["op"] != "DELETE"],
                       self.ev["harm_drop"], set(self.ev["excluded"]))
        bad = [e for e in record["edits"] if e["written"] in harm]
        pruned = None
        if bad:
            # the rules that made tasks fail come out, and are listed to later steps like the edits the check blocked;
            # the rest of the step is run again on its own
            record["removed_after_retest"] = [{"edit": e, "problems": ["H"], "tasks": harm[e["written"]]} for e in bad]
            (step_dir / "merge.json").write_text(json.dumps(record, indent=2) + "\n")
            buffer.append({"step": s, "blocked": [{"edit": e, "problems": ["H"]} for e in bad]})
            if len(bad) < len(record["edits"]):
                pruned = f"v{versions + 2:02d}"
                shutil.rmtree(self.out / "plugins" / pruned, ignore_errors=True)
                shutil.copytree(self.out / "plugins" / version, self.out / "plugins" / pruned)
                apply(self.out / "plugins" / pruned, undo(bad, self.out / "plugins" / current))
            self.emit({"event": "harm", "step": s, "candidate": version, "removed": [e["written"] for e in bad],
                       "tasks": {e["written"]: harm[e["written"]] for e in bad}, **({"version": pruned} if pruned else {})})
            if pruned:
                # a candidate that passed with the bad rules in it keeps the rest as it is: those rules only cost it
                # successes. One that did not is run again without them, as they may have pulled it under the margin.
                rerun = b - a < self.ev["keep_margin"]
                retest.update(candidate=pruned, rerun=rerun,
                              pruned={"candidate": version, "successes": {"current": a, "candidate": b}})
                if rerun:
                    again = run(self.domain, recs, self.system(pruned), model, user, trials, seed,
                                step_dir / "pruned.json")
                    retest.update(runs=len(again.simulations),
                                  successes={"current": a, "candidate": sum(succeeded(x) for x in again.simulations)})
        retest["kept"] = (not bad or pruned is not None) and (
            retest["successes"]["candidate"] - a >= self.ev["keep_margin"])
        self.emit(retest)
        return (retest["candidate"] if retest["kept"] else current), versions + (2 if pruned else 1)

    def go(self):
        self.setup()
        events, done, current, versions, buffer = self.state()
        if any(e["event"] == "stop" for e in events):
            print(f"finished before: {current}")
            return
        if not events:
            self.emit({"event": "start", "plugin": START})
        self.index()
        for s, (failing, controls) in enumerate(self.steps, 1):
            if s not in done:
                print(f"{now()} step {s}/{len(self.steps)}: {len(failing)} failing, {len(controls)} controls, "
                      f"plugin {current}")
                current, versions = self.step(s, failing, controls, current, versions, buffer)
                self.index()
        self.emit({"event": "stop", "reason": "pool_exhausted", "plugin": current})
        print(f"{now()} done: {current}")
