"""Smoke test of harness RSI: a few round-1 tasks of each domain with H0, then one short plugin evolution on retail, end
to end. Prints PASS or FAIL per item.

    GPUS=0 scripts/serve_gemma.sh       # in another terminal; then, from the repository root:
    python scripts/smoke_harness.py

Starts afresh in runs/smoke/. Uses config/harness_rsi_config.toml shrunk to a quick run: 1–2 trials, steps of 2
failing and 2 control tasks. About 50 DeepSeek conversations, a few cents.
"""
import json
import shutil
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import rsi.benchmarks.run as R
from rsi.agent.plugin import apply, documents, rules, skill_text
from rsi.harness_rsi import editor, evidence
from rsi.harness_rsi.evolve import Attempt

OUT = R.ROOT / "runs/smoke"
# round-1 tasks, per domain a mix of ones H0 failed and ones it solved in round 1's collection
TASKS = {"retail": ["generated_budget_group_W3529525_v0", "generated_budget_group_W3529525_v1",
                    "generated_address_default_W2403075",
                    "generated_addr_pair_address_items_home_given+undo_daiki_silva_5033_1"],
         "airline": ["generated_basic_change_J590M3_v0", "generated_basic_change_J590M3_v1",
                     "generated_bags_remove_0TN6YG", "generated_balances_anya_brown_2655"]}
SMALLER = {"max_tokens = 32768": "max_tokens = 16384", "input_tokens = 30000": "input_tokens = 20000",
           "trials = 4": "trials = 2", "trials = 6": "trials = 2", "failing_per_step = 16": "failing_per_step = 2",
           "controls_per_step = 16": "controls_per_step = 2", "keep_margin = 4": "keep_margin = 1",
           "concurrency = 256": "concurrency = 8", "workers = 16": "workers = 4"}
H0 = ["plugins/retail", "plugins/airline"]
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}", flush=True)


@contextmanager
def quiet():
    """tau2's progress and logs go to runs/smoke/run.log."""
    with open(OUT / "run.log", "a", buffering=1) as f, redirect_stdout(f), redirect_stderr(f):
        yield


def recs(domain):
    by_id = {r["task"]["id"]: r for r in R.records(domain, "round1")}
    return [by_id[t] for t in TASKS[domain]]


def conversations(cfg):
    for domain in TASKS:
        with quiet():
            res = R.run(domain, recs(domain), H0, cfg["system"]["model"], cfg["user"]["train"], 1, 7,
                        OUT / "conversations" / f"{domain}.json")
        sims = res.simulations
        check(f"{domain}: conversations finished with a reward", all(s.reward_info for s in sims),
              f"{sum(s.reward_info is not None for s in sims)}/{len(sims)}")
        for s in sims:  # what the model loads is its own choice: reported, not checked
            print(f"INFO  {domain} {s.task_id}: loads {evidence.loads(s)}, writes "
                  f"{[w[1] for w in evidence.writes(s)]}, reward {s.reward_info.reward if s.reward_info else None}",
                  flush=True)
        leaked = [c.name for s in sims for m in s.messages for c in getattr(m, "tool_calls", None) or []
                  if c.name == "load_skill"]
        check(f"{domain}: load_skill stays out of the trajectory", not leaked)
        check(f"{domain}: Results record the plugins", res.info.agent_info.llm_args.get("plugins") == H0)


def edits():
    plugin = OUT / "edit" / "retail"
    shutil.copytree(R.ROOT / "plugins/retail", plugin)
    ids = apply(plugin, [{"op": "ADD", "file": "retail:exchange", "text": "Before the exchange call, ask whether "
                          "anything else in the order should be exchanged."}])
    docs = documents(plugin)
    shown = skill_text(docs, "retail:exchange")
    check("edit: rule added and shown by load_skill", ids == ["retail:exchange-1"] and "[retail:exchange-1]" in shown
          and rules(docs["retail:exchange"])[0][0] == "retail:exchange-1")
    keys = editor.db_keys([R.ROOT / "tau2-bench/data/tau2/domains/retail/db.json"])
    order = next(k for k in sorted(keys) if k.startswith("#W"))
    e = {"op": "ADD", "file": "retail:exchange", "rule_id": "", "text": f"Never change {order}.", "support": ["t"],
         "reason": ""}
    check("edit: code checks reject a rule naming an order", "database IDs" in
          (editor.edit_problem(e, plugin, {"t"}, set(), 1, keys, set(), set()) or ""))


def evolution(cfg):
    logs = OUT / "r1"
    with quiet():
        R.run("retail", recs("retail"), H0, cfg["system"]["model"], cfg["user"]["train"], cfg["collect"]["trials"],
              R.run_seed(1, 0), logs / "collect" / "retail.json")
        attempt = Attempt("retail", 1, False, logs=logs)
    print(f"evolution: {len(attempt.steps)} steps {attempt.steps}", flush=True)
    with quiet():
        attempt.go()
    events = [json.loads(line) for line in open(attempt.out / "events.jsonl")]
    kinds = [e["event"] for e in events]
    check("evolution: events from start to stop", kinds[0] == "start" and kinds[-1] == "stop" and "step" in kinds,
          " ".join(kinds))
    for e in events:
        if e["event"] in ("merge", "retest"):
            print(f"      {json.dumps(e)}")
    steps = sorted((attempt.out / "steps").glob("s*"))
    analyses = [json.loads(line) for s in steps if (s / "analysis.jsonl").exists() for line in open(s / "analysis.jsonl")]
    check("evolution: every analysis got a reply", analyses and all(a["output"] for a in analyses),
          f"{sum(bool(a['output']) for a in analyses)}/{len(analyses)}")
    for a in analyses:
        o = a["output"] or {}
        print(f"      {a['id']} {a['task_id']}: {o.get('relation')} {o.get('op')} {o.get('file') or o.get('rule_id')}"
              f" | {o.get('text')} | dropped: {a['dropped']}\n        diagnosis: {o.get('diagnosis')}")
    raw = [json.loads(line) for s in steps if (s / "editor_raw.jsonl").exists() for line in open(s / "editor_raw.jsonl")]
    check("editor: token counts logged for every call, thinking for those that thought",  # Gemma at times skips it
          raw and all(r["usage"] for r in raw) and any(r["thinking"] for r in raw),
          f"{len(raw)} calls, {sum(bool(r['thinking']) for r in raw)} with thinking")
    to_check = 0
    for s in steps:
        if (s / "merge.json").exists():
            m = json.loads((s / "merge.json").read_text())
            print(f"      {s.name} merge: {m['summary']!r}; edits {m['edits']}; dropped {m['dropped_by_checks']}")
            to_check += sum(e["op"] != "DELETE" for e in m["edits"]) + sum(
                d["reason"].startswith("check ") for d in m["dropped_by_checks"])
    checked = sum(r["id"].startswith("check") and r["try"] == 0 for r in raw)
    check(f"editor: every edit that passed the code checks was checked {editor.CHECKS} times",
          checked == editor.CHECKS * to_check, f"{checked} checks for {to_check} edits")
    versions = sorted(p.name for p in (attempt.out / "plugins").iterdir() if p.is_dir())
    made = [e["candidate"] for e in events if e["event"] == "merge" and e.get("candidate")]
    check("evolution: every candidate saved as a plugin version", set(made) <= set(versions), " ".join(versions))
    index = json.loads((attempt.out / "plugins/versions.json").read_text())
    check("evolution: versions.json says what each version holds and whether it was kept",
          all(index[e["candidate"]]["kept"] == e["kept"] for e in events if e["event"] == "retest")
          and set(index) == set(versions), json.dumps({v: [x["holds"], x.get("kept")] for v, x in index.items()}))
    for v in made:
        new = [r for p in documents(attempt.out / "plugins" / v).values() for r in rules(p)]
        print(f"      {v}: {new}")
    before = (attempt.out / "events.jsonl").read_text()
    with quiet():
        Attempt("retail", 1, False, logs=logs).go()
    check("evolution: a rerun of a finished attempt changes nothing",
          (attempt.out / "events.jsonl").read_text() == before)


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    text = R.CONFIG.read_text()
    for old, new in SMALLER.items():
        assert old in text, old
        text = text.replace(old, new)
    (OUT / "config.toml").write_text(text)
    R.CONFIG = OUT / "config.toml"
    cfg = R.config()
    edits()
    conversations(cfg)
    evolution(cfg)
    print(f"\n{sum(results)}/{len(results)} passed", flush=True)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
