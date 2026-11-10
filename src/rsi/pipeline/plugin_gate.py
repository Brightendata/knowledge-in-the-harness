"""Step 3 of a round, the plugin gate on V, run by rsi.pipeline.harness_rsi after each evolution attempt: the attempt's
final plugin against the plugin it started from, same model and other-domain plugin, one new seed for both, the
[user.gate] simulator as the user. Accepted when the gate's score, (1 - w) mean pass^1 + w mean pass^4 with
w = [gate] pass4_weight, is higher. Writes
artifacts/logs/r<k>/gate/plugin_<domain>[_retry]/ (current.json, candidate.json, decision.json, run.log) and copies the
gated plugin to artifacts/logs/r<k>/plugins/<domain>_final<1|2>_<accepted|rejected>/ (1 the first attempt, 2 the
retry); an interrupted gate resumes when harness_rsi runs again.
"""
import json
import shutil
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime

from rsi.benchmarks.run import ROOT, config, paired, records, run, run_seed, server
from rsi.harness_rsi.evolve import START


def score(w):
    """A gate's score, (1 - w) pass^1 + w pass^4, as weights of pass^k (rsi.benchmarks.run.pass_k)."""
    return {1: 1 - w, 4: w}


def accepts(d):
    """Whether a difference in the gates' score passes: the candidate's is higher. The margin absorbs float error."""
    return d > 1e-9


def gate(domain, round_, retry):
    """The decision on an attempt's final plugin, or None when the attempt kept no edit and there is nothing to gate. A
    decision made before is returned as it was."""
    suffix = "_retry" if retry else ""
    made = ROOT / f"artifacts/logs/r{round_}/gate/plugin_{domain}{suffix}/decision.json"
    if made.exists():
        return json.loads(made.read_text())
    evolved = ROOT / f"artifacts/logs/r{round_}/harness_rsi/{domain}{suffix}"
    stops = [e for e in map(json.loads, open(evolved / "events.jsonl")) if e["event"] == "stop"]
    if not stops:
        raise SystemExit(f"plugin evolution in {evolved} has not finished")
    final = stops[-1]["plugin"]
    if final == START:
        return None
    setup = json.loads((evolved / "config.json").read_text())
    current = [setup["start_plugin"], setup["other_plugin"]]
    candidate = [str(evolved.relative_to(ROOT) / "plugins" / final), setup["other_plugin"]]
    cfg = config()
    seed = run_seed(round_, 2, int(retry))
    out = ROOT / f"artifacts/logs/r{round_}/gate/plugin_{domain}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "run.log").exists():
        print(f"logging to {out / 'run.log'}")
    tasks = records(domain, "validation")
    common = (setup["model"], cfg["user"]["gate"], cfg["gate"]["trials"], seed)
    with open(out / "run.log", "a", buffering=1) as f, redirect_stdout(f), redirect_stderr(f):
        server(cfg["run"]["gpus"])
        cur = run(domain, tasks, current, *common, out / "current.json")
        cand = run(domain, tasks, candidate, *common, out / "candidate.json")
    decision = {"time": datetime.now().isoformat(timespec="seconds"), "seed": seed, "user": cfg["user"]["gate"],
                "current": {"plugin": setup["start_plugin"], "model": setup["model"]},
                "candidate": {"plugin": candidate[0], "model": setup["model"]},
                "other_plugin": setup["other_plugin"],
                **{f"pass^{k}": paired(cur, cand, k, cfg["gate"]["bootstrap"]) for k in (1, 4)},
                "score": paired(cur, cand, score(cfg["gate"]["pass4_weight"]), cfg["gate"]["bootstrap"])}
    decision["accepted"] = accepts(decision["score"]["diff"])
    saved = ROOT / (f"artifacts/logs/r{round_}/plugins/{domain}_final{1 + int(retry)}_"
                    f"{'accepted' if decision['accepted'] else 'rejected'}")
    shutil.rmtree(saved, ignore_errors=True)
    shutil.copytree(ROOT / candidate[0], saved)
    decision["saved"] = str(saved.relative_to(ROOT))
    (out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    for key in ("pass^1", "pass^4", "score"):
        d = decision[key]
        print(f"{domain} {key}: {100 * d['current']:.1f} -> {100 * d['candidate']:.1f} "
              f"(diff {100 * d['diff']:+.1f}, 95% CI {100 * d['ci95'][0]:+.1f} to {100 * d['ci95'][1]:+.1f})")
    print(f"{domain}: {'accepted' if decision['accepted'] else 'not accepted'}, plugin saved to {decision['saved']}")
    return decision
