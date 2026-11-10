"""Step 6 of a round, the model gate on V, DeepSeek as the user and the plugins H(k). With [model_gate] screen_trials
above 0, every GRPO checkpoint of a domain first runs V that many times under a seed of its own, and the
[model_gate] finalists with the highest pass^1 there (the earlier on a tie) go on; with 0, every checkpoint is a
finalist. The finalists run V [gate] trials times under the seed of the domain's last plugin gate, and are compared
with H(k) on the round's starting model as that gate scored it. In a round whose
evolution kept no edit there was no plugin gate, and H(k) = H(k-1) on M(k-1) runs V here, beside the finalists, under a
seed of its own. A finalist passes when its score, (1 - w) mean pass^1 + w mean pass^4 with w = [model_gate]
pass4_weight, is higher; of those that pass, the one with the highest score is accepted; with none, the model stays
M(k-1). The finalists run afresh, so their scores do not carry the screening's luck. The checkpoints of all domains are
served at once as LoRA adapters, the model server restarted on [model_gate] gpus.

    python -m rsi.pipeline.model_gate --round 1 [--domain airline]

Run from the repository root on the GPU server, after rsi.pipeline.grpo, with DEEPSEEK_API_KEY in .env. Writes
artifacts/logs/r<k>/gate/model_<domain>/ (screen/<checkpoint>.json, <finalist>.json, current.json when it runs H(k)
itself, decision.json) and the log to artifacts/logs/r<k>/gate/model_run.log; an interrupted gate resumes.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime

import numpy as np
from tau2.data_model.simulation import Results

from rsi.benchmarks.run import ROOT, config, paired, pass_k, records, run, run_seed, served, server, start_system
from rsi.pipeline.grpo import grpo_config
from rsi.pipeline.harness_rsi import harness_rsi
from rsi.pipeline.plugin_gate import accepts, score


def baseline(domain, round_):
    """The seed of the domain's model gate and the tau2 Results of H(k) on M(k-1) on V it compares with: the round's
    last plugin gate's, or, with none, the gate's own run of it."""
    for suffix in ("_retry", ""):
        path = ROOT / f"artifacts/logs/r{round_}/gate/plugin_{domain}{suffix}"
        if (path / "decision.json").exists():
            decision = json.loads((path / "decision.json").read_text())
            return decision["seed"], path / ("candidate.json" if decision["accepted"] else "current.json")
    return run_seed(round_, 5), ROOT / f"artifacts/logs/r{round_}/gate/model_{domain}/current.json"


def finalists(screened, n):
    """The n checkpoints with the highest mean pass^1 in the screening, the earlier on a tie. screened: name -> Results,
    in checkpoint order."""
    mean = {name: np.mean(list(pass_k(r, 1).values())) for name, r in screened.items()}
    return sorted(sorted(mean, key=lambda name: -mean[name])[:n], key=list(screened).index)


def choose(scores):
    """The accepted finalist: the highest score among those that pass, the earlier on a tie; None when none does."""
    return max((n for n, s in scores.items() if s["passes"]), key=lambda n: scores[n]["score"]["candidate"],
               default=None)


def model_gate(domains, round_, plugins):
    """plugins: both domains' H(k)."""
    cfg = config()
    gate = grpo_config("model_gate")
    out = ROOT / f"artifacts/logs/r{round_}/gate"
    ckpts = {}
    for d in domains:
        found = sorted((ROOT / f"artifacts/checkpoints/r{round_}/{d}").glob("epoch*"),
                       key=lambda p: float(p.name[len("epoch"):]))
        if found and not (out / f"model_{d}/decision.json").exists():
            ckpts[d] = found
    if not ckpts:
        return
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "model_run.log").exists():
        print(f"model gate: {sum(map(len, ckpts.values()))} checkpoints on V, logging to {out / 'model_summary.log'}")

    def runs(jobs):
        """Results of (domain, model, seed, trials, path) jobs, side by side, sharing [model_gate] concurrency."""
        for _, model, _, _, _ in jobs:
            served(model)
        with ThreadPoolExecutor(len(jobs)) as pool:
            return dict(zip((j[4] for j in jobs), pool.map(
                lambda j: run(j[0], records(j[0], "validation"), plugins, j[1], cfg["user"]["gate"], j[3], j[2], j[4],
                              concurrency=max(1, gate["concurrency"] // len(jobs))), jobs)))

    with open(out / "model_run.log", "a", buffering=1) as f, redirect_stdout(f), redirect_stderr(f):
        server(gate["gpus"])
        screen = runs([(d, str(c.relative_to(ROOT)), run_seed(round_, 6), gate["screen_trials"],
                        out / f"model_{d}/screen/{c.name}.json") for d, cs in ckpts.items() for c in cs]) \
            if gate["screen_trials"] else {}
        chosen = {d: finalists({c.name: screen[out / f"model_{d}/screen/{c.name}.json"] for c in cs},
                               gate["finalists"]) if screen else [c.name for c in cs] for d, cs in ckpts.items()}
        compared = {d: baseline(d, round_) for d in ckpts}
        own = [(d, start_system(round_)[d]["model"], seed, cfg["gate"]["trials"], path)
               for d, (seed, path) in compared.items() if not path.exists()]
        done = runs(own + [(d, f"artifacts/checkpoints/r{round_}/{d}/{name}", compared[d][0], cfg["gate"]["trials"],
                            out / f"model_{d}/{name}.json") for d, names in chosen.items() for name in names])
    for d, (seed, path) in compared.items():
        current = done[path] if path in done else Results.load(path)
        scores = {}
        for name in chosen[d]:
            cand = done[out / f"model_{d}/{name}.json"]
            s = {f"pass^{k}": paired(current, cand, k, cfg["gate"]["bootstrap"]) for k in (1, 4)}
            s["score"] = paired(current, cand, score(gate["pass4_weight"]), cfg["gate"]["bootstrap"])
            scores[name] = {**s, "passes": accepts(s["score"]["diff"])}
        best = choose(scores)
        screened = {c.name: float(np.mean(list(pass_k(screen[out / f"model_{d}/screen/{c.name}.json"], 1).values())))
                    for c in ckpts[d]} if screen else {}
        decision = {"time": datetime.now().isoformat(timespec="seconds"), "seed": seed,
                    **({"screen_seed": run_seed(round_, 6)} if screen else {}),
                    "user": cfg["user"]["gate"], "plugins": plugins, "current": str(path.relative_to(ROOT)),
                    **({"screen": screened} if screen else {}), "finalists": scores, "accepted": best,
                    "saved": f"artifacts/checkpoints/r{round_}/{d}/{best}" if best else None}
        (out / f"model_{d}/decision.json").write_text(json.dumps(decision, indent=2) + "\n")
        lines = [f"{d} screen ({gate['screen_trials']} runs a task), pass^1: "
                 + ", ".join(f"{name} {100 * p:.1f}" for name, p in screened.items())] if screen else []
        lines += [f"{d} {name}: {scored(s)}, {'passes' if s['passes'] else 'fails'}" for name, s in scores.items()]
        lines.append(f"{d}: accepted {best}, {scored(scores[best])}" if best else
                     f"{d}: no finalist passes; the model stays M(k-1)")
        with open(out / "model_summary.log", "a") as brief:
            brief.write("".join(f"{datetime.now():%Y-%m-%d %H:%M:%S} {line}\n" for line in lines))
        print(lines[-1])


def scored(s):
    """A finalist's score, pass^1 and pass^4 on V and their differences from the current model, in points."""
    return ", ".join(f"{k} {100 * s[k]['candidate']:.1f} ({100 * s[k]['diff']:+.1f})"
                     for k in ("score", "pass^1", "pass^4"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--domain", choices=["retail", "airline"], help="one domain only; both by default")
    args = parser.parse_args()
    plugins = [harness_rsi(d, args.round) for d in ["airline", "retail"]]
    model_gate([args.domain] if args.domain else ["airline", "retail"], args.round, plugins)


if __name__ == "__main__":
    main()
