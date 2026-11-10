"""Step 4 of a round, the difficulty profile: H(k), the plugins after the round's plugin gates, with the model the round
started from runs every task of batch B(k) as many times as the collection did, DeepSeek as the user. GRPO picks its
tasks and their rollouts by how often each task succeeds here; next to the collection, the profile shows which tasks
the new plugins made solvable. Retail first, then airline.

    python -m rsi.pipeline.profile --round 1 [--domain retail]

Run from the repository root on the GPU server, with DEEPSEEK_API_KEY in .env; the model server moves to [run] gpus
first when it runs elsewhere. H(k) comes from
rsi.pipeline.harness_rsi, which first runs whatever of plugin evolution and its gates is unfinished. Writes
artifacts/logs/r<k>/profile/<domain>.json (tau2 Results) and <domain>.log; an interrupted run resumes, a finished one
is skipped.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout

from rsi.benchmarks.run import ROOT, config, outcomes, pass_k, records, run, run_seed, server, start_system
from rsi.pipeline.harness_rsi import harness_rsi


def profile(domain, round_, plugins):
    """plugins: both domains' H(k)."""
    cfg = config()
    out = ROOT / f"artifacts/logs/r{round_}/profile/{domain}.json"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".log")
    if not log.exists():
        print(f"logging to {log}")
    trials = cfg["collect"]["trials"]
    with open(log, "a", buffering=1) as f, redirect_stdout(f), redirect_stderr(f):
        server(cfg["run"]["gpus"])
        results = run(domain, records(domain, f"round{round_}"), plugins, start_system(round_)[domain]["model"],
                      cfg["user"]["train"], trials, run_seed(round_, 3), out)
    solved = [sum(w) for w in outcomes(results).values()]
    print(f"{domain}: {len(solved)} tasks, pass^1 {100 * sum(pass_k(results, 1).values()) / len(solved):.1f}, "
          f"pass^4 {100 * sum(pass_k(results, 4).values()) / len(solved):.1f}; solved in none / some / all of "
          f"{trials} runs: {solved.count(0)} / {sum(0 < s < trials for s in solved)} / {solved.count(trials)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--domain", choices=["retail", "airline"], help="one domain only; both by default")
    args = parser.parse_args()
    plugins = [harness_rsi(d, args.round) for d in ["airline", "retail"]]
    for domain in [args.domain] if args.domain else ["retail", "airline"]:
        profile(domain, args.round, plugins)


if __name__ == "__main__":
    main()
