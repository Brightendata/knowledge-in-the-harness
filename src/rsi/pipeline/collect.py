"""Step 1 of a round, the collection: the system S(k-1) (rsi.benchmarks.run.start_system: [system] of
config/harness_rsi_config.toml in round 1, then what the earlier rounds' gates accepted) runs every task of batch B(k)
four times, DeepSeek as the user. Retail first, then airline.

    python -m rsi.pipeline.collect --round 1 [--domain retail]

Run from the repository root on the GPU server, with DEEPSEEK_API_KEY in .env; the model server moves to [run] gpus
first when it runs elsewhere. Writes
artifacts/logs/r<k>/collect/<domain>.json (tau2 Results) and <domain>.log; an interrupted run resumes, a finished one
is skipped.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout

from rsi.benchmarks.run import ROOT, config, outcomes, pass_k, records, run, run_seed, server, start_system


def collect(domain, round_):
    cfg = config()
    out = ROOT / f"artifacts/logs/r{round_}/collect/{domain}.json"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".log")
    if not log.exists():
        print(f"logging to {log}")
    system = start_system(round_)
    plugins = [system["retail"]["plugin"], system["airline"]["plugin"]]
    with open(log, "a", buffering=1) as f, redirect_stdout(f), redirect_stderr(f):
        server(cfg["run"]["gpus"])
        results = run(domain, records(domain, f"round{round_}"), plugins, system[domain]["model"],
                      cfg["user"]["train"], cfg["collect"]["trials"], run_seed(round_, 0), out)
    wins = outcomes(results)
    print(f"{domain}: {len(wins)} tasks, pass^1 {100 * sum(pass_k(results, 1).values()) / len(wins):.1f}, "
          f"pass^4 {100 * sum(pass_k(results, 4).values()) / len(wins):.1f}, "
          f"{sum(not all(w) for w in wins.values())} with a failure")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--domain", choices=["retail", "airline"], help="one domain only; both by default")
    args = parser.parse_args()
    for domain in [args.domain] if args.domain else ["retail", "airline"]:
        collect(domain, args.round)


if __name__ == "__main__":
    main()
