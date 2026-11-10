"""Steps 2 and 3 of a round for each domain, airline first: plugin evolution, then the plugin gate on V; when the gate
does not accept, one retry with the retry recipe, and its gate. An attempt that kept no edit is not gated.

    python -m rsi.pipeline.harness_rsi --round 1 [--domain retail]

Run from the repository root on the GPU server after rsi.pipeline.collect for the same round, with DEEPSEEK_API_KEY in
.env; the model server moves to [run] gpus first when it runs elsewhere. Writes
artifacts/logs/r<k>/harness_rsi/<domain>[_retry]/ (its run.log follows the evolution step by step),
gate/plugin_<domain>[_retry]/ and plugins/; a rerun picks up where it stopped.
"""
import argparse
import json
from contextlib import redirect_stderr, redirect_stdout

from rsi.benchmarks.run import config, server, start_system
from rsi.harness_rsi.evolve import Attempt
from rsi.pipeline.plugin_gate import gate


def evolve(domain, round_, retry):
    """An attempt of plugin evolution; one that finished before is skipped."""
    attempt = Attempt(domain, round_, retry)
    path = attempt.out / "events.jsonl"
    if path.exists() and any(json.loads(line)["event"] == "stop" for line in open(path)):
        return
    attempt.out.mkdir(parents=True, exist_ok=True)
    log = attempt.out / "run.log"
    if not log.exists():
        print(f"{domain}{' retry' if retry else ''}: {len(attempt.steps)} steps, logging to {log}")
    with open(log, "a", buffering=1) as f, redirect_stdout(f), redirect_stderr(f):
        server(config()["run"]["gpus"])
        attempt.go()
    events = [json.loads(line) for line in open(path)]
    print(f"{domain}: final plugin {events[-1]['plugin']}; kept "
          f"{sum(e['event'] == 'retest' and e['kept'] for e in events)} of {sum(e['event'] == 'retest' for e in events)}"
          f" candidates")


def harness_rsi(domain, round_):
    """Evolution and gate of one domain, then the retry and its gate if the gate did not accept. Returns H(k): the
    plugin a gate accepted, else the one the round started from."""
    evolve(domain, round_, False)
    decision = gate(domain, round_, False)
    if decision is not None and not decision["accepted"]:
        evolve(domain, round_, True)
        decision = gate(domain, round_, True)
    return decision["saved"] if decision and decision["accepted"] else start_system(round_)[domain]["plugin"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--domain", choices=["retail", "airline"], help="one domain only; both by default")
    args = parser.parse_args()
    for domain in [args.domain] if args.domain else ["airline", "retail"]:
        harness_rsi(domain, args.round)


if __name__ == "__main__":
    main()
