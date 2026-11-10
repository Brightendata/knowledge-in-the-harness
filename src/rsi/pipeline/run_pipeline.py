"""One round of the pipeline, a step at a time for all domains: the collection, retail first,
then plugin evolution with its gate (and retry), airline first, then the difficulty profile, retail first, then GRPO,
airline first, and the model gate of both domains. The steps also run alone: rsi.pipeline.collect,
rsi.pipeline.harness_rsi, rsi.pipeline.profile, rsi.pipeline.grpo, rsi.pipeline.model_gate.

    python -m rsi.pipeline.run_pipeline --round 1

Run from the repository root; needs the model server at localhost:8001 (for GRPO and the model gate on the GPU server,
as rsi.pipeline.grpo says) and DEEPSEEK_API_KEY in .env. Outputs are the steps' own, under artifacts/logs/r<k>/ and
artifacts/checkpoints/r<k>/; what it prints is also appended, each line with its time, to
artifacts/logs/r<k>/pipeline.log. A rerun skips the finished steps, picks up where it stopped and appends to the log:
after a session that finished ("round <k>: done") it notes "round <k>: resumed"; after one stopped inside a step, by
an error (whose traceback goes to stderr only) or by hand, it notes nothing, and each step notes its start only once.
"""
import argparse
import sys
from contextlib import redirect_stdout
from datetime import datetime

from rsi.benchmarks.run import ROOT
from rsi.pipeline.collect import collect
from rsi.pipeline.grpo import grpo
from rsi.pipeline.harness_rsi import harness_rsi
from rsi.pipeline.model_gate import model_gate
from rsi.pipeline.profile import profile

COLLECT = ["retail", "airline"]
EVOLVE = ["airline", "retail"]


class Tee:
    """Writes to the terminal and to the round's log, where each line starts with the time."""

    def __init__(self, path):
        self.term, self.log, self.line_start = sys.stdout, open(path, "a", buffering=1), True

    def write(self, s):
        self.term.write(s)
        for part in s.splitlines(keepends=True):
            self.log.write((f"{datetime.now():%Y-%m-%d %H:%M:%S} " if self.line_start else "") + part)
            self.line_start = part.endswith("\n")
        return len(s)

    def flush(self):
        self.term.flush()
        self.log.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()
    log = ROOT / f"artifacts/logs/r{args.round}/pipeline.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    lines = log.read_text().splitlines() if log.exists() else []
    with redirect_stdout(Tee(log)):
        if not lines:
            print(f"round {args.round}: started")
        elif lines[-1].endswith(f"round {args.round}: done"):
            print(f"round {args.round}: resumed")
        for domain in COLLECT:
            collect(domain, args.round)
        plugins = [harness_rsi(domain, args.round) for domain in EVOLVE]
        for domain in COLLECT:
            profile(domain, args.round, plugins)
        for domain in EVOLVE:
            grpo(domain, args.round, plugins)
        model_gate(EVOLVE, args.round, plugins)
        print(f"round {args.round}: done")


if __name__ == "__main__":
    main()
