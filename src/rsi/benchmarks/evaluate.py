"""A system version H<i>-M<j> on a domain's official tasks, as in the official setting: base split, 4 trials, seed 300,
DeepSeek as the user. Only for reporting: no decision may use these results.

    python -m rsi.benchmarks.evaluate H0-M0 --retail|--airline

H<i> of a domain is its plugin after i accepted plugin gates (H0 is plugins/<domain>), next to the other domain's plugin
as it stood at the end of that round; M<j> is its model after j accepted model gates (M0 is the base model), an adapter
the model server loads from the repository.
Writes results/<version>/<domain>.json and the log to artifacts/logs/eval/<version>/<domain>.log; an interrupted run
resumes. Needs the model server and, in .env, DEEPSEEK_API_KEY (the user) and OPENAI_API_KEY (retail's NL-assertion
judge).
"""
import argparse
import json
import re
from contextlib import redirect_stderr, redirect_stdout

from tau2.config import DEFAULT_MAX_STEPS
from tau2.runner import get_tasks

from rsi.benchmarks.run import OFFICIAL_DB, ROOT, config, run

SEED = 300  # tau2's default, which the leaderboard's runs use
DOMAINS = ("retail", "airline")


def plugins(domain, i, logs=ROOT / "artifacts/logs"):
    """Both domains' plugin directories once `domain` reached H<i>: gates accepted round by round, a retry after the
    first attempt; the other domain's plugin is its latest accepted one by the end of that round."""
    current, count = {d: f"plugins/{d}" for d in DOMAINS}, {d: 0 for d in DOMAINS}
    rounds = sorted(int(p.name[1:]) for p in logs.glob("r*") if p.name[1:].isdigit())
    for r in [0] + rounds:
        for d in DOMAINS:
            for attempt in ("", "_retry"):
                path = logs / f"r{r}/gate/plugin_{d}{attempt}/decision.json"
                decision = json.loads(path.read_text()) if path.exists() else {}
                if decision.get("accepted"):
                    current[d], count[d] = decision["saved"], count[d] + 1
        if count[domain] == i:
            return [current[d] for d in DOMAINS]
    raise SystemExit(f"{domain} has no H{i}: {count[domain]} plugin gates accepted so far")


def model_of(domain, j, logs=ROOT / "artifacts/logs"):
    """The domain's model once it reached M<j>: the base model for M0, else the adapter directory that its j-th accepted
    model gate saved."""
    if j == 0:
        return config()["system"]["model"]
    rounds = sorted(int(p.name[1:]) for p in logs.glob("r*") if p.name[1:].isdigit())
    paths = [logs / f"r{r}/gate/model_{domain}/decision.json" for r in rounds]
    saved = [d["saved"] for d in (json.loads(p.read_text()) for p in paths if p.exists()) if d.get("accepted")]
    if len(saved) < j:
        raise SystemExit(f"{domain} has no M{j}: {len(saved)} model gates accepted so far")
    return saved[j - 1]


def official(domain, name, plugin_dirs, model):
    """Run the official tasks on a system, into results/<name>/<domain>.json with its log."""
    log = ROOT / "artifacts/logs/eval" / name / f"{domain}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"logging to {log}")
    tasks = [{"task": t.model_dump(), "db": str(OFFICIAL_DB[domain])} for t in get_tasks(domain, "base")]
    with open(log, "a", buffering=1) as f, redirect_stdout(f), redirect_stderr(f):
        run(domain, tasks, plugin_dirs, model, config()["user"]["test"], 4, SEED,
            ROOT / "results" / name / f"{domain}.json", max_steps=DEFAULT_MAX_STEPS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="H<i>-M<j>")
    domain = parser.add_mutually_exclusive_group(required=True)
    domain.add_argument("--retail", dest="domain", action="store_const", const="retail")
    domain.add_argument("--airline", dest="domain", action="store_const", const="airline")
    args = parser.parse_args()
    m = re.fullmatch(r"H(\d+)-M(\d+)", args.version)
    if not m:
        raise SystemExit(f"{args.version}: a version is H<i>-M<j>")
    official(args.domain, args.version, plugins(args.domain, int(m[1])), model_of(args.domain, int(m[2])))


if __name__ == "__main__":
    main()
