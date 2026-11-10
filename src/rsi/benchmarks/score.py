"""pass^1 to pass^n of saved tau2 results, each file on its own row.

    python -m rsi.benchmarks.score results/official_setting/retail.json [results/<version>/<domain>.json ...]

pass^k is tau2's own per-task pass^k (get_tasks_pass_hat_k, which its leaderboard submissions report), averaged over
tasks, with a 95% normal interval over tasks. A trial tau2 gave up on after its retries ends in an infrastructure error
with no reward; tau2's pass^k counts it as a failure, and it is listed apart. tau2's "average reward" leaves such trials
out, so it is not pass^1 and is not used.
"""
import sys

import pandas as pd
from tau2.data_model.simulation import Results, TerminationReason
from tau2.metrics.agent_metrics import get_tasks_pass_hat_k


def score(path):
    """pass^k with its interval, the trials ending in an infrastructure error, and the shares of tasks solved in none,
    some or all of their trials."""
    results = Results.load(path)
    task = get_tasks_pass_hat_k(results)
    passes = [c for c in task.columns if c.startswith("pass^")]
    row = {"tasks": len(task), "trials": results.info.num_trials,
           "infrastructure errors": sum(s.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR
                                        for s in results.simulations)}
    for k in passes:
        row[k] = f"{100 * task[k].mean():.1f} ± {196 * task[k].std(ddof=1) / len(task) ** 0.5:.1f}"
    p = task["pass^1"]
    row["none"], row["some"], row["all"] = (f"{100 * x:.0f}%" for x in ((p == 0).mean(), ((p > 0) & (p < 1)).mean(),
                                                                         (p == 1).mean()))
    return row


if __name__ == "__main__":
    print(pd.DataFrame([score(p) for p in sys.argv[1:]], index=sys.argv[1:]).to_string())
