"""Gemma 4 12B behind tau2's official agent on a domain's official tasks: base split, 4 trials, DeepSeek as the user, as
in rsi.benchmarks.evaluate.

    python -m rsi.benchmarks.baseline --retail|--airline [--concurrency N]

Writes results/official_setting/<domain>.json and the run's log to artifacts/logs/eval/official_setting/<domain>.log.
The model is served at localhost:8001 with thinking on; DEEPSEEK_API_KEY in .env serves the user and OPENAI_API_KEY
retail's NL-assertion judge.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import httpx
import litellm
from tau2.data_model.simulation import TextRunConfig
from tau2.runner import get_tasks
from tau2.runner.batch import run_tasks

from rsi.benchmarks.run import DEEPSEEK

ROOT = Path(__file__).resolve().parents[3]
AGENT = "openai/gemma-4-12b"
# The local model has no price in litellm's table, and tau2 logs an error for every agent call it cannot price.
litellm.register_model({AGENT: {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0, "litellm_provider": "openai",
                                "mode": "chat"}})
# tau2 caps litellm at 10 connections per process (utils/llm_utils.py), too few to keep the GPU busy
litellm.client_session = httpx.Client(limits=httpx.Limits(max_connections=256, max_keepalive_connections=64))
# The Gemma 4 model card's sampling, one setting for all uses. max_tokens only stops a rare runaway generation: the
# longest reply of the paper's run has 5,102 tokens.
AGENT_ARGS = {"api_base": "http://localhost:8001/v1", "api_key": "EMPTY", "temperature": 1.0, "top_p": 0.95,
              "top_k": 64, "max_tokens": 32768}
# [user.test] of config/harness_rsi_config.toml
USER = "deepseek/deepseek-flash"
USER_ARGS = {"temperature": 0.0, "extra_body": {"thinking": {"type": "disabled"}}}
# litellm's bundled price table, its fallback when the download fails, lacks DeepSeek: tau2 would log an error for every
# user call and record no cost
if USER not in litellm.model_cost:
    litellm.register_model({USER: DEEPSEEK})


def config(domain, concurrency, user=USER, user_args=USER_ARGS):
    return TextRunConfig(domain=domain, llm_agent=AGENT, llm_args_agent=AGENT_ARGS, llm_user=user,
                         llm_args_user=user_args, num_trials=4, max_concurrency=concurrency, auto_resume=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    domain = parser.add_mutually_exclusive_group(required=True)
    domain.add_argument("--retail", dest="domain", action="store_const", const="retail")
    domain.add_argument("--airline", dest="domain", action="store_const", const="airline")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    log = ROOT / "artifacts/logs/eval/official_setting" / f"{args.domain}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"logging to {log}")
    with open(log, "a", buffering=1) as f, redirect_stdout(f), redirect_stderr(f):
        run_tasks(config(args.domain, args.concurrency), get_tasks(args.domain, "base"),
                  save_path=ROOT / "results/official_setting" / f"{args.domain}.json")
