"""A model behind an API as the agent of our harness with the H0 plugins, on a domain's official tasks in the setting of
rsi.benchmarks.evaluate (base split, 4 trials, seed 300, the [user.test] simulator), reasoning on at low effort. Only
for reporting.

    python -m rsi.benchmarks.evaluate_api <model> --retail|--airline [--concurrency N]

<model> is a litellm model name: openai/responses/gpt-5.6-sol (OpenAI at its flex tier, through the Responses API:
its chat completions take no tools with reasoning on), or openrouter/<author>/<slug> such as
openrouter/anthropic/claude-sonnet-5 and openrouter/z-ai/glm-5.2. Writes results/H0-<slug>/<domain>.json and the
log to artifacts/logs/eval/H0-<slug>/<domain>.log, an interrupted run resuming; then prints pass^k, the agent's cost
and how often the agent loaded a write's skill before making it. Needs, in .env, the model's key (OPENAI_API_KEY or
OPENROUTER_API_KEY), DEEPSEEK_API_KEY for the user and, for retail, OPENAI_API_KEY for the NL-assertion judge.
"""
import argparse
import json
import shutil
from contextlib import redirect_stderr, redirect_stdout

import litellm
import tau2.utils.llm_utils as llm_utils
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import Results, TerminationReason, TextRunConfig
from tau2.runner import get_tasks
from tau2.runner.batch import run_tasks

from rsi.benchmarks.evaluate import SEED
from rsi.benchmarks.run import DEEPSEEK, ROOT, config, merge, save
from rsi.benchmarks.score import score
from rsi.harness_rsi.evidence import SKILL, loads, writes

PLUGINS = ["plugins/retail", "plugins/airline"]
# $ per token, input / output / cached input (OpenAI's at its flex tier), from OpenRouter's listing (2026-10-25), for a
# model litellm's price table lacks: tau2 would log an error for every call it cannot price and record no cost
PRICES = {
    "gpt-5.6-sol": (2e-6, 10e-6, 0.2e-6),
    "openrouter/anthropic/claude-sonnet-5": (2e-6, 10e-6, 0.2e-6),
    "openrouter/z-ai/glm-5.2": (1.4e-6, 4.4e-6, 0.26e-6),
}
# the OpenRouter host a model's calls go to: the lab's own, which serves the original weights and, for Claude, keeps
# the prompt cache with one host
HOSTS = {"anthropic/claude-sonnet-5": ["anthropic"], "z-ai/glm-5.2": ["z-ai"]}


def model_args(model):
    """The agent's model arguments: our harness's max_tokens, and reasoning at low effort in the request body, in the
    field each API reads (litellm drops a reasoning parameter for a model its table does not list); OpenAI's reasoning
    models take max_completion_tokens only, and run at the flex tier: the same model at half price, slower and now and
    then turned away (429), which the retries and the rerun of infrastructure errors absorb."""
    a = config()["agent"]
    args = {"max_tokens": a["max_tokens"]}
    if model.startswith("openrouter/"):
        slug = model.removeprefix("openrouter/")
        args["extra_body"] = {"reasoning": {"effort": "low"}}
        if slug in HOSTS:
            args["extra_body"]["provider"] = {"order": HOSTS[slug], "allow_fallbacks": False}
        if slug.startswith("anthropic/"):
            # Anthropic caches a prompt only up to a marked block: marking the last message lets each call read the
            # previous call's prompt from the cache
            args["cache_control_injection_points"] = [{"location": "message", "index": -1}]
    else:
        args = {"max_completion_tokens": args.pop("max_tokens"), "service_tier": "flex",
                "extra_body": {"reasoning_effort": "low"}}
    return args


convert = llm_utils.to_litellm_messages


def with_reasoning(messages):
    """tau2's conversion of a conversation for the model, keeping the reasoning of the agent's own replies, which tau2
    drops: a Claude tool call must come back with the thinking before it. The user simulator's messages carry no
    raw_data and stay as they are."""
    out = []
    for m in messages:
        d = convert([m])
        if d and isinstance(m, AssistantMessage) and (m.raw_data or {}).get("choices"):
            msg = m.raw_data["choices"][0].get("message") or {}
            details = msg.get("reasoning_details") or (msg.get("provider_specific_fields") or {}).get("reasoning_details")
            d[0].update({k: v for k, v in (("reasoning_content", msg.get("reasoning_content")),
                                            ("reasoning_details", details)) if v})
        out += d
    return out


def skill_use(results, domain):
    """Conversations whose agent loaded the domain's entry, and writes made after loading their category's skill, of all
    writes the environment accepted."""
    entry = before = total = 0
    for s in results.simulations:
        loaded = loads(s)
        entry += any(n == domain for _, n in loaded)
        for turn, tool, _ in writes(s):
            total += 1
            before += f"{domain}:{SKILL[tool]}" in {n for t, n in loaded if t <= turn}
    return entry, before, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="a litellm model name, e.g. openrouter/anthropic/claude-sonnet-5")
    domain = parser.add_mutually_exclusive_group(required=True)
    domain.add_argument("--retail", dest="domain", action="store_const", const="retail")
    domain.add_argument("--airline", dest="domain", action="store_const", const="airline")
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()
    model, user = args.model, config()["user"]["test"]
    # litellm prices a call through its Responses bridge under the bare model name
    priced = model.removeprefix("openai/responses/")
    if priced in PRICES and priced not in litellm.model_cost:
        i, o, c = PRICES[priced]
        litellm.register_model({priced: {"input_cost_per_token": i, "output_cost_per_token": o,
                                         "cache_read_input_token_cost": c, "mode": "chat",
                                         "litellm_provider": priced.split("/")[0] if "/" in priced else "openai"}})
    if user["model"] not in litellm.model_cost:
        litellm.register_model({user["model"]: DEEPSEEK})
    if model.startswith("openrouter/"):
        # OpenAI keeps its reasoning to itself and rejects fields it does not know
        llm_utils.to_litellm_messages = with_reasoning
    name = f"H0-{model.rsplit('/', 1)[-1]}"
    out = ROOT / "results" / name / f"{args.domain}.json"
    if not out.exists():
        log = ROOT / "artifacts/logs/eval" / name / f"{args.domain}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"logging to {log}")
        run_config = TextRunConfig(domain=args.domain, agent="plugin_agent", llm_agent=model,
                                   llm_args_agent={**model_args(model), "plugins": PLUGINS,
                                                   "max_skill_loads": config()["agent"]["max_skill_loads"]},
                                   llm_user=user["model"], llm_args_user=user["args"], num_trials=4, seed=SEED,
                                   max_concurrency=args.concurrency, auto_resume=True)
        # tau2's dir format, a file per conversation, as rsi.benchmarks.run uses: its single-file format rereads and
        # rewrites the whole file after every conversation
        part = out.parent / f"{out.stem}.parts" / "results.json"
        with open(log, "a", buffering=1) as f, redirect_stdout(f), redirect_stderr(f):
            run_tasks(run_config, get_tasks(args.domain, "base"), save_path=part, results_format="dir")
        results = merge([part], args.domain)
        # conversations the API never finished count as failures: with many, stop and keep the parts, whose resume
        # reruns them (rsi.benchmarks.run.run)
        errors = sum(s.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR
                     and (s.info or {}).get("error_type") not in ("ValueError", "AgentError")
                     for s in results.simulations)
        if errors > 0.02 * len(results.simulations):
            raise RuntimeError(f"{errors} of {len(results.simulations)} conversations ended in infrastructure errors; "
                               "run again to rerun them")
        save(json.loads(results.model_dump_json()), out)
        shutil.rmtree(part.parent)
    results = Results.load(out)
    entry, before, total = skill_use(results, args.domain)
    print(score(out))
    print(f"agent ${sum(s.agent_cost or 0 for s in results.simulations):.2f}, user "
          f"${sum(s.user_cost or 0 for s in results.simulations):.2f}; entry loaded in {entry} of "
          f"{len(results.simulations)} conversations; {before} of {total} writes made after loading their skill")


if __name__ == "__main__":
    main()
