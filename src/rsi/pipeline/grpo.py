"""Step 5 of a round, GRPO: for each domain a LoRA adapter, trained from the round's starting model on batch B(k) with
the plugins H(k), DeepSeek as the user ([grpo] of config/grpo_config.toml). Airline first, then retail.

    python -m rsi.pipeline.grpo --round 1 [--domain airline]

Run from the repository root on the GPU server, with DEEPSEEK_API_KEY in .env: the model server
(scripts/serve_gemma.sh, restarted when needed) runs the rollouts on [grpo] server_gpus, [grpo] concurrency of
them at once, the trainer the updates on [grpo] gpus. Writes artifacts/logs/r<k>/grpo/<domain>/ (config.json,
events.jsonl, run.log, steps/sNN/ with the step's rollouts as tau2 Results and the adapter after the step) and the
adapter of every checkpoint to artifacts/checkpoints/r<k>/<domain>/; an interrupted run resumes at the step it stopped
in.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import tomllib
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime

from tau2.data_model.simulation import Results, TerminationReason

from rsi.benchmarks.run import ROOT, config, outcomes, records, run, run_seed, serve, server, start_system
from rsi.pipeline.harness_rsi import harness_rsi

# the agent in rollouts: vLLM's token ids and logprobs of every model call, the trainer's samples; an empty reply ends
# the conversation, scored 0, so that GRPO learns from it instead of tau2 running the conversation again
ROLLOUT_AGENT = {"logprobs": True, "extra_body": {"return_token_ids": True}, "end_on_empty": True}
TRAIN_KEYS = ("lora_rank", "lora_alpha", "lora_targets", "lr", "mini_batches", "clip", "dual_clip", "max_token_grad",
              "max_grad_norm")
# where the trainer and the model server run, which the run's record leaves out
MACHINE = ("base", "python", "gpus", "server_gpus", "concurrency")


def grpo_config(section="grpo"):
    return tomllib.loads((ROOT / "config/grpo_config.toml").read_text())[section]


def rollouts_for(successes, trials, counts):
    """Rollouts of a task that succeeded `successes` times in its latest `trials` runs: counts[0] when it never
    succeeded, counts[2] when it always did, counts[1] in between."""
    return counts[0] if successes == 0 else counts[2] if successes == trials else counts[1]


def batches(task_ids, epoch, g):
    """The steps of an epoch: the tasks shuffled anew each epoch, dealt into [grpo] steps_per_epoch batches."""
    order = sorted(task_ids)
    random.Random(g["shuffle_seed"] + epoch).shuffle(order)
    return [order[i::g["steps_per_epoch"]] for i in range(g["steps_per_epoch"])]


def empty_replies(results):
    """Conversations of a step's rollouts that the agent ended on an empty reply (an agent error, scored 0)."""
    return sum(s.termination_reason == TerminationReason.AGENT_ERROR for r in results for s in r.simulations)


def adapter_name(domain, round_, step):
    return f"{domain}-r{round_}-s{step:02d}"


def rollouts(domain, round_, step, by_count, plugins, model, step_dir):
    """tau2 Results of a step's rollouts: one run per rollout count, side by side, sharing [grpo] concurrency."""
    cfg, concurrency = config(), grpo_config()["concurrency"]
    total = sum(n * len(v) for n, v in by_count.items())

    def one(item):
        n, task_recs = item
        return run(domain, task_recs, plugins, model, cfg["user"]["train"], n, run_seed(round_, 4, 0, step),
                   step_dir / f"g{n:02d}.json", agent_extra=ROLLOUT_AGENT,
                   concurrency=max(1, round(concurrency * n * len(task_recs) / total)))

    with ThreadPoolExecutor(len(by_count)) as pool:
        return list(pool.map(one, by_count.items()))


def train(g, step_dir, previous, files, log):
    """The GRPO update of the adapter in `previous` (None at the first step) on the rollouts in `files`, under torchrun
    on the [grpo] gpus, into step_dir/adapter."""
    settings = step_dir / "train_settings.json"
    settings.write_text(json.dumps({"base": g["base"], "adapter": previous and str(previous),
                                    "out": str(step_dir / "adapter"), "rollouts": [str(f) for f in files],
                                    **{k: g[k] for k in TRAIN_KEYS}}, indent=1))
    subprocess.run([g["python"], "-m", "torch.distributed.run", "--standalone",
                    f"--nproc_per_node={len(g['gpus'].split(','))}", "-m", "rsi.grpo.train", str(settings)],
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": g["gpus"], "PYTHONPATH": str(ROOT / "src"),
                        # packs of very different lengths otherwise fragment the memory
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
                   stdout=log, stderr=log, check=True)
    settings.unlink()


def grpo(domain, round_, plugins):
    """plugins: both domains' H(k). Training goes on from the domain's model M(k-1): the base model, or the adapter of
    an earlier round, whose optimizer starts afresh. A step's rollouts run on the model server while the step before it
    trains on the other GPUs, so they come from the adapter two steps back (M(k-1) for the first two steps); the loss,
    clipped against the sampler's own logprobs, corrects for that step of lag."""
    cfg, g = config(), grpo_config()
    start = start_system(round_)[domain]["model"]
    out = ROOT / f"artifacts/logs/r{round_}/grpo/{domain}"
    out.mkdir(parents=True, exist_ok=True)
    events_path = out / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
    if any(e["event"] == "stop" for e in events):
        return

    def log(event):
        events.append({"time": f"{datetime.now():%Y-%m-%dT%H:%M:%S}", **event})
        with open(events_path, "a") as f:
            f.write(json.dumps(events[-1]) + "\n")

    fresh = not events
    if fresh:
        (out / "config.json").write_text(json.dumps({
            "round": round_, "domain": domain, "plugins": plugins, "model": start,
            "agent": cfg["agent"], "user": cfg["user"]["train"],
            "grpo": {k: v for k, v in g.items() if k not in MACHINE}}, indent=1) + "\n")
        log({"event": "start"})
    recs = {r["task"]["id"]: r for r in records(domain, f"round{round_}") if r["task"]["id"] not in g["excluded"]}
    profile = outcomes(Results.load(ROOT / f"artifacts/logs/r{round_}/profile/{domain}.json"))
    plan = [(epoch, batch) for epoch in range(1, g["epochs"] + 1) for batch in batches(recs, epoch, g)]
    done = {e["step"]: e for e in events if e["event"] == "step"}
    # task -> [successes, runs] in each step's rollouts
    solved = {s: e["successes"] for s, e in done.items()}
    checkpoints = ROOT / f"artifacts/checkpoints/r{round_}/{domain}"

    def adapter(step):
        return out / "steps" / f"s{step:02d}" / "adapter"

    def model(step):
        return start if step <= 2 else adapter_name(domain, round_, step - 2)

    def roll(step):
        """The step's rollouts per task: by the profile in the first epoch, by the previous epoch's rollouts after."""
        epoch, batch = plan[step - 1]
        if epoch == 1:
            latest = {t: (sum(w), len(w)) for t, w in profile.items()}
        else:
            latest = {t: tuple(v) for s, (e, _) in enumerate(plan, 1) if e == epoch - 1 for t, v in solved[s].items()}
        by_count = {}
        for t in batch:
            by_count.setdefault(rollouts_for(*latest[t], g["rollouts"]), []).append(recs[t])
        if step > 2:
            serve(model(step), adapter(step - 2))
        print(f"step {step}: {model(step)}; " + ", ".join(f"{len(v)} tasks x {n}" for n, v in by_count.items()))
        return by_count, rollouts(domain, round_, step, by_count, plugins, model(step), out / "steps" / f"s{step:02d}")

    if fresh:
        print(f"{domain} GRPO: {len(plan)} steps over {len(recs)} tasks, logging to {out / 'summary.log'}")
    with (open(out / "run.log", "a", buffering=1) as f, open(out / "summary.log", "a", buffering=1) as brief,
          redirect_stdout(f), redirect_stderr(f), ThreadPoolExecutor(1) as pool):
        server(g["server_gpus"])
        pending = {}
        for step, (epoch, _) in enumerate(plan, 1):
            if step in done:
                continue
            step_dir = out / "steps" / f"s{step:02d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            by_count, results = (pending.pop(step) if step in pending else pool.submit(roll, step)).result()
            solved[step] = {t: [sum(w), len(w)] for r in results for t, w in outcomes(r).items()}
            if step < len(plan):
                pending[step + 1] = pool.submit(roll, step + 1)
            if not adapter(step).exists():
                train(g, step_dir, adapter(step - 1) if step > 1 else ROOT / start if "/" in start else None,
                      [step_dir / f"g{n:02d}.json" for n in by_count], f)
            name = f"epoch{step / g['steps_per_epoch']:g}"
            checkpoint = step / g["steps_per_epoch"] in g["checkpoints"]
            if checkpoint:
                shutil.rmtree(checkpoints / name, ignore_errors=True)
                shutil.copytree(adapter(step), checkpoints / name, ignore=shutil.ignore_patterns("optimizer.pt"))
            metrics = json.loads((adapter(step) / "train.json").read_text())
            log({"event": "step", "step": step, "epoch": epoch, "model": model(step),
                 "rollouts": {n: len(v) for n, v in by_count.items()},
                 "conversations": sum(n * len(v) for n, v in by_count.items()),
                 "solved": sum(s for s, _ in solved[step].values()), "successes": solved[step],
                 "empty_replies": empty_replies(results), "train": metrics})
            done[step] = events[-1]
            # no rollouts and no update use it any more; a checkpoint's adapter is kept as its copy
            shutil.rmtree(adapter(step - 2), ignore_errors=True)
            brief.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} step {step} (epoch {epoch}, rollouts of {model(step)}): "
                        f"{events[-1]['solved']}/{events[-1]['conversations']} solved, {metrics['informative']} of "
                        f"{metrics['groups']} groups informative"
                        + (f", {events[-1]['empty_replies']} ended on an empty reply" if events[-1]["empty_replies"]
                           else "")
                        + f"; loss {metrics['loss']:.4f}, largest grad norm {max(metrics['grad_norms']):.3f}"
                        + (f", {metrics['packs_left_out']} packs left out (largest gradient per token "
                           f"{metrics['max_token_grad']:.0f})" if metrics["packs_left_out"] else "")
                        + f", logprob gap {metrics['logprob_gap']:.4f}, clipped {metrics['clipped']:.4f}"
                        + (f", {metrics['dropped_calls']} calls left out" if metrics["dropped_calls"] else "")
                        + (f"; checkpoint {name}" if checkpoint else "") + "\n")
    shutil.rmtree(adapter(len(plan) - 1), ignore_errors=True)
    log({"event": "stop", "adapter": str(adapter(len(plan)).relative_to(ROOT))})
    print(f"{domain} GRPO: done, checkpoints {', '.join(p.name for p in sorted(checkpoints.glob('epoch*')))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--domain", choices=["retail", "airline"], help="one domain only; both by default")
    args = parser.parse_args()
    plugins = [harness_rsi(d, args.round) for d in ["airline", "retail"]]
    for domain in [args.domain] if args.domain else ["airline", "retail"]:
        grpo(domain, args.round, plugins)


if __name__ == "__main__":
    main()
