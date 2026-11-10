"""Running tasks on a system (both domains' plugins and a served model) with a user simulator, and the scores that
compare runs. Settings in config/harness_rsi_config.toml.
"""
import json
import os
import random
import resource
import shutil
import signal
import subprocess
import threading
import time
import tomllib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import litellm
import numpy as np
import tau2.runner.batch
import tau2.user.user_simulator
from tau2.data_model.simulation import Results, TerminationReason, TextRunConfig
from tau2.data_model.tasks import Task
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.environment import get_environment as airline_environment
from tau2.domains.airline.utils import AIRLINE_DB_PATH
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import get_environment as retail_environment
from tau2.domains.retail.utils import RETAIL_DB_PATH
from tau2.metrics.agent_metrics import is_successful, pass_hat_k
from tau2.registry import registry
from tau2.runner.batch import run_tasks

import rsi.agent.plugin  # noqa: F401  registers the plugin agent with tau2
from rsi.agent.plugin import ROOT

CONFIG = ROOT / "config/harness_rsi_config.toml"
SERVER = ROOT / "scripts/serve_gemma.sh"
POOL = {"retail": ROOT / "data/processed/retail_final.jsonl", "airline": ROOT / "data/processed/airline_final.jsonl"}
OFFICIAL_DB = {"retail": RETAIL_DB_PATH, "airline": AIRLINE_DB_PATH}
DB = {"retail": (RetailDB, retail_environment), "airline": (FlightDB, airline_environment)}
OTHER = {"retail": "airline", "airline": "retail"}
# tau2 records the git commit of the working directory, which is this repository; the results record the pinned
# tau2-bench commit instead, the benchmark version they were run on
TAU2_COMMIT = next(d for d in tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
                   if d.startswith("tau2 ")).rsplit("@", 1)[1]
# DeepSeek's price (official page, 2026-10-08), for when litellm's bundled table, used if its download fails, lacks it
DEEPSEEK = {"input_cost_per_token": 0.30e-6, "output_cost_per_token": 1.20e-6, "cache_read_input_token_cost": 0.006e-6,
            "litellm_provider": "deepseek", "mode": "chat"}

# tau2 caps litellm at 10 connections per process (utils/llm_utils.py), too few to keep the GPU busy. A conversation in
# flight holds one connection, so the cap stays above the configs' largest concurrency, or it caps that instead
litellm.client_session = httpx.Client(limits=httpx.Limits(max_connections=512, max_keepalive_connections=64))
# A pipeline process keeps hundreds of conversations' connections and files open for a day or more: the usual soft
# limit of 1,024 open files runs out
SOFT, HARD = resource.getrlimit(resource.RLIMIT_NOFILE)
if 0 <= SOFT < 65536:
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536 if HARD == resource.RLIM_INFINITY else min(65536, HARD), HARD))


class ThreadRandom(threading.local):
    """Per-thread stand-in for the global random module, from which tau2's runner/batch.py draws a run's trial seeds.
    run() starts runs side by side in threads, which would otherwise reseed each other's draws, and a resumed run only
    skips the trials whose seeds match. Gives the same seeds as the global module."""

    def __init__(self):
        rng = random.Random()
        self.seed, self.randint = rng.seed, rng.randint


tau2.runner.batch.random = ThreadRandom()

# DeepSeek keeps the connection of a request it has queued alive with blank lines, so litellm's 600 s timeout, counted
# from the last byte received, never fires: one user-simulator call held a whole step for over an hour. Each call gets
# a deadline from its start instead; past it the call raises and tau2 runs the conversation again. The call's thread is
# left to end on its own (a daemon, so it never holds the process at exit).
USER_DEADLINE = 600
_user_generate = tau2.user.user_simulator.generate


def user_generate(*args, **kwargs):
    out = {}

    def call():
        try:
            out["reply"] = _user_generate(*args, **kwargs)
        except Exception as e:
            out["error"] = e

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    thread.join(USER_DEADLINE)
    if thread.is_alive():
        raise TimeoutError(f"no reply from the user simulator within {USER_DEADLINE} s")
    if "error" in out:
        raise out["error"]
    return out["reply"]


tau2.user.user_simulator.generate = user_generate


def config():
    return tomllib.loads(CONFIG.read_text())


def records(domain, split):
    """Task records ({task, source, source_id, db}) of one key of data/splits/<domain>.json, in split order."""
    by_id = {}
    with open(POOL[domain]) as f:
        for line in f:
            rec = json.loads(line)
            by_id[rec["task"]["id"]] = rec
    return [by_id[i] for i in json.loads((ROOT / "data/splits" / f"{domain}.json").read_text())[split]]


def start_system(round_, logs=ROOT / "artifacts/logs"):
    """S(k-1), the system round k starts from: domain -> its plugin and its model (the base model's name, or a LoRA
    adapter's directory in the repository), as the latest accepted plugin gate and model gate of the earlier rounds
    saved them; [system] of the config where none did."""
    cfg = config()["system"]
    system = {d: {"plugin": cfg[f"{d}_plugin"], "model": cfg["model"]} for d in ("retail", "airline")}
    for r in range(1, round_):
        for d, s in system.items():
            for gate, key in ((f"plugin_{d}", "plugin"), (f"plugin_{d}_retry", "plugin"), (f"model_{d}", "model")):
                path = logs / f"r{r}/gate/{gate}/decision.json"
                decision = json.loads(path.read_text()) if path.exists() else {}
                if decision.get("accepted"):
                    s[key] = decision["saved"]
    return system


# runs side by side in threads (GRPO's rollouts, the model gate) can meet an unregistered snapshot at once, and tau2
# refuses to register a name twice
REGISTERING = threading.Lock()


def env_domain(domain, db_path):
    """tau2 domain running on a task's DB snapshot, registered on first use. tau2's evaluator rebuilds the predicted
    and the reference environment from the constructor registered under the environment's name, so every call must get
    its own copy of the DB: on a shared one every trajectory matches the reference."""
    if (ROOT / db_path).resolve() == Path(OFFICIAL_DB[domain]).resolve():
        return domain
    name = f"{domain}-{Path(db_path).stem}"
    with REGISTERING:
        if name not in registry.get_domains():
            db_class, get_environment = DB[domain]
            db = db_class.load(ROOT / db_path)

            def environment(solo_mode=False, **_):
                env = get_environment(db=db.model_copy(deep=True), solo_mode=solo_mode)
                env.domain_name = name
                return env

            registry.register_domain(environment, name)
    return name


def environment(domain, record):
    """A fresh environment on the record's DB snapshot."""
    return registry.get_env_constructor(env_domain(domain, record["db"]))()


def agent_args(cfg, plugins):
    a = cfg["agent"]
    args = {"api_base": cfg["server"]["api_base"], "api_key": "EMPTY",
            **{k: a[k] for k in ("temperature", "top_p", "top_k", "max_tokens")}}
    if plugins:
        args.update(plugins=[str(p) for p in plugins], max_skill_loads=a["max_skill_loads"])
    return args


def save(data, path):
    """Write JSON Results data with the run info, each task and each conversation on a line of its own: tau2's own
    save puts every field on its own line, millions of lines per run. The rename keeps readers from a half-written
    file."""
    fields = []
    for key, value in data.items():
        if isinstance(value, list):
            value = "[\n" + ",\n".join(json.dumps(x, ensure_ascii=False) for x in value) + "\n]"
        else:
            value = json.dumps(value, ensure_ascii=False)
        fields.append(f"{json.dumps(key)}: {value}")
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text("{\n" + ",\n".join(fields) + "\n}\n")
    tmp.replace(path)


def merge(paths, domain):
    """One Results of a domain from runs over disjoint tasks."""
    runs = [Results.load(p) for p in paths]
    info = runs[0].info.model_copy(deep=True)
    info.environment_info.domain_name = domain
    info.git_commit = TAU2_COMMIT
    return Results(timestamp=runs[0].timestamp, info=info, tasks=[t for r in runs for t in r.tasks],
                   simulations=[s for r in runs for s in r.simulations])


def serve(name, path):
    """Load a LoRA adapter into the model server under `name`; one already loaded stays. A relative path is relative to
    the repository, where scripts/serve_gemma.sh starts the server."""
    r = httpx.post(f"{config()['server']['api_base']}/load_lora_adapter",
                   json={"lora_name": name, "lora_path": str(path)}, timeout=600)
    if r.status_code != 200 and "already" not in r.text:
        raise RuntimeError(f"loading adapter {name}: {r.status_code} {r.text}")


def served(model):
    """The name the model server knows a model by: a served name as it is; a LoRA adapter given as its directory in
    the repository, artifacts/checkpoints/r<k>/<domain>/<checkpoint>, is loaded as <domain>-r<k>-<checkpoint>."""
    if "/" not in model:
        return model
    *_, round_dir, domain, checkpoint = Path(model).parts
    serve(f"{domain}-{round_dir}-{checkpoint}", model)
    return f"{domain}-{round_dir}-{checkpoint}"


def server_pids():
    """PIDs of the model server: the processes whose argv is <python> .../vllm serve ..., as scripts/serve_gemma.sh
    starts it. A command line that only mentions "vllm serve", such as a shell's that greps for it, is not one."""
    pids = []
    for proc in Path("/proc").iterdir():
        if proc.name.isdigit():
            try:
                argv = (proc / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if len(argv) > 2 and argv[1].endswith(b"/vllm") and argv[2] == b"serve":
                pids.append(int(proc.name))
    return pids


def server(gpus):
    """On the GPU server: make the model server run on `gpus`, a replica on each. One already serving there stays; any
    other stops and a new one starts (scripts/serve_gemma.sh, logging to [server] log), with no adapter loaded.
    Returns once it answers."""
    cfg = config()["server"]
    pids = server_pids()

    def devices(pid):
        env = Path(f"/proc/{pid}/environ").read_bytes().decode(errors="replace").split("\0")
        return next((x.split("=", 1)[1] for x in env if x.startswith("CUDA_VISIBLE_DEVICES=")), None)

    def free():
        used = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True).stdout
        used = dict(line.split(", ") for line in used.splitlines())
        return all(int(used[i]) < 1024 for i in gpus.split(","))

    if pids and all(devices(p) == gpus for p in pids):
        return
    for pid in pids:
        os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 600
    while any(Path(f"/proc/{p}").exists() for p in pids) or not free():
        if time.time() > deadline:
            raise RuntimeError(f"the model server did not stop, or GPUs {gpus} stay busy")
        time.sleep(5)
    # detached into its own session, so that it outlives the pipeline
    subprocess.run(["sh", "-c", f"GPUS={gpus} setsid {SERVER} >> {cfg['log']} 2>&1 < /dev/null &"], check=True)
    deadline = time.time() + 1200
    while time.time() < deadline:
        try:
            if httpx.get(f"{cfg['api_base']}/models", timeout=5).status_code == 200:
                print(f"model server up on GPUs {gpus}")
                return
        except httpx.HTTPError:
            pass
        time.sleep(5)
    raise RuntimeError(f"the model server on GPUs {gpus} did not come up; see {cfg['log']}")


def run(domain, recs, plugins, model, user, trials, seed, save_path, concurrency=None, agent_extra=None,
        max_steps=None):
    """tau2 Results of a system on task records of a domain. plugins: both domains' plugin directories, or [] for
    tau2's own agent; model: a served name or an adapter directory (served). A finished save_path is returned as is;
    an interrupted run resumes. Tasks on different DB snapshots run as separate tau2 domains side by side, sharing the
    concurrency ([run] concurrency by default). agent_extra: more arguments of the agent and its model calls;
    max_steps: turns before tau2 ends a conversation ([run] max_steps by default)."""
    save_path = Path(save_path)
    if save_path.exists():
        return Results.load(save_path)
    model = served(model)
    cfg = config()
    litellm.register_model({f"openai/{model}": {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0,
                                                "litellm_provider": "openai", "mode": "chat"}})
    if user["model"].startswith("deepseek/") and user["model"] not in litellm.model_cost:
        litellm.register_model({user["model"]: DEEPSEEK})
    groups = defaultdict(list)
    for r in recs:
        groups[env_domain(domain, r["db"])].append(Task.model_validate(r["task"]))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def one(name, tasks):
        # tau2's dir format, a file per conversation: its JSON format rereads and rewrites the whole file after every
        # conversation, which with the token ids and logprobs of GRPO rollouts takes longer than the conversations
        part = save_path.parent / f"{save_path.stem}.{name}" / "results.json"
        conc = max(1, round((concurrency or cfg["run"]["concurrency"]) * len(tasks) / len(recs)))
        run_tasks(TextRunConfig(domain=name, agent="plugin_agent" if plugins else "llm_agent", llm_agent=f"openai/{model}",
                                llm_args_agent={**agent_args(cfg, plugins), **(agent_extra or {})},
                                llm_user=user["model"], llm_args_user=user["args"], num_trials=trials, seed=seed,
                                max_concurrency=conc, max_steps=max_steps or cfg["run"]["max_steps"],
                                auto_resume=True), tasks, save_path=part, results_format="dir")
        return part

    with ThreadPoolExecutor(len(groups)) as pool:
        parts = list(pool.map(lambda kv: one(*kv), sorted(groups.items())))
    results = merge(parts, domain)
    # A conversation tau2 still fails after its retries counts as a failure. Many such that the agent's own output did
    # not cause (an empty message, too many skill loads) means the model server or the user simulator was unreachable:
    # stop and keep the parts, whose resume reruns them.
    errors = sum(s.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR
                 and (s.info or {}).get("error_type") not in ("ValueError", "AgentError") for s in results.simulations)
    if errors > 0.02 * len(results.simulations):
        raise RuntimeError(f"{errors} of {len(results.simulations)} conversations ended in infrastructure errors; run "
                           "again once the model server and the user simulator are reachable")
    save(json.loads(results.model_dump_json()), save_path)
    for p in parts:
        shutil.rmtree(p.parent)
    return results


def outcomes(results):
    """task id -> success of each trial, in trial order. A trial tau2 gave up on (no reward) counts as a failure."""
    out = defaultdict(list)
    for s in sorted(results.simulations, key=lambda s: (s.task_id, s.trial)):
        out[s.task_id].append(s.reward_info is not None and is_successful(s.reward_info.reward))
    return dict(out)


def pass_k(results, k):
    """task id -> tau2's pass^k estimate; k may also be {k: weight}, for the weighted sum of those estimates."""
    if isinstance(k, dict):
        parts = {j: pass_k(results, j) for j in k}
        return {t: sum(w * parts[j][t] for j, w in k.items()) for t in parts[next(iter(k))]}
    return {t: pass_hat_k(len(x), sum(x), k) for t, x in outcomes(results).items()}


def paired(current, candidate, k, boot, seed=0):
    """Mean pass^k (k as in pass_k) of both systems over the same tasks, the difference, and its 95% bootstrap interval
    over tasks."""
    a, b = pass_k(current, k), pass_k(candidate, k)
    d = np.array([b[t] - a[t] for t in sorted(a)])
    means = np.random.default_rng(seed).choice(d, (boot, len(d))).mean(1)
    return {"current": float(np.mean(list(a.values()))), "candidate": float(np.mean(list(b.values()))),
            "diff": float(d.mean()), "ci95": [float(x) for x in np.percentile(means, [2.5, 97.5])]}


def run_seed(round_, kind, attempt=0, step=0):
    """Seed of one run: kind 0 the collection, 1 a step of plugin evolution, 2 a plugin gate, 3 the difficulty profile,
    4 the rollouts of a GRPO step, 5 a model gate in a round without a plugin gate, 6 the model gate's screening of the
    checkpoints. Every run gets its own seed, because under one seed tau2 gives the i-th trial of every task the same
    sampling seed, so a rerun would partly replay earlier conversations; the two systems of one comparison share
    theirs."""
    return 1_000_000 * round_ + 100_000 * kind + 1_000 * attempt + step
