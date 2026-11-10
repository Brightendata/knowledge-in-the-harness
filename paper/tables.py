"""The paper's tables and statistics, from the released records: results/ (official tasks), artifacts/logs/ (collection,
plugin evolution, gates, profiling passes, GRPO) and data/test_annotation.json.

    python paper/tables.py

Prints the statistics section by section, in the order of the paper. Confidence intervals are paired bootstrap intervals
over tasks (10,000 resamples, seed 0); simulations use seed 0. Runs in a few minutes on a laptop.
"""
import gzip
import json
import random
import re
import statistics as st
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln
from scipy.stats import beta as beta_dist, binom, norm

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "artifacts/logs"
D = ["retail", "airline"]
TRAJ = ["H0-M0", "H1-M0", "H1-M1", "H2-M1", "H2-M2"]
SYSTEMS = ["official_setting", *TRAJ, "H2-M0", "H0-M2"]
FRONTIER = ["H0-gpt-5.6-sol", "H0-claude-sonnet-5", "H0-glm-5.2"]
ann = json.loads((ROOT / "data/test_annotation.json").read_text())
rng = np.random.default_rng(0)


def section(title):
    print(f"\n## {title}\n")


def read(path):
    return json.loads(Path(path).read_text())


def load(system, domain):
    with gzip.open(ROOT / f"results/{system}/{domain}.json.gz", "rt") as f:
        return json.load(f)["simulations"]


def ok(sim):
    return (sim.get("reward_info") or {}).get("reward", 0) == 1


sims = {(v, d): load(v, d) for v in SYSTEMS + FRONTIER for d in D}
succ = {key: Counter() for key in sims}
for key, ss in sims.items():
    for s in ss:
        succ[key][s["task_id"]] += ok(s)
tasks = {d: sorted(succ["H0-M0", d], key=int) for d in D}


def pk(v, d, k, ts=None):
    return 100 * st.mean(comb(succ[v, d][t], k) / comb(4, k) for t in ts or tasks[d])


def ci(terms, d, b=10000, seed=0, ts=None):
    """Point estimate and 95% paired-bootstrap interval of sum(w * pass^k(v)) over terms [(w, v, k)], on the tasks ts
    (default: all official tasks of the domain)."""
    r = random.Random(seed)
    per = [sum(w * comb(succ[v, d][t], k) / comb(4, k) for w, v, k in terms) for t in ts or tasks[d]]
    boots = sorted(100 * st.mean(r.choices(per, k=len(per))) for _ in range(b))
    return 100 * st.mean(per), boots[int(0.025 * b)], boots[int(0.975 * b) - 1]


def fmt(e):
    return f"{e[0]:+.1f} [{e[1]:+.1f}, {e[2]:+.1f}]"


def diff(a, b, k):
    return [(1, a, k), (-1, b, k)]


def p(xs, k):
    return comb(sum(xs), k) / comb(len(xs), k)


# ---------------------------------------------------------------------------------------------------------------------
section("Official tasks: pass^1 / pass^2 / pass^3 / pass^4 (main results; pass^k table)")
for v in SYSTEMS + FRONTIER:
    print(f"{v:20s} " + " | ".join(" / ".join(f"{pk(v, d, k):.1f}" for k in (1, 2, 3, 4)) for d in D))

section("Comparisons with 95% CIs, pass^1; pass^4 (retail | airline)")
for a, b in [("H2-M2", "official_setting"), ("H2-M2", "H0-M0"), ("H0-M0", "official_setting"),
             ("H1-M0", "official_setting"), ("H2-M0", "official_setting"), ("H2-M0", "H0-M0"), ("H0-M2", "H0-M0"),
             ("H2-M2", "H2-M0"), ("H2-M2", "H0-M2")]:
    print(f"{a} - {b}: " + " | ".join(f"{fmt(ci(diff(a, b, 1), d))}; {fmt(ci(diff(a, b, 4), d))}" for d in D))
inter = lambda k: [(1, "H2-M2", k), (-1, "H2-M0", k), (-1, "H0-M2", k), (1, "H0-M0", k)]
print("interaction: " + " | ".join(f"{fmt(ci(inter(1), d))}; {fmt(ci(inter(4), d))}" for d in D))
for v in ("H0-M0", "H2-M2"):
    sig = sum(not (lo <= 0 <= hi) for f in FRONTIER for d in D for k in (1, 4) for _, lo, hi in [ci(diff(v, f, k), d)])
    print(f"{v} vs the frontier models: {sig} of 12 intervals exclude zero")
for d in D:
    for k in (1, 4):
        mean = st.mean(pk(f, d, k) for f in FRONTIER)
        ref, fin = pk("official_setting", d, k), pk("H2-M2", d, k)
        print(f"{d} pass^{k}: H2-M2 closes {100 * (fin - ref) / (mean - ref):.0f}% of the gap from the reference agent "
              f"({ref:.1f}) to the frontier mean ({mean:.1f})")
print("accepted updates, official pass^1 / pass^4 change: " + "; ".join(
    f"{d} {a}->{b} {pk(b, d, 1) - pk(a, d, 1):+.1f}/{pk(b, d, 4) - pk(a, d, 4):+.1f}"
    for d in D for a, b in zip(TRAJ, TRAJ[1:])))
for f in FRONTIER:
    print(f"agent cost per conversation, {f}: " + " | ".join(
        f"{d} ${st.mean(s.get('agent_cost') or 0 for s in sims[f, d]):.3f}" for d in D))

# ---------------------------------------------------------------------------------------------------------------------
section("Mean success by the number of H0-M0 successes (rules rescue, weights retain)")
for d in D:
    for name, lo, hi in (("0/4", 0, 0), ("1-3/4", 1, 3), ("4/4", 4, 4)):
        ts = [t for t in tasks[d] if lo <= succ["H0-M0", d][t] <= hi]
        row = {v: pk(v, d, 1, ts) for v in ("official_setting", "H0-M0", "H2-M0", "H0-M2", "H2-M2")}
        print(f"{d:7s} {name:5s} n={len(ts):3d}: reference {row['official_setting']:.1f}, H0-M0 {row['H0-M0']:.1f}, "
              f"H2-M0 {row['H2-M0']:.1f}, H0-M2 {row['H0-M2']:.1f}, H2-M2 {row['H2-M2']:.1f}")
for d in D:
    never = [t for t in tasks[d] if succ["H0-M0", d][t] == 0]
    always = [t for t in tasks[d] if succ["H0-M0", d][t] == 4]
    print(f"{d}: rules minus weights (H2-M0 - H0-M2) on never-solved tasks "
          f"{fmt(ci([(1, 'H2-M0', 1), (-1, 'H0-M2', 1)], d, ts=never))}; weights minus rules on always-solved tasks "
          f"{fmt(ci([(1, 'H0-M2', 1), (-1, 'H2-M0', 1)], d, ts=always))}")
contrast = lambda: [(1, "H0-M2", 4), (-1, "H0-M2", 1), (-1, "H2-M0", 4), (1, "H2-M0", 1)]
print("weights minus harness from H0-M0, (gain in pass^4 - gain in pass^1): "
      + " | ".join(f"{d} {fmt(ci(contrast(), d))}" for d in D))
# Split-half bins: bin on two H0-M0 trials and score H0-M0 on the other two, so its column is free of selection.
for d in D:
    trial = defaultdict(dict)
    for s in sims["H0-M0", d]:
        trial[s["task_id"]][s["trial"]] = ok(s)
    for bin_on, score_on in (((0, 1), (2, 3)), ((2, 3), (0, 1))):
        for name, hits in (("never", 0), ("always", 2)):
            ts = [t for t in tasks[d] if sum(trial[t][i] for i in bin_on) == hits]
            held = 100 * st.mean(trial[t][i] for t in ts for i in score_on)
            print(f"{d} split-half, {name} on trials {bin_on} (n={len(ts)}): H0-M0 on trials {score_on} {held:.1f}, "
                  + ", ".join(f"{v} {pk(v, d, 1, ts):.1f}" for v in ("H2-M0", "H0-M2", "H2-M2")))

section("pass^1 by annotated knowledge point, n >= 10 plus price_calc and bags_add: reference, H0-M0, H2-M0, H0-M2, H2-M2")
LABELS = {  # tag in data/test_annotation.json -> name in the paper's table
    "find_res": "Find a reservation", "fallback": "Fallback condition", "multi": "Multiple requests",
    "cancel_refuse": "Refused cancel", "pressure": "Persistent user", "cabin_change": "Change cabin",
    "search_direct": "Direct flights", "cabin_diff": "Fare difference", "false_claim": "False user claim",
    "mod_flights": "Change flights", "mod_pay": "Change payment", "bag_allowance": "Baggage allowance",
    "budget": "Budget condition", "price_calc": "Price computation", "bags_add": "Adding bags",
    "mod_items": "Modify items", "return": "Return", "var_pref": "Preferred variant", "exchange": "Exchange",
    "reactive": "Reactive user", "info_amount": "State an amount", "compute": "Compute a total",
    "order_info": "Order information", "status": "Status mismatch", "var_extreme": "Extreme variant",
    "mod_addr": "Order address", "addr_from_order": "Address from order", "cancel": "Cancel",
    "var_exact": "Exact variant", "cancel_partial_no": "No partial cancel", "scope_all": "Scope over all",
    "change_mind": "Change of mind", "avail": "Availability check", "items_once": "One call per order",
    "find_order": "Find an order", "order": "Order of requests", "same_name": "Same-name items",
    "deny": "Out-of-policy ask", "emotion": "Emotional user", "cond_value": "Value condition",
    "gc_balance": "Gift-card balance", "scope_except": "All except some", "user_addr": "Default address",
    "var_relative": "Relative variant", "var_relaxed": "Relaxed variant",
}
for d in D:
    count = Counter(t for k in tasks[d] for t in ann[d][k])
    for g, n in sorted(count.items(), key=lambda x: (-x[1], x[0])):
        if n >= 10 or (d == "airline" and g in ("price_calc", "bags_add")):
            ts = [t for t in tasks[d] if g in ann[d][t]]
            print(f"{d:7s} {g:18s} {LABELS.get(g, '(outcome type)'):19s} {n:3d}  " + "  ".join(
                f"{pk(v, d, 1, ts):5.1f}" for v in ("official_setting", "H0-M0", "H2-M0", "H0-M2", "H2-M2")))


def loads(sim):
    return {l["name"] for m in sim["messages"] if m["role"] == "assistant"
            for l in (m.get("raw_data") or {}).get("skill_loads", [])}


section("Airline: the reference agent's lead over H0-M0 by task group, and the baggage-allowance tasks")
gap = {t: (succ["H0-M0", "airline"][t] - succ["official_setting", "airline"][t]) * 25 / len(tasks["airline"])
       for t in tasks["airline"]}
bag = {t for t in tasks["airline"] if {"bag_allowance", "bags_add"} & set(ann["airline"][t])}
cancel = {t for t in tasks["airline"] if {"cancel_refuse", "cancel_ok", "book_insurance"} & set(ann["airline"][t])} - bag
print(f"total {sum(gap.values()):+.1f}; baggage tasks {sum(gap[t] for t in bag):+.1f}; "
      f"other cancellation tasks {sum(gap[t] for t in cancel):+.1f}")
nobag = [t for t in tasks["airline"] if t not in bag]
for k in (1, 4):
    mean = st.mean(pk(f, "airline", k, nobag) for f in FRONTIER)
    ref, fin = pk("official_setting", "airline", k, nobag), pk("H2-M2", "airline", k, nobag)
    print(f"without the {len(bag)} baggage tasks, pass^{k}: H2-M2 closes {100 * (fin - ref) / (mean - ref):.0f}% of the "
          f"gap from the reference agent ({ref:.1f}) to the frontier mean ({mean:.1f})")
indep = [t for t in tasks["airline"] if t not in bag | cancel]
for name, ts in (("split-dependent (baggage, cancellation, insurance)", sorted(bag | cancel, key=int)),
                 ("split-independent", indep), ("cancellation or insurance without baggage", sorted(cancel, key=int))):
    print(f"{name}, {len(ts)} tasks, pass^1: " + ", ".join(
        f"{v} {pk(v, 'airline', 1, ts):.1f}" for v in ("official_setting", "H0-M0", "H1-M0", "H2-M0", "H0-M2", "H2-M2"))
        + f"; H0 costs {pk('official_setting', 'airline', 1, ts) - pk('H0-M0', 'airline', 1, ts):.1f}, H2-M2 minus "
        f"reference {pk('H2-M2', 'airline', 1, ts) - pk('official_setting', 'airline', 1, ts):+.1f}")
print(f"never-solved airline tasks that depend on the split: "
      f"{sum(succ['H0-M0', 'airline'][t] == 0 for t in bag | cancel)} of "
      f"{sum(succ['H0-M0', 'airline'][t] == 0 for t in tasks['airline'])}")
allowance = [t for t in tasks["airline"] if "bag_allowance" in ann["airline"][t]]
for v in ("official_setting", "H0-M0", "H0-M2", "H1-M0", "H2-M0", "H1-M1", "H2-M1", "H2-M2", *FRONTIER):
    rs = [s for s in sims[v, "airline"] if s["task_id"] in allowance]
    if v == "official_setting":
        print(f"{v}: success {sum(map(ok, rs))}/{len(rs)}")
        continue
    b = [ok(s) for s in rs if "airline:book" in loads(s)]
    nb = [ok(s) for s in rs if "airline:book" not in loads(s)]
    print(f"{v}: loads book {100 * len(b) / len(rs):.0f}%; success {sum(b)}/{len(b)} with, {sum(nb)}/{len(nb)} without")

section("Plugin size and the share a conversation reads")
for name, v in (("H0", "H0-M0"), ("H2", "H2-M2")):
    docs = {}
    for d in D:
        path = ROOT / f"plugins/{d}" if name == "H0" else next(LOGS.glob(f"r2/plugins/{d}_final*_accepted"))
        docs[d] = {(d if q.parent.name == "agents" else f"{d}:{q.parent.name}"): len(q.read_text())
                   for q in list(path.glob("agents/*.md")) + list(path.glob("skills/*/SKILL.md"))}
    installed = {k: n for x in docs.values() for k, n in x.items()}
    for d in D:
        read_ = [sum(installed.get(n, 0) for n in loads(s)) for s in sims[v, d]]
        print(f"{d} {name}: {sum(docs[d].values())} characters in {len(docs[d])} documents; a conversation of {v} reads "
              f"{100 * st.mean(read_) / sum(docs[d].values()):.0f}% of them and "
              f"{100 * st.mean(read_) / sum(installed.values()):.1f}% of all installed policy text")
for d in D:
    split = read(ROOT / f"data/splits/{d}.json")
    tag = {}
    with open(ROOT / f"data/processed/{d}_final.jsonl") as f:
        for line in f:
            x = json.loads(line)
            tag[x["task"]["id"]] = set(x["tags"])
    with open(ROOT / f"data/processed/{d}_generated.jsonl") as f:
        n_gen = sum(1 for _ in f)
    kps = {t for k in tasks[d] for t in ann[d][k]}
    cover = {s: [sum(g in tag[t] for t in split[s]) for g in kps] for s in ("validation", "round1", "round2", "round3")}
    print(f"{d}: {n_gen} candidates; {len(kps)} knowledge points of the official tasks; fewest tasks per knowledge point "
          f"in V {min(cover['validation'])}, in a training batch {min(min(cover[s]) for s in ('round1', 'round2', 'round3'))}; "
          f"knowledge points with >= 12 tasks in a training batch: "
          + ", ".join(f"{100 * np.mean(np.array(cover[s]) >= 12):.0f}%" for s in ("round1", "round2", "round3")))

# ---------------------------------------------------------------------------------------------------------------------
section("Plugin evolution: diagnoses, edits, checks, retests, final rules")
rel, ops, total, valid = Counter(), Counter(), 0, 0
for f in sorted(LOGS.glob("r*/harness_rsi/*/steps/s*/analysis.jsonl")):
    for a in map(json.loads, f.open()):
        total += 1
        if a.get("output"):
            valid += 1
            ops[a["output"]["op"]] += 1
            rel[a["output"]["relation"]] += 1
print(f"diagnoses {total}, with a valid reply {valid}; ops {dict(ops)}")
print("root causes: " + ", ".join(f"{k} {v} ({100 * v / valid:.1f}%)" for k, v in rel.items()))
merged = checked = blocked = reverted = no_diagnosis = 0
types = Counter()
for f in sorted(LOGS.glob("r*/harness_rsi/*/steps/s*/merge.json")):
    m = read(f)
    merged += len((m.get("output") or {}).get("edits", []))
    checked += len(m["edits"])
    for x in m["dropped_by_checks"]:
        if x["reason"].startswith("check"):
            blocked += 1
            types.update(set(x["problems"]))
        no_diagnosis += x["reason"].startswith("REPLACE or DELETE without")
    reverted += len(m.get("removed_after_retest", []))
print(f"merged edits {merged}, passed the checks and retested {checked}, blocked by self-review {blocked} "
      f"(at least one review reported: {dict(sorted(types.items()))}), rejected by the code check on rewrites and "
      f"deletions {no_diagnosis}, reverted after retest {reverted}")
steps = kept = 0
for f in sorted(LOGS.glob("r*/harness_rsi/*/events.jsonl")):
    ev = [json.loads(x) for x in f.open()]
    steps += sum(e["event"] == "step" for e in ev)
    kept += sum(e["event"] == "retest" and e["kept"] for e in ev)
final = [m.group(1) for q in LOGS.glob("r2/plugins/*_accepted/**/*.md")
         for m in re.finditer(r"^- \*\*\[([^\]]+)\]\*\* (.*)$", q.read_text(), re.M)]
loading = [m.group(1) for q in LOGS.glob("r2/plugins/*_accepted/**/*.md")
           for m in re.finditer(r"^- \*\*\[([^\]]+)\]\*\* (.*)$", q.read_text(), re.M) if "load" in m.group(2).lower()]
print(f"steps {steps}, kept {kept}; rules in the final plugins {len(final)}, of which loading rules {loading}")

section("Retest operating characteristics on the 60 steps (keep if the candidate gains >= 4 successes in 6 trials)")
steps_c = []
for att in sorted(LOGS.glob("r*/harness_rsi/*")):
    for ev in map(json.loads, (att / "events.jsonl").open()):
        if ev["event"] == "step":
            cur = read(att / "steps" / f"s{ev['step']:02d}" / "outcomes.json")["current"]
            steps_c.append(([sum(cur[t]) for t in ev["failing"]], [sum(cur[t]) for t in ev["regression"]]))
oc = defaultdict(list)
for fail, ctrl in steps_c:
    c, nf = np.array(fail + ctrl), len(fail)
    for _ in range(2000):
        q0 = rng.beta(1 + c, 1 + 6 - c)
        base = rng.binomial(6, q0).sum()
        for name in ("no effect", "effective", "harmful"):
            q = q0.copy()
            if name == "effective":  # halve the failure rate of a random half of the failed tasks
                i = rng.choice(nf, nf // 2, replace=False)
                q[i] = 1 - (1 - q[i]) / 2
            elif name == "harmful":  # halve the success rate of a random half of the control tasks
                i = nf + rng.choice(len(ctrl), len(ctrl) // 2, replace=False)
                q[i] = q[i] / 2
            oc[name].append(rng.binomial(6, q).sum() - base >= 4)
print(", ".join(f"{k}: kept {100 * np.mean(v):.1f}%" for k, v in oc.items()))
# A task losing >= 4 of 6 successes with no true change (first condition of an attributed revert), same noise model.
grid = np.linspace(0.0005, 0.9995, 1000)
pmf = binom.pmf(np.arange(7)[:, None], 6, grid)
drop = sum(pmf[i] * pmf[j] for i in range(7) for j in range(7) if i - j >= 4)
chance = lambda cs: [float((w * drop).sum() / w.sum()) for w in (beta_dist.pdf(grid, 1 + c, 7 - c) for c in cs)]
cf, cc = chance([c for fail, _ in steps_c for c in fail]), chance([c for _, ctrl in steps_c for c in ctrl])
print(f"chance drop of >= 4 successes: {100 * np.mean(cf + cc):.2f}% of {len(cf) + len(cc)} task instances (failed "
      f"tasks {100 * np.mean(cf):.2f}%, controls {100 * np.mean(cc):.2f}%); at most {100 * drop.max():.2f}%")

# ---------------------------------------------------------------------------------------------------------------------
section("GRPO: rollouts, filtered packed sequences, optimizer steps")
conv = sum(e["conversations"] for f in LOGS.glob("r*/grpo/*/events.jsonl") for e in map(json.loads, f.open())
           if e["event"] == "step")
packs = [json.loads(x) for f in sorted(LOGS.glob("r*/grpo/*/packs.jsonl")) for x in f.open()]
g = np.array([x["grad_per_token"] for x in packs])
print(f"rollout conversations {conv}; packed sequences {len(packs)}; per-token gradient median {np.median(g):.2f}, "
      f"max {g.max():.0f}; above 30: {(g > 30).sum()} ({100 * (g > 30).mean():.2f}%)")
shares, nsteps = [], 0
for f in sorted(LOGS.glob("r*/grpo/*/packs.jsonl")):
    by = defaultdict(list)
    for x in map(json.loads, f.open()):
        by[x["update"]].append(x)
    nsteps += len(by)
    for xs in by.values():
        tot = sum(x["grad_norm"] ** 2 for x in xs)
        shares.append(sum(x["grad_norm"] ** 2 for x in xs if x["grad_per_token"] > 30) / tot if tot else 0)
print(f"optimizer steps {nsteps}; packs above 30 would contribute on average {100 * np.mean(shares):.0f}% of a step's "
      f"squared gradient norm")
norms = [float(m.group(1)) for f in LOGS.glob("r*/grpo/*/summary.log") for line in f.open()
         for m in [re.search(r"largest grad norm ([\d.]+)", line)] if m]
print(f"largest gradient norm before clipping over all steps: {max(norms):.2f}")
tok = {f"{f.parts[-4]} {f.parts[-2]}": [json.loads(x)["tokens"] for x in f.open()]
       for f in sorted(LOGS.glob("r*/grpo/*/packs.jsonl"))}
print("packed sequence length: " + ", ".join(f"{k} {np.mean(v) / 1e3:.1f}k" for k, v in tok.items())
      + f"; all {np.mean([t for v in tok.values() for t in v]) / 1e3:.1f}k")
for v in TRAJ:
    calls, sep, packed = [], [], []
    for d in D:
        for s in sims[v, d]:
            ms = [m for m in s["messages"] if m["role"] == "assistant" and m.get("usage")]
            if not ms:  # a run that stopped on an infrastructure error before its first message (scored 0)
                continue
            us = [m["usage"] for m in ms]  # a message's usage sums its model calls, one more per skill load
            calls.append(len(ms) + sum(len((m.get("raw_data") or {}).get("skill_loads", [])) for m in ms))
            sep.append(sum(u["prompt_tokens"] + u["completion_tokens"] for u in us))
            packed.append(us[-1]["prompt_tokens"] + sum(u["completion_tokens"] for u in us))
    print(f"{v} on the official tasks: {np.mean(calls):.1f} model calls per conversation, {np.mean(sep) / 1e3:.1f}k "
          f"tokens processed call by call, {np.mean(packed) / 1e3:.1f}k as one packed sequence (last prompt plus all "
          f"completions)")

section("Harness updates as a curriculum, and GRPO's signal")


def cls(v):
    return "never" if sum(v) == 0 else "always" if sum(v) == len(v) else "mixed"


for r in (1, 2):
    for d in D:
        col, pro = read(LOGS / f"r{r}/collect/{d}.outcomes.json"), read(LOGS / f"r{r}/profile/{d}.outcomes.json")
        lifted = {t for t in col if cls(col[t]) == "never" and cls(pro[t]) != "never"}
        never_c = sum(cls(v) == "never" for v in col.values())
        never_p = {t for t in pro if cls(pro[t]) == "never"}
        groups, from_lifted, from_never, s_l, n_l = Counter(), Counter(), Counter(), 0, 0
        for ev in map(json.loads, (LOGS / f"r{r}/grpo/{d}/events.jsonl").open()):
            if ev["event"] == "step":
                for t, (s, k) in ev["successes"].items():
                    if 0 < s < k:
                        groups[ev["epoch"]] += 1
                        from_lifted[ev["epoch"]] += t in lifted
                        from_never[ev["epoch"]] += t in never_p
                    if ev["epoch"] == 1 and t in lifted:
                        s_l, n_l = s_l + s, n_l + k
        print(f"r{r} {d}: never solved {never_c} in the collection -> {len(never_p)} in the profile; lifted {len(lifted)}, "
              f"which succeed in {100 * s_l / max(n_l, 1):.0f}% of epoch-1 rollouts and give {from_lifted[1]} of "
              f"{groups[1]} mixed groups; profile-never tasks with a mixed group in epoch 1: {from_never[1]} of {len(never_p)}")
tags = {}
for line in open(ROOT / "data/processed/airline_final.jsonl"):
    x = json.loads(line)
    tags[x["task"]["id"]] = set(x["tags"])
col, pro = read(LOGS / "r1/collect/airline.outcomes.json"), read(LOGS / "r1/profile/airline.outcomes.json")
never = [t for t in col if cls(col[t]) == "never"]
lifted = [t for t in never if cls(pro[t]) != "never"]
split = lambda t: bool({"bag_allowance", "bags_add", "cancel_refuse", "cancel_ok", "book_insurance"} & tags[t])
print(f"r1 airline, tasks involving baggage, cancellation, or insurance: {sum(map(split, never))} of the {len(never)} "
      f"never solved in the collection, {sum(map(split, lifted))} of the {len(lifted)} lifted")

section("GRPO on V by task class (classified by an independent run of the previous system), all three checkpoints")
agg = defaultdict(lambda: np.zeros(5))
for r in (1, 2):
    for d in D:
        dec = read(LOGS / f"r{r}/gate/model_{d}/decision.json")
        control = Path(dec["current"])
        arms = read(ROOT / control.parent / "outcomes.json")
        cls_run, ctrl = arms["current"]["rewards"], arms[control.stem]["rewards"]
        chk = read(LOGS / f"r{r}/gate/model_{d}/outcomes.json")
        for e in ("epoch3", "epoch3.5", "epoch4"):
            for t, v in cls_run.items():
                agg[cls(v)] += [1, p(ctrl[t], 1), p(ctrl[t], 4), p(chk[e]["rewards"][t], 1), p(chk[e]["rewards"][t], 4)]
for c, (m, a1, a4, b1, b4) in agg.items():
    print(f"{c}: {100 * a1 / m:.1f} / {100 * a4 / m:.1f} -> {100 * b1 / m:.1f} / {100 * b4 / m:.1f}")

# ---------------------------------------------------------------------------------------------------------------------
section("Noise model of the gates (beta-binomial prior fitted to all V runs of a domain)")


def betabinom_nll(log_ab, counts, n):
    a, b = np.exp(log_ab)
    return -np.sum(gammaln(n + 1) - gammaln(counts + 1) - gammaln(n - counts + 1)
                   + betaln(counts + a, n - counts + b) - betaln(a, b))


def unbiased(c, n, k):
    """pass^k estimate with n trials, vectorized over c."""
    return np.array([comb(int(x), k) / comb(n, k) for x in c.ravel()]).reshape(c.shape)


sigma = {}
for d in D:
    runs = []
    for g_ in sorted(LOGS.glob("r*/gate/*/outcomes.json")):
        if d in g_.parent.name:
            runs += [np.array([sum(v) for _, v in sorted(arm["rewards"].items())]) for arm in read(g_).values()]
    counts = np.concatenate(runs)
    a, b = np.exp(minimize(betabinom_nll, np.log([2.0, 1.0]), args=(counts, 8)).x)
    for n in (4, 8, 12):
        sd1, sd4, sds, cor = [], [], [], []
        for c in runs:
            q = rng.beta(a + c, b + 8 - c, size=(400, len(c)))
            x, y = rng.binomial(n, q), rng.binomial(n, q)
            d1 = (x / n - y / n).mean(1) * 100
            d4 = (unbiased(x, n, 4) - unbiased(y, n, 4)).mean(1) * 100
            sd1.append(d1.std()), sd4.append(d4.std()), sds.append((0.8 * d1 + 0.2 * d4).std())
            cor.append(np.corrcoef(d1, d4)[0, 1])
        print(f"{d} n={n:2d}: sd of the difference in pass^1 {np.mean(sd1):.2f}, pass^4 {np.mean(sd4):.2f}, "
              f"correlation {np.mean(cor):.2f}, score S {np.mean(sds):.2f}")
        if n == 8:
            sigma[d] = np.mean(sds)
s = float(np.mean(list(sigma.values())))
z = rng.normal(size=(200000, 4)) * s / np.sqrt(2)
print(f"operating characteristics with sigma_S = {s:.2f} (true change in S: harness gate per attempt / with the retry / "
      f"model gate, best of three checkpoints against a shared control)")
for delta in (-1.5, 0.0, 0.5, 1.0, 2.0):
    one = norm.cdf(delta / s)
    retry = 1 - (1 - one) ** 2
    best3 = np.mean((delta + z[:, 1:] - z[:, :1]).max(1) > 0)
    print(f"  {delta:+.1f}: {100 * one:.1f}% / {100 * retry:.1f}% / {100 * best3:.1f}%")

section("Validation scores of the accepted versions (V scores table) and transfer to the official tasks")
v_p1 = {}
for d in D:
    seq = []
    for r in (1, 2):
        plugin = [read(q) for name in (f"plugin_{d}", f"plugin_{d}_retry")
                  for q in [LOGS / f"r{r}/gate/{name}/decision.json"] if q.exists()]
        model = read(LOGS / f"r{r}/gate/model_{d}/decision.json")
        if r == 1:
            seq.append({k: plugin[0][k]["current"] for k in ("score", "pass^1", "pass^4")})
        acc = next(x for x in plugin if x["accepted"])
        seq.append({k: acc[k]["candidate"] for k in ("score", "pass^1", "pass^4")})
        best = max((e for e, x in model["finalists"].items() if x["passes"]),
                   key=lambda e: model["finalists"][e]["score"]["candidate"])
        seq.append({k: model["finalists"][best][k]["candidate"] for k in ("score", "pass^1", "pass^4")})
        for x in plugin:
            print(f"{d} round {r} harness gate: current S {100 * x['score']['current']:.2f}, candidate "
                  f"{100 * x['score']['candidate']:.2f}, accepted {x['accepted']}")
    v_p1[d] = [100 * x["pass^1"] for x in seq]
    for v, x in zip(TRAJ, seq):
        print(f"{d} {v}: S {100 * x['score']:.2f}, V pass^1 / pass^4 {100 * x['pass^1']:.1f} / {100 * x['pass^4']:.1f}")
xs = [q for d in D for q in np.diff(v_p1[d])]
ys = [q for d in D for q in np.diff([pk(v, d, 1) for v in TRAJ])]
rest = sorted(zip(xs, ys))[:-1]  # without the update with the largest V gain
print(f"update gains in pass^1, V against official: r = {np.corrcoef(xs, ys)[0, 1]:.2f} "
      f"({np.corrcoef(*zip(*rest))[0, 1]:.2f} without the largest); "
      f"largest |difference| {max(abs(a - b) for a, b in zip(xs, ys)):.1f}")
for r in (1, 2):
    for d in D:
        col = read(LOGS / f"r{r}/collect/{d}.outcomes.json")
        print(f"round {r} {d}: the deployed system on the new batch, pass^1 / pass^4 "
              f"{100 * st.mean(p(v, 1) for v in col.values()):.1f} / {100 * st.mean(p(v, 4) for v in col.values()):.1f}; "
              f"tasks with a failed trial {sum(cls(v) != 'always' for v in col.values())}")

section("Selection under dialogue noise: same-seed pairing and regression to the mean")
raw, within, same_open, same, pairs_n = [], [], [], 0, 0
for g_ in sorted(LOGS.glob("r*/gate/*/outcomes.json")):
    arms = read(g_)
    if g_.parent.name.startswith("plugin"):
        comps = [(arms["current"], arms["candidate"], True)]
    else:
        control = Path(read(g_.parent / "decision.json")["current"])
        ctrl = read(ROOT / control.parent / "outcomes.json")[control.stem]
        comps = [(ctrl, arms[e], False) for e in arms]
    for x, y, harness in comps:
        ts = sorted(set(x["rewards"]) & set(y["rewards"]))
        u = np.array([w for t in ts for w in x["rewards"][t]], float)
        w = np.array([w for t in ts for w in y["rewards"][t]], float)
        raw.append(np.corrcoef(u, w)[0, 1])
        uc = np.concatenate([np.array(x["rewards"][t]) - np.mean(x["rewards"][t]) for t in ts])
        wc = np.concatenate([np.array(y["rewards"][t]) - np.mean(y["rewards"][t]) for t in ts])
        within.append((uc * wc).sum() / np.sqrt((uc ** 2).sum() * (wc ** 2).sum()))
        if harness:
            m_ = np.array([a_ == b_ for t in ts
                           for a_, b_ in zip(x["first_user_utterance"][t], y["first_user_utterance"][t])])
            same += m_.sum()
            pairs_n += len(m_)
            same_open.append((uc[m_] * wc[m_]).sum() / np.sqrt((uc[m_] ** 2).sum() * (wc[m_] ** 2).sum()))
print(f"{len(raw)} comparisons: raw correlation of paired outcomes {min(raw):.2f} to {max(raw):.2f}; within-task "
      f"{min(within):+.2f} to {max(within):+.2f}; harness gates: first user utterance identical in {same} of {pairs_n} "
      f"same-seed pairs ({100 * same / pairs_n:.0f}%), and the within-task correlation of these pairs is "
      f"{np.mean(same_open):+.2f} on average ({min(same_open):+.2f} to {max(same_open):+.2f})")
inst, ctrl_rate = [], []
flips = n_steps = 0
for att in sorted(LOGS.glob("r*/harness_rsi/*")):
    r, d = att.parts[-3], att.name.replace("_retry", "")
    col = read(LOGS / r / "collect" / f"{d}.outcomes.json")
    any_kept = False
    for ev in map(json.loads, (att / "events.jsonl").open()):
        if ev["event"] == "step" and ev["plugin"] == "v00" and not any_kept:
            cur = read(att / "steps" / f"s{ev['step']:02d}" / "outcomes.json")["current"]
            inst += [(sum(col[t]) / len(col[t]), sum(cur[t]) / len(cur[t])) for t in ev["failing"]]
            ctrl_rate += [sum(cur[t]) / len(cur[t]) for t in ev["regression"]]
        if ev["event"] == "retest":
            out = read(att / "steps" / f"s{ev['step']:02d}" / "outcomes.json")
            cand = out.get("pruned") or out["candidate"]
            naive = sum(sum(cand[t]) - 6 * sum(col[t]) / len(col[t]) for t in cand) >= 4
            flips += naive != ev["kept"]
            n_steps += 1
            any_kept |= ev["kept"]
gain = [b_ - a_ for a_, b_ in inst]
boot = sorted(np.mean(rng.choice(gain, len(gain))) for _ in range(10000))
print(f"failed tasks rerun before any edit: {100 * np.mean([a_ for a_, _ in inst]):.1f}% in the collection, "
      f"{100 * np.mean([b_ for _, b_ in inst]):.1f}% in the rerun ({len(inst)} task instances; difference "
      f"{100 * np.mean(gain):+.1f} [{100 * boot[250]:+.1f}, {100 * boot[9749]:+.1f}]); always-solved controls "
      f"100% -> {100 * np.mean(ctrl_rate):.1f}%")
print(f"a retest against the collection would reverse {flips} of the {n_steps} keep-or-revert decisions")
