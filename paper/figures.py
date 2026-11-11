"""The paper's figures from the released records: results/ (official tasks) and the gate decisions in artifacts/logs/.

    python paper/figures.py

Writes trajectory.pdf, ablation.pdf, gains.pdf and transfer.pdf to paper/figures/.
"""
import gzip
import json
from collections import Counter
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
BLUE, ORANGE, GRAY, TEAL = "#1f4e79", "#b35806", "#8c8c8c", "#1f8a70"
NO_DATE = {"CreationDate": None}  # byte-identical PDFs on every run
DOMAINS = ["retail", "airline"]
TRAJ = ["H0-M0", "H1-M0", "H1-M1", "H2-M1", "H2-M2"]
FRONTIER = ["H0-gpt-5.6-sol", "H0-claude-sonnet-5", "H0-glm-5.2"]


def successes(version, domain):
    """Successes and trials of each task."""
    c, n = Counter(), Counter()
    with gzip.open(ROOT / f"results/{version}/{domain}.json.gz", "rt") as f:
        for s in json.load(f)["simulations"]:
            n[s["task_id"]] += 1
            c[s["task_id"]] += (s.get("reward_info") or {}).get("reward", 0) == 1
    return c, n


def passk(c, n):
    return {k: 100 * np.mean([comb(c[t], k) / comb(n[t], k) for t in n]) for k in (1, 4)}


def tint(color, white=0.6):
    """The colour mixed with white."""
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (255 - x) * white):02x}" for x in rgb)


def v_pass1(domain):
    """V pass^1 of each trajectory version, from the gate decisions; a version gated twice keeps its first score."""
    out = []
    for r in (1, 2):
        plugin = [json.loads(p.read_text()) for name in (f"plugin_{domain}", f"plugin_{domain}_retry")
                  for p in [ROOT / f"artifacts/logs/r{r}/gate/{name}/decision.json"] if p.exists()]
        model = json.loads((ROOT / f"artifacts/logs/r{r}/gate/model_{domain}/decision.json").read_text())
        if r == 1:
            out.append(plugin[0]["pass^1"]["current"])
        out.append(next(d for d in plugin if d["accepted"])["pass^1"]["candidate"])
        best = max((e for e, x in model["finalists"].items() if x["passes"]),
                   key=lambda e: model["finalists"][e]["score"]["candidate"])
        out.append(model["finalists"][best]["pass^1"]["candidate"])
    return [100 * x for x in out]


S = {(v, d): successes(v, d) for d in DOMAINS for v in TRAJ + ["H2-M0", "H0-M2", "official_setting"] + FRONTIER}
P = {key: passk(*s) for key, s in S.items()}
OUT.mkdir(parents=True, exist_ok=True)

# Trajectory on the official tasks: rows pass^1 and pass^4, columns the domains; the reference agent as a dashed line
# and the range of the three frontier models as a band.
fig, axes = plt.subplots(2, 2, figsize=(3.0, 3.0), sharex=True)
x = np.arange(len(TRAJ))
YLIM = {1: (50, 92), 4: (30, 78)}
for col, d in enumerate(DOMAINS):
    for row, k in enumerate((1, 4)):
        ax = axes[row, col]
        lo, hi = min(P[f, d][k] for f in FRONTIER), max(P[f, d][k] for f in FRONTIER)
        ax.axhspan(lo, hi, facecolor="#dddddd", lw=0)
        ax.axhline(P["official_setting", d][k], color=ORANGE, ls="--", lw=0.9)
        ax.plot(x, [P[v, d][k] for v in TRAJ], "-", color=BLUE, marker="o", ms=3, lw=1.1)
        ax.set_ylim(*YLIM[k])
        if row == 0:
            ax.set_title(d.capitalize(), pad=3)
        if col == 0:
            ax.set_ylabel(f"pass$^{k}$")
        ax.set_xticks(x, TRAJ, rotation=45, ha="right", rotation_mode="anchor")
ref, band = P["official_setting", "retail"][1], max(P[f, "retail"][1] for f in FRONTIER)
axes[0, 0].text(0.0, ref - 4.6, "reference", fontsize=7.5, color=ORANGE)
axes[0, 0].text(0.0, band + 1.0, "frontier", fontsize=7.5, color="#444444")
fig.tight_layout(h_pad=0.5, w_pad=0.6)
fig.savefig(OUT / "trajectory.pdf", metadata=NO_DATE)

# The 2x2 ablation at full text width, for the evolved rules alone, the trained weights alone, and both.
# (a) Gains over H0-M0: pass^1 in the light tone, pass^4 in the dark tone.
# (b) Success on the tasks H0-M0 solved in 0, 1-3, or 4 of its four trials.
LAYERS = [("rules only", "H2-M0", ORANGE), ("weights only", "H0-M2", BLUE), ("both", "H2-M2", TEAL)]
VALUE = dict(ha="center", va="bottom", fontsize=6, color="#333333")
fig = plt.figure(figsize=(6.3, 1.82))
gs = fig.add_gridspec(1, 5, width_ratios=[1.12, 1.12, 0.40, 1.27, 1.27], wspace=0.1,
                      left=0.062, right=0.998, bottom=0.235, top=0.845)
ga = [fig.add_subplot(gs[0])]
ga.append(fig.add_subplot(gs[1], sharey=ga[0]))
gb = [fig.add_subplot(gs[3])]
gb.append(fig.add_subplot(gs[4], sharey=gb[0]))
for ax, d in zip(ga, DOMAINS):
    for i, (_, v, color) in enumerate(LAYERS):
        for off, k, tone in ((-0.21, 1, tint(color)), (0.21, 4, color)):
            g = P[v, d][k] - P["H0-M0", d][k]
            ax.bar(i + off, g, width=0.34, color=tone)
            ax.text(i + off, g + 0.4, f"{g:.1f}", **VALUE)
    ax.set_xticks(range(3), ["rules\nonly", "weights\nonly", "both"], fontsize=7)
    ax.set_ylim(0, 26)
    ax.set_yticks(range(0, 21, 10))
ga[0].set_ylabel("Gain over H0-M0 (points)")
ga[0].legend(handles=[Patch(color="#c4c4c4", label="pass$^1$"), Patch(color="#555555", label="pass$^4$")],
             loc="upper left", handlelength=1.0, handletextpad=0.4, borderaxespad=0.1, labelspacing=0.25)
for ax, d in zip(gb, DOMAINS):
    c0, n0 = S["H0-M0", d]
    bins = [[t for t in n0 if c0[t] == 0], [t for t in n0 if 0 < c0[t] < n0[t]], [t for t in n0 if c0[t] == n0[t]]]
    for i, (_, v, color) in enumerate(LAYERS):
        c, n = S[v, d]
        for b, tasks in enumerate(bins):
            y = 100 * np.mean([c[t] / n[t] for t in tasks])
            ax.bar(b + (i - 1) * 0.29, y, width=0.23, color=color)
            ax.text(b + (i - 1) * 0.29, y + 1.5, f"{y:.0f}", **VALUE)
    ax.set_xticks(range(3), ["0/4", "1–3/4", "4/4"])
    ax.set_ylim(0, 132)
    ax.set_yticks(range(0, 101, 25))
gb[0].set_ylabel("Success (%)")
gb[0].legend(handles=[Patch(color=color, label=name) for name, _, color in LAYERS], fontsize=7,
             loc="upper left", handlelength=1.0, handletextpad=0.4, borderaxespad=0.1, labelspacing=0.25)
fig.text((gb[0].get_position().x0 + gb[1].get_position().x1) / 2, 0.025, "Tasks by H0-M0 successes in four trials",
         ha="center", va="bottom", fontsize=8)
for ax, d in zip(ga + gb, DOMAINS * 2):
    ax.set_title(d.capitalize(), pad=3)
    ax.set_xlim(-0.6, 2.6)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
for ax in (ga[1], gb[1]):
    ax.tick_params(labelleft=False)
for ax, tag in ((ga[0], "(a)"), (gb[0], "(b)")):
    ax.text(-0.3, 1.1, tag, transform=ax.transAxes, fontsize=8.5, fontweight="bold", ha="left", va="bottom")
fig.savefig(OUT / "ablation.pdf", metadata=NO_DATE)

# pass^1 gain against pass^4 gain of each layer contrast: the accepted updates and the ablation contrasts.
CONTRASTS = {
    "weights": [("H1-M0", "H1-M1"), ("H2-M1", "H2-M2"), ("H0-M0", "H0-M2"), ("H2-M0", "H2-M2")],
    "rules, base weights": [("H0-M0", "H1-M0"), ("H0-M0", "H2-M0")],
    "rules, trained weights": [("H1-M1", "H2-M1"), ("H0-M2", "H2-M2")],
}
STYLE = {"weights": dict(marker="o", color=BLUE, s=20),
         "rules, base weights": dict(marker="s", color=ORANGE, s=20),
         "rules, trained weights": dict(marker="D", facecolors="none", edgecolors=ORANGE, s=24)}
fig, ax = plt.subplots(figsize=(3.0, 3.0))
ax.plot([-1, 20], [-1, 20], ls=":", color=GRAY, lw=0.8)
for name, pairs in CONTRASTS.items():
    pts = [(P[b, d][1] - P[a, d][1], P[b, d][4] - P[a, d][4]) for a, b in pairs for d in DOMAINS]
    ax.scatter(*zip(*pts), lw=0.9, label=name, **STYLE[name])
ax.set_xlim(-1, 20)
ax.set_ylim(-1, 20)
ax.set_aspect("equal")
ax.set_xticks(range(0, 21, 5))
ax.set_yticks(range(0, 21, 5))
ax.set_xlabel("Gain in pass$^1$ (points)")
ax.set_ylabel("Gain in pass$^4$ (points)")
ax.legend(loc="upper left", handletextpad=0.3)
fig.tight_layout()
fig.savefig(OUT / "gains.pdf", metadata=NO_DATE)

# Stage gains in pass^1 on V against the official tasks.
fig, ax = plt.subplots(figsize=(3.0, 2.4))
xs, ys = [], []
for d, marker in zip(DOMAINS, ("o", "^")):
    v = np.diff(v_pass1(d))
    t = np.diff([P[x, d][1] for x in TRAJ])
    ax.scatter(v, t, marker=marker, s=18, color=BLUE, label=d.capitalize())
    xs += list(v)
    ys += list(t)
r = np.corrcoef(xs, ys)[0, 1]
ax.plot([-1, 14], [-1, 14], ls=":", color=GRAY, lw=0.8)
ax.text(0.97, 0.05, f"$r = {r:.2f}$", transform=ax.transAxes, ha="right")
ax.set_xlim(-0.5, 14)
ax.set_ylim(-0.5, 14)
ax.set_xticks(range(0, 15, 2))
ax.set_yticks(range(0, 15, 2))
ax.set_xlabel("Gain in pass$^1$ on V (points)")
ax.set_ylabel("Gain in pass$^1$ on official tasks (points)")
ax.legend(loc="upper left", handletextpad=0.2)
fig.tight_layout()
fig.savefig(OUT / "transfer.pdf", metadata=NO_DATE)
print(f"r = {r:.3f}")
