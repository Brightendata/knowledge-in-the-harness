"""How far a split of our tasks is from the official base tasks: how well a classifier tells the two apart (AUC), on
what a task asks for (its knowledge-point tags, data/test_annotation.json for the official tasks) and its shape (writes,
kinds of write, a condition, information to state, pressure, script length); and which pairs of knowledge points the
official tasks combine that our tasks lack.

Usage, from the repository root, after the domain's generator:
    python -m rsi.data_preprocessing.alignment airline                   # V and the three training batches
    python -m rsi.data_preprocessing.alignment retail validation train   # train: the three batches together
Prints per split the AUC of a linear model (catches a tag or a shape whose share differs) and of a random forest (also
catches combinations: each tag at the same share on both sides, but not together), each beside the 95th percentile and
p of a null from shuffled labels; then the tags and tag pairs whose shares differ most (z of two proportions); then the
pairs of knowledge points found in two or more official stories that the candidates (almost) never have, or the split
has at under a third of the official share, and how many official tasks have a pair of the first kind (a pair the
generator cannot make: a missing kind of request, not a matter of choosing). Takes about a minute per split.

Cross-validation keeps each official story (variants with near-identical scripts) and each of our variant groups within
one fold: split across folds, the forest learns the stories and the AUC comes out too high. The AUC is a diagnostic, not
an acceptance test: common.select chooses V and the batches to match the official shares of these same tags and shapes,
so the held-out folds are not independent of that choice, and the matching alone drives the AUC below 0.5 (the shuffled
null leaves the choice out). An AUC above the null shows what the choice failed to match; one at or below 0.5 says
nothing about what the features leave out. The z of a pair ignores that official variants come in stories, so read the
pairs as which kinds of task to look at, not as targets."""
import json
import re
import sys
from itertools import combinations

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold

from rsi.data_preprocessing.common import ANNOTATION, ROOT, WRITES, outputs, pairs

COND = re.compile(r"\bif\b|\bunless\b|\botherwise\b|\bin case\b|\bonly if\b", re.I)
PRESS = re.compile(r"insist|push back|do not take|don't take|persist|demand|keep asking|ask again", re.I)
STORY = 0.4  # TF-IDF cosine above which two official scripts are variants of one story
SHUFFLES = 30
MODELS = {"linear": lambda: LogisticRegression(max_iter=2000),
          "forest": lambda: RandomForestClassifier(200, min_samples_leaf=3, n_jobs=-1, random_state=0)}


def script(task):
    ins = task["user_scenario"]["instructions"]
    return f"{ins.get('reason_for_call') or ''} {ins.get('task_instructions') or ''}"


def shape(task, tags, domain):
    """Writes (up to 4), kinds of write, a condition, information to state, pressure, log of the script's words."""
    text = script(task)
    writes = [a["name"] for a in task["evaluation_criteria"].get("actions") or [] if a["name"] in WRITES[domain]]
    return [min(len(writes), 4), len(set(writes)), bool(COND.search(text)), any(t.startswith("info_") for t in tags),
            bool(PRESS.search(text)), np.log1p(len(text.split()))]


def stories(tasks):
    """A story number per official task: scripts with TF-IDF cosine above STORY are linked, linked ones are a story."""
    sim = cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform([script(t) for t in tasks]))
    return connected_components(sim > STORY, directed=False)[1]


def fold_auc(model, X, y, groups, seed):
    """AUC of out-of-fold predictions, 5 folds that keep each group within one."""
    p = np.zeros(len(y))
    for train, test in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        p[test] = MODELS[model]().fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    return roc_auc_score(y, p)


def gaps(ours, official, top=8):
    """The tags and tag pairs whose shares differ most: (z, name, share of ours, share of the official tasks)."""
    tags = sorted({t for s in official for t in s})
    out = {}
    for name, keys in (("tags", [(t,) for t in tags]), ("pairs", list(combinations(tags, 2)))):
        rows = []
        for key in keys:
            a, b = (sum(all(t in s for t in key) for s in x) for x in (ours, official))
            if a + b < 4:
                continue
            p = (a + b) / (len(ours) + len(official))
            z = (b / len(official) - a / len(ours)) / np.sqrt(p * (1 - p) * (1 / len(ours) + 1 / len(official)))
            rows.append((z, " & ".join(key), a / len(ours), b / len(official)))
        out[name] = sorted(rows, key=lambda r: -abs(r[0]))[:top]
    return out


def compare(domain, split):
    """Prints how far the split (or "train", the three batches) is from the official tasks."""
    annotation = json.loads(ANNOTATION.read_text())[domain]
    official = {t["id"]: t for t in json.loads((ROOT / f"tau2-bench/data/tau2/domains/{domain}/tasks.json").read_text())}
    T = [(official[k], set(v)) for k, v in annotation.items()]
    _, final, splits_path, _ = outputs(domain)
    records = {r["task"]["id"]: r for r in map(json.loads, open(final))}
    splits = json.loads(splits_path.read_text())
    ids = [i for k in ("round1", "round2", "round3") for i in splits[k]] if split == "train" else splits[split]
    G = [(records[i]["task"], set(records[i]["tags"])) for i in ids]
    tagset = sorted({t for _, s in T for t in s})
    X = np.array([shape(t, s, domain) + [t_ in s for t_ in tagset] for t, s in T + G], float)
    y = np.array([1] * len(T) + [0] * len(G))
    story = stories([t for t, _ in T])
    ours = {g: k for k, g in enumerate(sorted({records[i]["group"] for i in ids}))}
    groups = np.concatenate([story, story.max() + 1 + np.array([ours[records[i]["group"]] for i in ids])])
    print(f"{domain} {split}: {len(G)} tasks in {len(ours)} groups, against {len(T)} official tasks in "
          f"{story.max() + 1} stories")
    rng = np.random.default_rng(0)
    every, n_official = np.unique(groups), len(np.unique(story))
    for model in MODELS:
        auc = np.mean([fold_auc(model, X, y, groups, seed) for seed in range(3)])
        null = []
        for seed in range(SHUFFLES):  # which groups count as official, shuffled
            picked = set(rng.choice(every, n_official, replace=False))
            null.append(fold_auc(model, X, np.array([g in picked for g in groups], int), groups, seed))
        print(f"  {model:6s} AUC {auc:.2f}   shuffled labels: 95th percentile {np.percentile(null, 95):.2f}, "
              f"p {np.mean(np.array(null) >= auc):.2f}")
    for name, rows in gaps([s for _, s in G], [s for _, s in T]).items():
        print(f"  {name} differing most (ours / official, %): " + "; ".join(
            f"{key} {100 * a:.0f} / {100 * b:.0f} (z {z:+.1f})" for z, key, a, b in rows))
    keys = list(annotation)
    seen = {}  # pair -> official stories and tasks it is in
    for idx, k in enumerate(keys):
        for p in pairs(set(annotation[k])):
            seen.setdefault(p, (set(), []))
            seen[p][0].add(story[idx])
            seen[p][1].append(k)
    candidates = [pairs(set(r["tags"])) for r in map(json.loads, open(outputs(domain)[0]))]
    ours = [pairs(s) for _, s in G]
    lacking, thin = [], []
    for p, (where, hit) in sorted(seen.items()):
        if len(where) < 2:
            continue
        made = sum(p in c for c in candidates)
        share, here = len(hit) / len(keys), sum(p in c for c in ours) / len(ours)
        if made < 10:
            lacking.append((p[5:], made, hit))
        elif here < share / 3:
            thin.append((p[5:], round(100 * here, 1), round(100 * share, 1)))
    touched = {k for _, _, hit in lacking for k in hit}
    print(f"  official tasks with a pair of 2+ stories under 10 candidates: {len(touched)}/{len(keys)}; pairs: "
          + "; ".join(f"{p} ({made}: {', '.join(hit)})" for p, made, hit in lacking))
    print("  pairs of 2+ stories under a third of their official share (ours / official, %): "
          + "; ".join(f"{p} {a} / {b}" for p, a, b in thin))


if __name__ == "__main__":
    domain, *names = sys.argv[1:]
    for split in names or ("validation", "round1", "round2", "round3"):
        compare(domain, split)
