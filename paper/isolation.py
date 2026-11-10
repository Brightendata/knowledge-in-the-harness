"""How far the generated tasks stand from the official ones (Appendix A): shared customers, identical reference writes,
TF-IDF similarity of the scenarios, and classifier two-sample tests on knowledge-point and structure features.

    python paper/isolation.py

Needs the tau2-bench clone at the repository root (README, Setup) for the official tasks and databases. The two-sample
tests run 5 repeats of 5-fold cross-validation that keeps each official story and each generated variant group within
one fold, and take a few minutes.
"""
import json
import re
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tau2.data_model.tasks import Task

from rsi.data_preprocessing.alignment import script, shape, stories
from rsi.data_preprocessing.common import outcome

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "LR": make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced", max_iter=2000)),
    "SVM": make_pipeline(StandardScaler(), SVC(C=1, kernel="rbf", gamma="scale", class_weight="balanced")),
    "RF": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced", random_state=0),
    "HGB": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, min_samples_leaf=10, class_weight="balanced",
                                          early_stopping=False, random_state=0),
}
annotation = json.loads((ROOT / "data/test_annotation.json").read_text())

for domain in ("retail", "airline"):
    tau2 = ROOT / f"tau2-bench/data/tau2/domains/{domain}"
    official = {t["id"]: t for t in json.loads((tau2 / "tasks.json").read_text())}
    base = [official[k] for k in annotation[domain]]
    records = {r["task"]["id"]: r for r in map(json.loads, open(ROOT / f"data/processed/{domain}_final.jsonl"))}
    splits = json.loads((ROOT / f"data/splits/{domain}.json").read_text())
    ids = splits["validation"] + [t for s in ("round1", "round2", "round3") for t in splits[s]]

    # customers named in the official tasks, by user id, order number, e-mail, or name with zip code
    db, text = json.loads((tau2 / "db.json").read_text()), json.dumps(base)
    if domain == "airline":
        customers = {u for u in db["users"] if u in text}
    else:
        customers = {db["orders"]["#" + o]["user_id"] for o in re.findall(r"W\d{7}", text) if "#" + o in db["orders"]}
        for s in map(json.dumps, base):
            customers |= {k for k, u in db["users"].items() if k in s or u["email"] in s or (
                f'{u["name"]["first_name"]} {u["name"]["last_name"]}' in s and u["address"]["zip"] in s)}
    shared = sum(records[i]["customer"] in customers for i in ids)
    val = {records[i]["customer"] for i in splits["validation"]}
    val_train = sum(records[i]["customer"] in val for i in ids[len(splits["validation"]):])
    # tasks that write nothing share the empty outcome and are left out
    taken = {outcome(Task.model_validate(t), domain) for t in base}
    mine = [outcome(Task.model_validate(records[i]["task"]), domain) for i in ids]
    identical = sum(o in taken for o in mine if o)

    tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words="english").fit_transform(
        [script(t) for t in base] + [script(records[i]["task"]) for i in ids])
    nearest = cosine_similarity(tfidf[len(base):], tfidf[:len(base)]).max(1)
    among = cosine_similarity(tfidf[:len(base)])
    np.fill_diagonal(among, 0)
    print(f"{domain}: {len(ids)} generated tasks; customers shared with official tasks {shared}, validation customers in "
          f"training batches {val_train}, writing tasks whose reference writes equal an official task's {identical}; "
          f"scenario TF-IDF cosine "
          f"to the nearest official scenario mean {nearest.mean():.2f}, max {nearest.max():.2f}; official to nearest "
          f"other official mean {among.max(1).mean():.2f}")

    tags = sorted({t for k in annotation[domain] for t in annotation[domain][k]})
    labeled = [(t, set(annotation[domain][t["id"]])) for t in base] + [(records[i]["task"], set(records[i]["tags"])) for i in ids]
    group = {g: n for n, g in enumerate(sorted({records[i]["group"] for i in ids}))}
    story = stories(base)
    groups = np.concatenate([story, story.max() + 1 + np.array([group[records[i]["group"]] for i in ids])])
    y = np.array([1] * len(base) + [0] * len(ids))
    X = np.column_stack([np.array([[g in ts for g in tags] for _, ts in labeled], float),
                         np.array([shape(t, ts, domain) for t, ts in labeled], float)])
    for name, estimator in MODELS.items():
        aucs = []
        for seed in range(5):
            scores = []
            for fit, held in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
                model = clone(estimator).fit(X[fit], y[fit])
                p = model.decision_function(X[held]) if hasattr(model, "decision_function") else model.predict_proba(X[held])[:, 1]
                scores.append(roc_auc_score(y[held], p))
            aucs.append(np.mean(scores))
        print(f"   two-sample test, {name}: AUC {np.mean(aucs):.2f}")
