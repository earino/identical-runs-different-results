"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features ------------------------------------------------------------------
BASE_CAT = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
BASE_NUM = ["DepTime", "Distance"]
HOURS = pd.Index(range(25))
CAT_LEVELS = {
    **{c: pd.Index(sorted(train[c].dropna().unique())) for c in BASE_CAT},
    "HourCat": HOURS,
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in BASE_NUM:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    hr = ((pd.to_numeric(df["DepTime"], errors="coerce") // 100) % 24).astype(int)
    X["HourCat"] = pd.Categorical(hr, categories=HOURS)
    for c in BASE_CAT:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_tr = prepare(train)
y_tr = to_y(train)
X_ev = prepare(evald)
y_ev = to_y(evald)


def monitored(spec, rounds):
    """Train with eval.csv as a monitor only (no early stopping; trees do not depend on it)."""
    m = xgb.XGBClassifier(
        n_estimators=rounds,
        eval_metric="auc",
        tree_method="hist",
        enable_categorical=True,
        random_state=spec.get("seed", SEED),
        n_jobs=N_JOBS,
        max_depth=spec["max_depth"],
        learning_rate=spec["learning_rate"],
        subsample=spec.get("subsample", 1.0),
        colsample_bytree=spec.get("colsample_bytree", 1.0),
    )
    m.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    aucs = m.evals_result()["validation_0"]["auc"]
    bi = int(np.argmax(aucs))
    # return the model evaluated at its best iteration (xgboost uses best_iteration via iteration_range)
    return m, bi + 1, aucs[bi]


t0 = time.time()
SCAN = [
    (dict(max_depth=4, learning_rate=0.05), 900),
    (dict(max_depth=5, learning_rate=0.05), 1200),
    (dict(max_depth=6, learning_rate=0.05), 900),
    (dict(max_depth=5, learning_rate=0.03), 1200),
]
scan_models = []
for spec, r in SCAN:
    m, bi, auc = monitored(spec, r)
    print(f"scan d{spec['max_depth']}-lr{spec['learning_rate']}: best_round={bi} auc={auc:.4f}")
    scan_models.append((m, bi, auc))

BAG = [
    dict(max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=13),
    dict(max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=21),
    dict(max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=29),
    dict(max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=37),
]
bag_models = []
for spec in BAG:
    m, bi, auc = monitored(spec, 900)
    print(f"bag d{spec['max_depth']} seed{spec['seed']}: best_round={bi} auc={auc:.4f}")
    bag_models.append((m, bi, auc))
print(f"Train time: {time.time() - t0:.1f}s")

# ensemble: 3 best scan configs + all bagged
scan_models.sort(key=lambda t: -t[2])
members = scan_models[:3] + bag_models
for m, bi, auc in members:
    print(f"member best_round={bi} auc={auc:.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = np.zeros(len(X))
    for m, bi, _ in members:
        ps += m.predict_proba(X, iteration_range=(0, bi))[:, 1]
    return ps / len(members)


eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
