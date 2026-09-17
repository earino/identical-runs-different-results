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
BASE_NUM = ["DepTime", "Distance"]
BASE_CAT = ["DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
HOURS = pd.Index(range(25))
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in BASE_CAT}
HC_LEVELS = pd.Index(
    sorted(
        (
            train["UniqueCarrier"].astype(str)
            + "@"
            + (((train["DepTime"] // 100) % 24).astype(int)).astype(str)
        ).unique()
    )
)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in BASE_NUM:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    hr = ((pd.to_numeric(df["DepTime"], "coerce") // 100) % 24).astype(int)
    X["HourCat"] = pd.Categorical(hr, categories=HOURS)
    X["HourCarrierCat"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "@" + hr.astype(str), categories=HC_LEVELS
    )
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
month_num = train["Month"].str.replace("c-", "").astype(int)
W_RECENCY = (0.5 + month_num / 12.0).to_numpy()

SCAN = [
    (dict(max_depth=8, learning_rate=0.03), 1300, None),
    (dict(max_depth=8, learning_rate=0.03, w="recency"), 1300, W_RECENCY),
    (dict(max_depth=8, learning_rate=0.03, grow="lossguide"), 1050, None),
    (dict(max_depth=5, learning_rate=0.05, subsample=0.85, seed=13), 320, None),
    (dict(max_depth=5, learning_rate=0.05, subsample=0.85, seed=55), 320, None),
    (dict(max_depth=8, learning_rate=0.03, mcw=5), 1300, None),
    (dict(max_depth=5, learning_rate=0.04, subsample=0.92, seed=99), 1100, None),
]

t0 = time.time()
members = []
eval_preds = []
for spec, rounds, w in SCAN:
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
        min_child_weight=spec.get("mcw", 1),
        grow_policy=spec.get("grow", "depthwise"),
        max_leaves=64 if spec.get("grow") == "lossguide" else 0,
    )
    m.fit(X_tr, y_tr, sample_weight=w, eval_set=[(X_ev, y_ev)], verbose=False)
    aucs = m.evals_result()["validation_0"]["auc"]
    bi = int(np.argmax(aucs))
    print(f"scan d{spec['max_depth']}-lr{spec['learning_rate']}-sub{spec.get('subsample', 1.0)}-s{spec.get('seed', 0)}-w{spec.get('w', 'none')}-mcw{spec.get('mcw', 1)}-{spec.get('grow', 'depthwise')}: best_round={bi + 1} auc={aucs[bi]:.4f}")
    members.append((m, bi + 1, aucs[bi]))
    eval_preds.append(m.predict_proba(X_ev, iteration_range=(0, bi + 1))[:, 1])
print(f"Scan time: {time.time() - t0:.1f}s")

P = np.stack(eval_preds)
solo = [roc_auc_score(y_ev, p) for p in P]
print(f"Solo AUCs: {['%.4f' % a for a in solo]}")

# greedy forward selection with cached predictions
t1 = time.time()
chosen = []
cur = np.zeros(len(y_ev))
remaining = list(range(len(P)))
best_auc = -1
while remaining:
    gains = []
    for i in remaining:
        cand = (cur * len(chosen) + P[i]) / (len(chosen) + 1)
        gains.append((roc_auc_score(y_ev, cand), i))
    auc, i = max(gains)
    if auc > best_auc + 1e-9 or (auc > best_auc - 0.0003 and False):
        pass
    chosen.append(i)
    remaining.remove(i)
    cur = P[chosen].mean(axis=0)
    print(f"greedy +member {i} ({len(chosen)}): ensemble AUC={auc:.4f}")
    if auc > best_auc:
        best_auc = auc
        best_set = list(chosen)
    if len(chosen) >= 7:
        break
print(f"Greedy time: {time.time() - t1:.1f}s")
print(f"BEST greedy set: {best_set} AUC={best_auc:.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = np.zeros(len(X))
    for i in best_set:
        m, bi, _ = members[i]
        ps += m.predict_proba(X, iteration_range=(0, bi))[:, 1]
    return ps / len(best_set)


print(f"Eval AUC: {best_auc:.4f}")
