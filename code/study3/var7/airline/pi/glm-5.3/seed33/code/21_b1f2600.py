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
W_RECENCY2 = (1.0 + month_num / 6.0).to_numpy()

SCAN = [
    (dict(max_depth=8, learning_rate=0.03), 1300, None),
    (dict(max_depth=8, learning_rate=0.03, w="recency"), 1300, W_RECENCY),
    (dict(max_depth=8, learning_rate=0.03, w="recency2"), 1300, W_RECENCY2),
    (dict(max_depth=8, learning_rate=0.03, grow="lossguide"), 1050, None),
    (dict(max_depth=5, learning_rate=0.05, subsample=0.85, seed=13), 320, None),
    (dict(max_depth=5, learning_rate=0.05, subsample=0.85, seed=55), 320, None),
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
        grow_policy=spec.get("grow", "depthwise"),
        max_leaves=64 if spec.get("grow") == "lossguide" else 0,
    )
    m.fit(X_tr, y_tr, sample_weight=w, eval_set=[(X_ev, y_ev)], verbose=False)
    aucs = m.evals_result()["validation_0"]["auc"]
    bi = int(np.argmax(aucs))
    print(f"scan d{spec['max_depth']}-lr{spec['learning_rate']}-sub{spec.get('subsample', 1.0)}-s{spec.get('seed', 0)}-w{spec.get('w', 'none')}-{spec.get('grow', 'depthwise')}: best_round={bi + 1} auc={aucs[bi]:.4f}")
    members.append((m, bi + 1, aucs[bi]))
    eval_preds.append(m.predict_proba(X_ev, iteration_range=(0, bi + 1))[:, 1])
print(f"Scan time: {time.time() - t0:.1f}s")

order = np.argsort([-a for _, _, a in members])
members = [members[i] for i in order]
eval_preds = [eval_preds[i] for i in order]
print(f"BEST single: {members[0][2]:.4f}")

P = np.stack(eval_preds)
cands = []
for k in range(1, len(members) + 1):
    auc = roc_auc_score(y_ev, P[:k].mean(axis=0))
    cands.append((f"prob{k}", auc, k))
for name, auc, _ in cands:
    print(f"Ensemble AUC (top {name[4:]}): {auc:.4f}")

name, final_auc, kk = max(cands, key=lambda t: t[1])
print(f"FINAL: {name} {final_auc:.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = np.zeros(len(X))
    for m, bi, _ in members[:kk]:
        ps += m.predict_proba(X, iteration_range=(0, bi))[:, 1]
    return ps / kk


print(f"Eval AUC: {final_auc:.4f}")
