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

MEMBERS = [
    dict(n_estimators=1243, max_depth=8, learning_rate=0.03, subsample=1.0, seed=SEED, w=None),
    dict(n_estimators=1278, max_depth=8, learning_rate=0.03, subsample=1.0, seed=SEED, w="recency"),
    dict(n_estimators=273, max_depth=5, learning_rate=0.05, subsample=0.85, seed=13, w="recency"),
    dict(n_estimators=272, max_depth=5, learning_rate=0.05, subsample=0.85, seed=77, w="recency"),
]

t0 = time.time()
models = []
for spec in MEMBERS:
    m = xgb.XGBClassifier(
        n_estimators=spec["n_estimators"],
        tree_method="hist",
        enable_categorical=True,
        random_state=spec["seed"],
        n_jobs=N_JOBS,
        max_depth=spec["max_depth"],
        learning_rate=spec["learning_rate"],
        subsample=spec["subsample"],
    )
    m.fit(X_tr, y_tr, sample_weight=W_RECENCY if spec["w"] else None)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = np.zeros(len(X))
    for m in models:
        ps += m.predict_proba(X)[:, 1]
    return ps / len(models)


t0 = time.time()
eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
