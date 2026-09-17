"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Findings baked in:
  - Month/DayofMonth dropped (2005 seasonality does not transfer); DayOfWeek kept.
  - One-hot for cats + smoothed quarter-hour-of-day TE + distance features (fit on train only).
  - Deep trees (24) are essential with one-hot; seed-bagged ensemble for variance reduction.
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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features (all statistics fit on train only) --------------------------------
CAT_COLS = ["DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
_y = (train[TARGET] == POSITIVE).astype(int)

_qh = (train["DepTime"] // 100) * 4 + (train["DepTime"] % 100) // 15
_g = pd.DataFrame({"k": _qh, "y": _y}).groupby("k")["y"].agg(["sum", "count"])
TE_QH = ((_g["sum"] + 100 * 0.5) / (_g["count"] + 100)).to_dict()

ORG_CNT = train["Origin"].value_counts().to_dict()
_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
ROUTE_DMEAN = train.groupby(_route)["Distance"].mean().to_dict()
DIST_MEAN = float(train["Distance"].mean())


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    dep = df["DepTime"]
    r = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    parts = [
        pd.DataFrame({
            "DepTime": dep.astype(float),
            "Distance": df["Distance"].astype(float),
            "te_qh": ((dep // 100) * 4 + (dep % 100) // 15).map(TE_QH).fillna(0.5).to_numpy(),
            "log_dist": np.log1p(df["Distance"].astype(float)),
            "org_cnt": np.log1p(df["Origin"].map(ORG_CNT).fillna(0).astype(float)),
            "dist_vs_route": df["Distance"] / r.map(ROUTE_DMEAN).fillna(DIST_MEAN),
        })
    ]
    for c in CAT_COLS:
        parts.append(pd.get_dummies(pd.Categorical(df[c], categories=levels[c]), prefix=c))
    return pd.concat(parts, axis=1)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: seed-bagged deep ensemble -------------------------------------------
BASE = dict(
    n_estimators=200,
    max_depth=24,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.6,
    tree_method="hist",
)
N_MODELS = 3

t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
models = []
for s in range(N_MODELS):
    m = xgb.XGBClassifier(random_state=s, n_jobs=N_JOBS, **BASE)
    m.fit(Xtr, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.zeros(len(X))
    for m in models:
        p += m.predict_proba(X)[:, 1]
    return p / len(models)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
