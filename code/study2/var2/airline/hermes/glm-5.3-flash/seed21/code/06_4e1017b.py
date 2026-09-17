"""XGBoost binary classifier for the airline delay task. THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Small diverse ensemble of 3 XGBoost models:
  A) full features, depth 4, seed 42
  B) full features, depth 6, colsample 0.7, seed 7
  C) minimal view (DepTime + Distance only), depth 4, seed 123  (decorrelated member)
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

# --- encoders (fitted on training data only) -----------------------------------
CAT_LEVELS = {
    "Month": [f"c-{i}" for i in range(1, 13)],
    "DayofMonth": [f"c-{i}" for i in range(1, 32)],
    "DayOfWeek": [f"c-{i}" for i in range(1, 8)],
}
for c in ("UniqueCarrier", "Origin", "Dest"):
    CAT_LEVELS[c] = pd.Index(sorted(train[c].dropna().unique()))
# carrier x half-hour-of-day interaction levels, from training data only
_dt_tr = pd.to_numeric(train["DepTime"])
_chh_tr = (_dt_tr // 100 * 2 + _dt_tr % 100 // 30).clip(0, 53).astype(int)
CHH_LEVELS = pd.Index((train["UniqueCarrier"].astype(str) + "_" + _chh_tr.astype(str)).unique())


def prepare(df: pd.DataFrame, kind: str = "full") -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = dt // 100
    minute = dt % 100

    # scheduled time of day
    X["dt_hour"] = hour
    X["dt_min"] = minute
    X["dt_sin"] = np.sin(2 * np.pi * (hour + minute / 60.0) / 24.0)
    X["dt_cos"] = np.cos(2 * np.pi * (hour + minute / 60.0) / 24.0)

    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["log_distance"] = np.log1p(X["Distance"])

    if kind == "full":
        X["dt_slot"] = pd.Categorical(
            hour.clip(0, 26).astype(int).astype(str).radd("h"),
            categories=[f"h{i}" for i in range(27)],
        )
        for c, levels in CAT_LEVELS.items():
            X[c] = pd.Categorical(df[c], categories=levels)
        chh = (dt // 100 * 2 + dt % 100 // 30).clip(0, 53).astype(int)
        X["carrier_halfhour"] = pd.Categorical(
            df["UniqueCarrier"].astype(str) + "_" + chh.astype(str), categories=CHH_LEVELS)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- ensemble -------------------------------------------------------------------
FULL = dict(n_estimators=1600, max_depth=4, learning_rate=0.04, tree_method="hist",
            enable_categorical=True, subsample=0.8, colsample_bytree=0.5,
            min_child_weight=10, reg_lambda=5.0, reg_alpha=1.0, random_state=SEED, n_jobs=N_JOBS)
MEMBERS = [
    (dict(FULL), "full"),
    (dict(FULL, max_depth=6, colsample_bytree=0.7, random_state=7), "full"),
    (dict(FULL, random_state=123, min_child_weight=20), "minimal"),
]

ytr = to_y(train)
t0 = time.time()
models = []
for params, kind in MEMBERS:
    m = xgb.XGBClassifier(**params)
    m.fit(prepare(train, kind), ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = [m.predict_proba(prepare(df, kind))[:, 1] for m, (_, kind) in zip(models, MEMBERS)]
    return np.mean(P, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
