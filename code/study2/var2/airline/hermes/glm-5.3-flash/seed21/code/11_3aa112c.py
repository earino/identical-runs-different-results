"""XGBoost binary classifier for the airline delay task. THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Ensemble of 14 XGBoost models over 4 feature views, equal-weight average:
  8 tuned members (d4-d6, subsample/colsample/min_child_weight/lambda/alpha tuned, seed bags)
    over views full (carrier x half-hour), full_qh (carrier x quarter-hour), geo (airports),
    minimal (DepTime + Distance only); all but geo use month-recency sample weights.
  6 "bare" members (n800 lr.08, xgboost defaults otherwise) over full/full_qh/minimal,
    which add inductive-bias diversity for another ~+0.002.
All category levels (including interaction levels) are fixed from training data only, so
unseen holdout levels map to NaN and native categorical handling stays valid.
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
_dt_tr = pd.to_numeric(train["DepTime"])
_hh_tr = (_dt_tr // 100 * 2 + _dt_tr % 100 // 30).clip(0, 53).astype(int)
CHH_LEVELS = pd.Index((train["UniqueCarrier"].astype(str) + "_" + _hh_tr.astype(int).astype(str)).unique())
_qh_tr = (_dt_tr // 100 * 4 + _dt_tr % 100 // 15).clip(0, 107).astype(int)
CQH_LEVELS = pd.Index((train["UniqueCarrier"].astype(str) + "_" + _qh_tr.astype(int).astype(str)).unique())
SLOT_LEVELS = [f"h{i}" for i in range(27)]
# month-recency weights (fixed formula, no fitting): Dec 2005 weighs 2.65x vs Jan
MONTH_W = train["Month"].str[2:].astype(int).map(lambda m: 1.0 + 0.15 * (m - 1)).to_numpy()


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

    if kind == "minimal":
        return X
    X["dt_slot"] = pd.Categorical(
        hour.clip(0, 26).astype(int).astype(str).radd("h"), categories=SLOT_LEVELS)
    if kind == "geo":
        for c in ("Origin", "Dest"):
            X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])
        return X
    for c, levels in CAT_LEVELS.items():
        X[c] = pd.Categorical(df[c], categories=levels)
    if kind == "full":
        chh = (dt // 100 * 2 + dt % 100 // 30).clip(0, 53).astype(int)
        X["carrier_halfhour"] = pd.Categorical(
            df["UniqueCarrier"].astype(str) + "_" + chh.astype(int).astype(str), categories=CHH_LEVELS)
    elif kind == "full_qh":
        qh = (dt // 100 * 4 + dt % 100 // 15).clip(0, 107).astype(int)
        X["carrier_qh"] = pd.Categorical(
            df["UniqueCarrier"].astype(str) + "_" + qh.astype(int).astype(str), categories=CQH_LEVELS)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- ensemble -------------------------------------------------------------------
TUNED = dict(n_estimators=1600, max_depth=4, learning_rate=0.04, tree_method="hist",
             enable_categorical=True, subsample=0.8, colsample_bytree=0.5,
             min_child_weight=10, reg_lambda=5.0, reg_alpha=1.0, n_jobs=N_JOBS)
BARE = dict(n_estimators=800, learning_rate=0.08, max_depth=6, n_jobs=N_JOBS)
# (params, view, weighted)
MEMBERS = [
    (dict(TUNED, max_depth=6, colsample_bytree=0.7, random_state=42), "full", True),
    (dict(TUNED, max_depth=6, colsample_bytree=0.7, random_state=7), "full", True),
    (dict(TUNED, max_depth=6, colsample_bytree=0.7, random_state=99), "full", True),
    (dict(TUNED, max_depth=5, random_state=2024), "full_qh", True),
    (dict(TUNED, max_depth=5, random_state=5), "full_qh", True),
    (dict(TUNED, random_state=2024), "geo", False),
    (dict(TUNED, random_state=123, min_child_weight=20), "minimal", True),
    (dict(TUNED, random_state=8, min_child_weight=20), "minimal", True),
    (dict(BARE, random_state=71), "full", True),
    (dict(BARE, random_state=72), "full", True),
    (dict(BARE, max_depth=5, random_state=71), "full_qh", True),
    (dict(BARE, max_depth=5, random_state=72), "full_qh", True),
    (dict(BARE, max_depth=4, random_state=71), "minimal", True),
    (dict(BARE, max_depth=4, random_state=72), "minimal", True),
    (dict(BARE, subsample=0.5, random_state=303), "full", True),
    (dict(BARE, subsample=0.5, random_state=305), "full", True),
]

ytr = to_y(train)
t0 = time.time()
models = []
for params, kind, weighted in MEMBERS:
    m = xgb.XGBClassifier(**params)
    m.fit(prepare(train, kind), ytr, sample_weight=MONTH_W if weighted else None)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = [m.predict_proba(prepare(df, kind))[:, 1] for m, (_, kind, _) in zip(models, MEMBERS)]
    return np.mean(P, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
