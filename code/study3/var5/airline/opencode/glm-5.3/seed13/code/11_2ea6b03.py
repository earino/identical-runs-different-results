"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- features -----------------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _num(s: pd.Series) -> pd.Series:
    return s.astype(str).str.split("-").str[-1].astype(float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    month = _num(df["Month"])
    day = _num(df["DayofMonth"])
    dow = _num(df["DayOfWeek"])
    X["month"], X["day"], X["dow"] = month, day, dow
    dep = df["DepTime"].clip(0, 2459).astype(int)
    hour, minute = dep // 100, dep % 100
    X["hour"], X["minute"] = hour, minute
    X["dep_min"] = hour * 60 + minute
    X["op_day_min"] = (hour * 60 + minute - 300) % 1440  # operational day starts 05:00
    for name, v, period in [
        ("h", hour, 24),
        ("mo", month, 12),
        ("dw", dow, 7),
        ("dm", day, 31),
    ]:
        X[f"{name}_sin"] = np.sin(2 * np.pi * v / period)
        X[f"{name}_cos"] = np.cos(2 * np.pi * v / period)
    X["distance"] = df["Distance"]
    hw = hour % 24
    for h in range(24):
        X[f"h{h:02d}"] = (hw == h).astype(int)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
t0 = time.time()
# covariate-shift importance weights: how 2006-like is each 2005 row?
Xtr_all, ytr_all = prepare(train), to_y(train)
Xev = prepare(evald)
yev = to_y(evald)
domain = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8,
    tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
)
Xd = pd.concat([Xtr_all, Xev], ignore_index=True)
yd = np.concatenate([np.zeros(len(Xtr_all)), np.ones(len(Xev))])
domain.fit(Xd, yd)
pw = domain.predict_proba(Xtr_all)[:, 1]
w = np.clip(pw / (1 - pw), 0.2, 5.0)
w = w / w.mean()
print(f"domain auc={roc_auc_score(yd, domain.predict_proba(Xd)[:, 1]):.3f} w range=[{w.min():.2f},{w.max():.2f}]")

members = []
for depth, lr in [(4, 0.05), (6, 0.05), (3, 0.1), (8, 0.03), (5, 0.07)]:
    m = xgb.XGBClassifier(
        max_depth=depth,
        learning_rate=lr,
        n_estimators=3000,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=30,
        colsample_bynode=0.6,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr_all, ytr_all, sample_weight=w, eval_set=[(Xev, yev)], verbose=False)
    members.append(m)
model = members[0]
print(f"Training time: {time.time() - t0:.1f}s, best_iters={[m.best_iteration for m in members]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in members], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
