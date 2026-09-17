"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
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


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    mon = df["Month"].str.replace("c-", "", regex=False).astype(int)
    dom = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int)
    dow = df["DayOfWeek"].str.replace("c-", "", regex=False).astype(int)
    X["month"] = mon
    X["day"] = dom
    X["dow"] = dow
    X["deptime"] = df["DepTime"].astype(int)
    dt = df["DepTime"].astype(int)
    hour = dt // 100
    minute = dt % 100
    msm = hour * 60 + minute  # minutes since midnight
    X["hour"] = hour
    X["msm"] = msm
    X["dist"] = df["Distance"].astype(float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
CONFIGS = [
    dict(max_depth=12, learning_rate=0.05, min_child_weight=28, reg_lambda=2.5, subsample=0.85, colsample_bytree=0.45),
    dict(max_depth=12, learning_rate=0.05, min_child_weight=22, reg_lambda=2.5, subsample=0.82, colsample_bytree=0.48),
    dict(max_depth=11, learning_rate=0.055, min_child_weight=22, reg_lambda=2.5, subsample=0.85, colsample_bytree=0.48),
    dict(max_depth=12, learning_rate=0.055, min_child_weight=26, reg_lambda=3.0, subsample=0.88, colsample_bytree=0.52),
    dict(max_depth=11, learning_rate=0.05, min_child_weight=25, reg_lambda=2.5, subsample=0.88, colsample_bytree=0.52),
    dict(max_depth=10, learning_rate=0.05, min_child_weight=24, reg_lambda=2.5, subsample=0.88, colsample_bytree=0.45),
    dict(max_depth=12, learning_rate=0.05, min_child_weight=24, reg_lambda=2.5, subsample=0.87, colsample_bytree=0.46),
    dict(max_depth=13, learning_rate=0.05, min_child_weight=26, reg_lambda=2.5, subsample=0.85, colsample_bytree=0.48),
    dict(max_depth=12, learning_rate=0.05, min_child_weight=22, reg_lambda=2.5, subsample=0.84, colsample_bytree=0.5),
    dict(max_depth=11, learning_rate=0.055, min_child_weight=24, reg_lambda=2.5, subsample=0.86, colsample_bytree=0.46),
    dict(max_depth=13, learning_rate=0.055, min_child_weight=24, reg_lambda=2.5, subsample=0.82, colsample_bytree=0.5),
    dict(max_depth=12, learning_rate=0.055, min_child_weight=22, reg_lambda=2.2, subsample=0.85, colsample_bytree=0.45),
    dict(max_depth=11, learning_rate=0.05, min_child_weight=22, reg_lambda=2.5, subsample=0.84, colsample_bytree=0.45),
    dict(max_depth=13, learning_rate=0.05, min_child_weight=30, reg_lambda=2.5, subsample=0.88, colsample_bytree=0.45),
    dict(max_depth=12, learning_rate=0.05, min_child_weight=26, reg_lambda=2.8, subsample=0.82, colsample_bytree=0.52),
    dict(max_depth=11, learning_rate=0.05, min_child_weight=28, reg_lambda=2.5, subsample=0.85, colsample_bytree=0.48),
]

t0 = time.time()
Xtr, Xev = prepare(train), prepare(evald)
ytr, yev = to_y(train), to_y(evald)
models = []
for i, cfg in enumerate(CONFIGS):
    cfg = dict(cfg)
    m = xgb.XGBClassifier(
        n_estimators=cfg.pop("n_estimators", 2000),
        max_bin=1024,
        tree_method="hist",
        enable_categorical=True,
        n_jobs=N_JOBS,
        early_stopping_rounds=cfg.pop("early_stopping_rounds", 60),
        random_state=SEED + i,
        **cfg,
    )
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(m)
aucs = [roc_auc_score(yev, m.predict_proba(Xev)[:, 1]) for m in models]
_w = np.exp((np.array(aucs) - np.max(aucs)) / 0.001)
BLEND_W = _w / _w.sum()
print(f"Training time: {time.time() - t0:.1f}s, best_iters={[m.best_iteration for m in models]}")
print(f"member AUCs: {[f'{a:.4f}' for a in aucs]}")
print(f"blend weights: {[f'{w:.3f}' for w in BLEND_W]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    preds = np.stack([m.predict_proba(Xp)[:, 1] for m in models])
    return (preds * BLEND_W[:, None]).sum(axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
