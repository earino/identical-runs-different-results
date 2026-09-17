"""XGBoost binary classifier for the airline delay task. THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Features (canonical order, append-only):
  dep_minute, Distance (linear); hh_block, Month, DayofMonth, DayOfWeek,
  UniqueCarrier, Origin, Dest (categorical, levels fit on train only).
Model: 5-seed bag. Per seed: ES on a stratified 15% split -> refit on full train
at that seed's best_iteration. Final prediction = mean of the 5 refit models.
E18 = E16 + grow_policy=lossguide.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# ---- canonical feature order (append-only) ------------------------------------
CAT_COLS = ["hh_block", "Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
PASS_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
FEATURE_COLS = ["dep_minute", "Distance"] + CAT_COLS

# categories learned from TRAIN ONLY; unseen levels in eval/holdout become NaN
hh_src = train["DepTime"] // 100
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in PASS_COLS}
cat_levels["hh_block"] = pd.Index(sorted(hh_src.dropna().unique()))
del hh_src


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    dt = df["DepTime"]
    X["dep_minute"] = (dt // 100) * 60 + (dt % 100)
    X["Distance"] = df["Distance"]
    X["hh_block"] = dt // 100
    for c in PASS_COLS:
        X[c] = df[c]
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X[FEATURE_COLS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_full, y_full = prepare(train), to_y(train)

common = dict(
    learning_rate=0.05,
    max_depth=0,
    grow_policy="lossguide",
    max_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
)

SEEDS = [42, 43, 44, 45, 46]
t0 = time.time()
MODELS = []
best_ns = []
for s in SEEDS:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full, test_size=0.15, random_state=s, stratify=y_full
    )
    es_model = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=50, **{**common, "random_state": s})
    es_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    n = int(es_model.best_iteration) + 1
    best_ns.append(n)
    m = xgb.XGBClassifier(n_estimators=n, **{**common, "random_state": s})
    m.fit(X_full, y_full, verbose=False)
    MODELS.append(m)
print(f"bag: best_iters={best_ns} time={time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = [m.predict_proba(X)[:, 1] for m in MODELS]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
