"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Learned so far (see experiments.tsv):
  - Categorical Month overfits year-specific noise; a numeric month generalizes better.
  - Depth 3-4, strong reg_lambda, ~100-200 rounds: best transfer to 2006.
  - Target encoding / route categorical memorize 2005 noise and hurt.
  - Schedule-position features from train only help: o_pct = position of the departure time within the
    origin airport's daily schedule (ECDF over 30-min bins), o_offset = minutes vs the origin's mean time.
  - Averaging a few diverse XGB configs (depth/rounds/lr/subsampling) adds a robust gain.
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
CAT_COLS = ["DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _mins(dep: pd.Series) -> np.ndarray:
    return (dep // 100).to_numpy() * 60 + (dep % 100).to_numpy()


# origin-schedule stats computed on TRAIN ONLY (30-minute-bin ECDF per origin + mean dep time)
NB = 56  # bins cover 0..1650 minutes
_o_bins = np.clip(np.searchsorted(np.arange(0, NB * 30, 30), _mins(train["DepTime"]), side="right") - 1, 0, NB - 2)
_o_cnt = pd.crosstab(train["Origin"], _o_bins).reindex(columns=range(NB - 1), fill_value=0)
_o_cum = _o_cnt.cumsum(axis=1).to_numpy()
_o_ecdf = (_o_cum - _o_cnt.to_numpy() / 2 + 0.5) / (_o_cum[:, -1:] + 1.0)
_o_pos = pd.Series(np.arange(len(_o_cnt)), index=_o_cnt.index)
_o_mean = pd.Series(_mins(train["DepTime"])).groupby(train["Origin"]).mean()
_train_dep_mean = float(_o_mean.mean())


def _binof(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(np.arange(0, NB * 30, 30), arr, side="right") - 1, 0, NB - 2)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    dm = _mins(df["DepTime"])
    X["DepTime"] = df["DepTime"].astype(int)
    X["Distance"] = df["Distance"].astype(float)
    # numeric month (c-1..c-12 -> 1..12): ordered splits generalize better than per-level splits
    X["month_n"] = df["Month"].astype(str).str.replace("c-", "", regex=False).astype(int)
    # schedule position within the origin's day (robust to the 2005 -> 2006 shift)
    idx = df["Origin"].map(_o_pos).fillna(-1).to_numpy().astype(int)
    X["o_pct"] = np.where(idx >= 0, _o_ecdf[np.maximum(idx, 0), _binof(dm)], 0.5)
    X["o_offset"] = dm - df["Origin"].map(_o_mean).fillna(_train_dep_mean).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: small ensemble of diverse XGB configs ------------------------------
ENSEMBLE = [
    dict(n_estimators=100, max_depth=3, learning_rate=0.1, min_child_weight=5, reg_lambda=30.0),
    dict(n_estimators=200, max_depth=3, learning_rate=0.05, min_child_weight=5, reg_lambda=30.0),
    dict(n_estimators=150, max_depth=4, learning_rate=0.05, min_child_weight=5, reg_lambda=30.0),
    dict(n_estimators=150, max_depth=4, learning_rate=0.1, min_child_weight=5, reg_lambda=30.0,
         subsample=0.8, colsample_bytree=0.8),
]

t0 = time.time()
X_train, y_train = prepare(train), to_y(train)
models = []
for cfg in ENSEMBLE:
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED,
                          n_jobs=N_JOBS, **cfg)
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
