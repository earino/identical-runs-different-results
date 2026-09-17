"""XGBoost binary classifier for airline delays. THE ONLY FILE THE AGENT EDITS.

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
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _int_col(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


_freq = {c: train[c].value_counts(normalize=True) for c in ("Origin", "Dest")}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # deterministic holiday-travel windows (structural, transfers across years)
    m = _int_col(df["Month"])
    dom = _int_col(df["DayofMonth"])
    X["thanksgiving"] = ((m == 11) & (dom >= 18)).astype(int)
    X["xmas_ny"] = (((m == 12) & (dom >= 15)) | ((m == 1) & (dom <= 4))).astype(int)
    X["july4"] = ((m == 7) & (dom <= 8)).astype(int)
    X["memorial"] = ((m == 5) & (dom >= 24)).astype(int)
    X["labor"] = ((m == 9) & (dom <= 4)).astype(int)
    X["presidents"] = ((m == 2) & (dom >= 15) & (dom <= 21)).astype(int)
    X["springbreak"] = ((m == 3) & (dom >= 8) & (dom <= 21)).astype(int)
    dow = _int_col(df["DayOfWeek"])
    X["thanks_dow"] = X["thanksgiving"] * dow
    X["xmas_dow"] = X["xmas_ny"] * dow
    # smooth wrapped distance (days) to key holiday anchors
    cum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    doy = cum[(m.clip(1, 12) - 1).astype(int).to_numpy()] + dom.clip(1, 31).to_numpy()
    for name, anchor in (("d_jul4", 185), ("d_thanks", 328), ("d_xmas", 359)):
        d = np.abs(doy - anchor)
        d = np.minimum(d, 365 - d)
        X[name] = d
    # traffic levels (fitted on train only): busy hubs cascade delays
    X["origin_n"] = np.log1p(df["Origin"].map(_freq["Origin"]).fillna(0).to_numpy())
    X["dest_n"] = np.log1p(df["Dest"].map(_freq["Dest"]).fillna(0).to_numpy())
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
N_SEEDS = 12
DEPTHS = [4, 5, 6, 3, 4, 5, 6, 4, 5, 3, 4, 5]
models = []
t0 = time.time()
for i in range(N_SEEDS):
    m = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=DEPTHS[i % len(DEPTHS)],
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=SEED + i,
        n_jobs=N_JOBS,
    )
    m.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  best_iters={[m.best_iteration for m in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
