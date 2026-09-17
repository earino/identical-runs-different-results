"""XGBoost binary classifier for airline delays. Contract: see program.md.

Exp 12: out-of-fold target encoding (5-fold) for the training fit to cut TE
self-leak; prediction path uses full-train smoothed maps as before.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

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

# smoothed target encodings, fit on TRAIN ONLY, applied inside prepare()
y_bin = (train[TARGET] == POSITIVE).astype(float)
GLOBAL_MEAN = float(y_bin.mean())
TE_K = {"UniqueCarrier": 50, "Origin": 200, "Dest": 200}


def _te_map(c: str, rows: pd.Index, k: float) -> pd.Series:
    stats = y_bin.loc[rows].groupby(train.loc[rows, c]).agg(["mean", "count"])
    return ((stats["count"] * stats["mean"] + k * GLOBAL_MEAN) / (stats["count"] + k))


te_maps = {c: _te_map(c, train.index, k) for c, k in TE_K.items()}

# out-of-fold encodings for the training fit (reduce self-leak); one variant per ensemble member
OOF_FOLDS = 5
te_oof_variants = []
for v in range(5):
    kf = KFold(n_splits=OOF_FOLDS, shuffle=True, random_state=SEED + v)
    te_oof = {c: pd.Series(GLOBAL_MEAN, index=train.index, dtype=float) for c in TE_K}
    for tr_idx, va_idx in kf.split(train):
        tr_idx, va_idx = train.index[tr_idx], train.index[va_idx]
        for c, k in TE_K.items():
            te_oof[c].loc[va_idx] = train.loc[va_idx, c].map(_te_map(c, tr_idx, k)).astype(float).fillna(GLOBAL_MEAN)
    te_oof_variants.append(te_oof)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    for c, m in te_maps.items():
        X[c + "_te"] = df[c].map(m).astype(float).fillna(GLOBAL_MEAN)
    # engineered numeric features (all derived from the row itself, nothing fit on data)
    dt = pd.to_numeric(df["DepTime"])
    hour = (dt // 100).astype(float)
    minute = (dt % 100).astype(float)
    tmin = hour * 60 + minute
    X["hour"] = hour
    X["minute"] = minute
    X["tod_sin"] = np.sin(2 * np.pi * tmin / 1440)
    X["tod_cos"] = np.cos(2 * np.pi * tmin / 1440)
    X["log_dist"] = np.log1p(pd.to_numeric(df["Distance"]))
    X["is_night"] = ((hour >= 21) | (hour <= 5)).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
BASE = dict(
    learning_rate=0.03,
    tree_method="hist",
    enable_categorical=True,
    colsample_bynode=0.35,
    n_jobs=N_JOBS,
)
CONFIGS = [  # diversity: vary depth / subsample / colsample / regularization
    dict(n_estimators=500, max_depth=6, subsample=0.7, colsample_bytree=0.7, min_child_weight=10, gamma=1.0, random_state=42),
    dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=10, gamma=0.5, random_state=43),
    dict(n_estimators=500, max_depth=7, subsample=0.6, colsample_bytree=0.6, min_child_weight=20, gamma=2.0, random_state=44),
    dict(n_estimators=500, max_depth=6, subsample=0.7, colsample_bytree=0.5, min_child_weight=5, gamma=1.0, random_state=45),
    dict(n_estimators=600, max_depth=6, learning_rate=0.02, subsample=0.9, colsample_bytree=0.9, min_child_weight=15, gamma=1.0, random_state=46),
]

t0 = time.time()
y_fit = to_y(train)
models = []
for i, cfg in enumerate(CONFIGS):
    X_fit = prepare(train)
    for c in TE_K:  # member-specific out-of-fold encodings
        X_fit[c + "_te"] = te_oof_variants[i % len(te_oof_variants)][c].to_numpy()
    m = xgb.XGBClassifier(**{**BASE, **cfg})
    m.fit(X_fit, y_fit)
    models.append(m)
print(f"Training time ({len(models)} models): {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
