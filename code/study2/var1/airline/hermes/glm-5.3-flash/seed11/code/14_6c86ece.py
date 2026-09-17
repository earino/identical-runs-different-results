"""XGBoost airline delay classifier (agent-edited). See program.md for the contract.

predict_proba(df) -> P(dep_delayed_15min == 'Y'). All feature engineering lives in prepare(df)
so the hidden holdout gets identical treatment. Encoders/statistics are fitted on train only.

This run: heterogeneous XGBoost bag (8 configs x 2 seeds) + smoothed dep_delay_rate
target encoding for the 5 low/med-cardinality categoricals, maps fitted on train only
(OOF-CV checked in exp8: ddr adds +0.002 in-sample AUC).
"""
import warnings

warnings.filterwarnings("ignore")

import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

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

DDR_COLS = ["Month", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
SMOOTH = 50.0  # smoothing strength toward the prior


def fit_ddr_maps(df: pd.DataFrame):
    y = (df[TARGET] == POSITIVE).astype(float)
    prior = float(y.mean())
    maps = {}
    for c in DDR_COLS:
        g = y.groupby(df[c]).agg(["mean", "count"])
        sm = (g["mean"] * g["count"] + prior * SMOOTH) / (g["count"] + SMOOTH)
        maps[c] = sm.to_dict()
    return maps, prior


def prepare(df: pd.DataFrame, ddr_maps=None, prior=None) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    if ddr_maps is not None:
        for c in DDR_COLS:
            X["ddr_" + c] = X[c].astype(object).map(ddr_maps[c]).astype(float).fillna(prior).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- OOF check of the encoding (diagnostic only) -------------------------------
y_full = to_y(train)
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
for tr_idx, va_idx in skf.split(train, y_full):
    mp, pr = fit_ddr_maps(train.iloc[tr_idx])
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED,
                          n_jobs=N_JOBS, n_estimators=60, learning_rate=0.05, max_depth=6,
                          subsample=0.9, colsample_bytree=0.85)
    m.fit(prepare(train.iloc[tr_idx], mp, pr), y_full[tr_idx], verbose=False)
    oof[va_idx] = m.predict_proba(prepare(train.iloc[va_idx], mp, pr))[:, 1]
print(f"OOF AUC (with ddr): {roc_auc_score(y_full, oof):.4f}")

# --- model: heterogeneous bag ---------------------------------------------------
CFGS = [(30, 0.1, 4), (60, 0.05, 4), (30, 0.1, 5), (60, 0.05, 5),
        (30, 0.1, 6), (60, 0.05, 6), (30, 0.1, 8), (60, 0.05, 8)]
MEMBERS = [c for c in CFGS for _ in range(2)]
CONFIG = dict(subsample=0.9, colsample_bytree=0.85)

mp, prior = fit_ddr_maps(train)
X = prepare(train, mp, prior)

t0 = time.time()
models = []
for k, (ne, lr, md) in enumerate(MEMBERS):
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED + k,
                          n_jobs=N_JOBS, n_estimators=ne, learning_rate=lr, max_depth=md, **CONFIG)
    m.fit(X, y_full, verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df, mp, prior)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
