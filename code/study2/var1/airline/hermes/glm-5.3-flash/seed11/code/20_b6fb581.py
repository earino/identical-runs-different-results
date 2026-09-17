"""Synthesis 6 (after exp 39; 1 experiment left)

Best: e143ab1 = 0.7201. Composition: drop raw DayofMonth+Month categories (keep their
ddr encodings) + ddr for Dow/Carrier/Origin/Dest (smooth=50, unseen->prior) + raw
DepTime/Distance/UniqueCarrier/Origin/Dest + hetero bag 36 (12 families x 3 seeds,
subsample .9, colsample .85, prob mean). smooth=100 identical (0.7201); simpler wins,
staying at 50.
Last experiment (exp40): widen the depth range once more with depth 12 — the last
untested diversity axis that has paid twice before (4-8 -> +0.0008; 3-10 -> +0.0008).
If it fails, revert and finalize on e143ab1.
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
# raw DayofMonth/Month memorize 2005-specific patterns; their smoothed ddr stays
feature_cols = [c for c in feature_cols if c not in ("DayofMonth", "Month")]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

DDR_COLS = ["Month", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "DayofMonth"]
SMOOTH = 50.0


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
    X = df[feature_cols].copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    if ddr_maps is not None:
        for c in DDR_COLS:
            X["ddr_" + c] = df[c].astype(object).map(ddr_maps[c]).astype(float).fillna(prior).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


y_full = to_y(train)

# --- model: heterogeneous bag (best known composition) ---------------------------
CFGS = [(30, 0.1, 3), (60, 0.05, 3), (30, 0.1, 4), (60, 0.05, 4), (30, 0.1, 5), (60, 0.05, 5),
        (30, 0.1, 6), (60, 0.05, 6), (30, 0.1, 8), (60, 0.05, 8), (30, 0.1, 10), (60, 0.05, 10),
        (30, 0.1, 12), (60, 0.05, 12)]
MEMBERS = [c for c in CFGS for _ in range(3)]
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
