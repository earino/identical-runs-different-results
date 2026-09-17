"""XGBoost binary classifier: freq encodings; PROBE over shallow depths / tree counts.

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

# frequency (count) statistics fitted on TRAIN ONLY, applied inside prepare()
FREQ_COLS = [c for c in ("UniqueCarrier", "Origin", "Dest") if c in cat_cols]
freq_maps = {c: train[c].value_counts() for c in FREQ_COLS}
route_key = ("Origin", "Dest") if "Origin" in cat_cols and "Dest" in cat_cols else None
if route_key:
    freq_maps["route"] = train.groupby(list(route_key)).size().to_dict()  # route frequency (count of rows)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN

    # frequency encodings (stats from train only)
    for c in FREQ_COLS:
        X[c + "_freq"] = np.log1p(df[c].map(freq_maps[c]).astype("float64"))
    if route_key:
        X["route_freq"] = np.log1p(df[list(route_key)].apply(tuple, axis=1).map(freq_maps["route"]).astype("float64"))

    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


BASE_PARAMS = dict(
    learning_rate=0.05,
    min_child_weight=1,
    subsample=0.9,
    reg_lambda=1.0,
    reg_alpha=0.0,
    gamma=0.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
)

# --- model --------------------------------------------------------------------
# ensemble of XGBoost models: different seeds/depths/tree-counts, probability-averaged
ENSEMBLE = [
    dict(max_depth=3, n_estimators=250, seed=42),
    dict(max_depth=3, n_estimators=300, seed=7, colsample_bytree=0.8, subsample=0.8),
    dict(max_depth=3, n_estimators=350, seed=13, colsample_bytree=0.9, subsample=0.85),
    dict(max_depth=3, n_estimators=400, seed=99),
    dict(max_depth=4, n_estimators=300, seed=2024),
]

t0 = time.time()
X = prepare(train)
y = to_y(train)
Xe = prepare(evald)
ye = to_y(evald)

preds = []
ENS_MODELS = []
for i, cfg in enumerate(ENSEMBLE):
    seed = cfg.pop("seed")
    m = xgb.XGBClassifier(**{**BASE_PARAMS, "random_state": seed, **cfg})
    m.fit(X, y)
    p = m.predict_proba(Xe)[:, 1]
    auc = roc_auc_score(ye, p)
    preds.append(p)
    ENS_MODELS.append(m)
    print(f"model {i}: {cfg} eval_auc={auc:.4f}")
    cfg["seed"] = seed


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    ps = [m.predict_proba(Xp)[:, 1] for m in ENS_MODELS]
    return np.mean(ps, axis=0)


eval_auc = roc_auc_score(ye, predict_proba(evald))
print(f"Ensemble eval AUC: {eval_auc:.4f}")
print(f"Training time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
