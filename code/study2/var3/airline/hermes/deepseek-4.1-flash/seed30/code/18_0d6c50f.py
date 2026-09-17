"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
# treat string columns as categoricals, except very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _hour_str(dep_time: pd.Series) -> pd.Series:
    """hhmm -> hour of day as a string; the few >2400 values are clipped rather than dropped."""
    return (pd.to_numeric(dep_time, errors="coerce") // 100).clip(0, 23).astype("Int64").astype(str)


def _grouped(s: pd.Series, levels: pd.Index) -> pd.Series:
    """Anything below the support floor falls into a shared OTHER bucket."""
    return s.astype("object").where(s.isin(levels), "OTHER").astype(str)


# Airport codes have 282 levels; the long tail is too sparse to estimate reliably and drifts year to
# year, so levels with fewer than GROUP_MIN training rows are pooled.
GROUP_MIN = 100
GROUP_COLS = [c for c in ("Origin", "Dest") if c in cat_cols]
GROUP_LEVELS = {c: pd.Index(train[c].value_counts().loc[lambda v: v >= GROUP_MIN].index) for c in GROUP_COLS}

# Hour of day is the strongest signal (and monotone), so its interactions are spelled out explicitly:
# a depth-3 tree cannot build them by itself. Each key keeps only cells with enough training support.
def _key_carrier_hour(df: pd.DataFrame) -> pd.Series:
    return df["UniqueCarrier"].astype(str) + "_" + _hour_str(df["DepTime"])


def _key_origin_hour(df: pd.DataFrame) -> pd.Series:
    return _grouped(df["Origin"], GROUP_LEVELS["Origin"]) + "_" + _hour_str(df["DepTime"])


KEY_BUILDERS = {"carrier_hour": (_key_carrier_hour, 75), "origin_hour": (_key_origin_hour, 300)}
KEY_KEEP = {k: pd.Index(b(train).value_counts().loc[lambda v: v >= m].index) for k, (b, m) in KEY_BUILDERS.items()}
KEY_LEVELS = {k: pd.Index(sorted(v)).append(pd.Index(["OTHER"])) for k, v in KEY_KEEP.items()}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in GROUP_COLS:
        v = _grouped(df[c], GROUP_LEVELS[c])
        X[c] = pd.Categorical(v, categories=GROUP_LEVELS[c].append(pd.Index(["OTHER"])))
    for c in cat_cols:
        if c in GROUP_COLS:
            continue
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    for k, (build, _) in KEY_BUILDERS.items():
        key = build(df)
        X[k] = pd.Categorical(key.where(key.isin(KEY_KEEP[k]), "OTHER"), categories=KEY_LEVELS[k])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Diverse bag: individually the deeper members are slightly worse on 2006 than the depth-3 members,
# but their errors decorrelate, which is what an average exploits.
PARAMS = [
    dict(n_estimators=2000, max_depth=3, learning_rate=0.02, subsample=0.9, colsample_bytree=0.9, random_state=42),
    dict(n_estimators=1500, max_depth=4, learning_rate=0.02, subsample=0.85, colsample_bytree=0.85, random_state=7),
    dict(n_estimators=600, max_depth=6, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, random_state=2024),
    dict(n_estimators=2000, max_depth=3, learning_rate=0.02, subsample=0.8, colsample_bytree=0.7, random_state=31337),
    dict(n_estimators=1200, max_depth=5, learning_rate=0.02, subsample=0.85, colsample_bytree=0.75, random_state=9001),
]


def make_model(p: dict) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        tree_method="hist", enable_categorical=True, n_jobs=N_JOBS, max_bin=1024, **p
    )


Xtr, ytr = prepare(train), to_y(train)
models = []
t0 = time.time()
for p in PARAMS:
    m = make_model(p)
    m.fit(Xtr, ytr, verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
