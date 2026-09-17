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
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# baseline: treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# Airport codes are 282 levels each; rare ones carry too little support to estimate reliably and
# drift year to year, so keep the busiest K and lump the tail into a single bucket.
GROUP_K = 25
GROUP_COLS = [c for c in ("Origin", "Dest") if c in cat_cols]
def _ch_key(df):
    hh = (pd.to_numeric(df["DepTime"], errors="coerce") // 100).clip(0, 23).astype("Int64").astype(str)
    return df["UniqueCarrier"].astype(str) + "_" + hh


# only carrier-hour cells with enough support get their own level; the sparse tail is pooled
CH_KEEP = pd.Index(_ch_key(train).value_counts().loc[lambda v: v >= 300].index)
CH_LEVELS = pd.Index(sorted(CH_KEEP)).append(pd.Index(["OTHER"]))

GROUP_LEVELS = {
    c: pd.Index(train[c].value_counts().nlargest(GROUP_K).index) for c in GROUP_COLS
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in GROUP_COLS:
        v = df[c].astype("object").where(df[c].isin(GROUP_LEVELS[c]), "OTHER")
        X[c] = pd.Categorical(v, categories=GROUP_LEVELS[c].append(pd.Index(["OTHER"])))
    for c in cat_cols:
        if c in GROUP_COLS:
            continue
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    ch = _ch_key(df)
    X["carrier_hour"] = pd.Categorical(ch.where(ch.isin(CH_KEEP), "OTHER"), categories=CH_LEVELS)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Diverse bag: individually the deeper members are slightly worse on 2006 than the depth-3 member,
# but their errors decorrelate, which is what an average exploits.
PARAMS = [
    dict(n_estimators=2000, max_depth=3, learning_rate=0.02, subsample=0.9, colsample_bytree=0.9, random_state=42),
    dict(n_estimators=1500, max_depth=4, learning_rate=0.02, subsample=0.85, colsample_bytree=0.85, random_state=7),
    dict(n_estimators=600, max_depth=6, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, random_state=2024),
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
