"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- train-only statistics (applied to any incoming frame inside prepare) ------
# peak-hour congestion: how many flights of the same carrier / out of the same airport leave in this hour
_t = pd.to_numeric(train["DepTime"], errors="coerce").fillna(-1).astype("int64")
_hr = (_t // 100).clip(0, 23).astype(str)
hour_counts = {}
for c in ("Origin", "Dest", "UniqueCarrier"):
    hour_counts[c] = (train[c].astype(str) + "|" + _hr).value_counts().to_dict()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    t = pd.to_numeric(X["DepTime"], errors="coerce").fillna(-1).astype("int64")
    hour = (t // 100).clip(0, 23)
    minute = (t % 100).clip(0, 59)
    X = X.drop(columns=["DepTime"])  # hhmm as an int has broken geometry; tod is the honest representation
    X["dep_hour"] = hour
    X["dep_min"] = minute
    X["dep_tod"] = hour * 60 + minute
    hs = hour.astype(str)
    for c in ("Origin", "Dest", "UniqueCarrier"):
        X[f"{c}_hour_count"] = (X[c].astype(str) + "|" + hs).map(hour_counts[c]).fillna(0).astype("float32")
    # aircraft arriving at the origin during this hour (delay propagation from inbound congestion),
    # and the airport's total movements in the hour
    inbound = (X["Origin"].astype(str) + "|" + hs).map(hour_counts["Dest"]).fillna(0).astype("float32")
    X["origin_inbound_hour_count"] = inbound
    X["origin_movements_hour"] = X["Origin_hour_count"] + inbound
    # traffic at the same airport in earlier hours: late aircraft from the last waves delay this one
    for base_col, map_name, tag in (("Origin", "Origin", "orig_dep"), ("Origin", "Dest", "orig_inb"), ("Dest", "Dest", "dest_dep"), ("UniqueCarrier", "UniqueCarrier", "car_dep")):
        for lag in range(1, 13):
            hl = (hour - lag).clip(0, 23).astype(str)
            X[f"{tag}_lag{lag}"] = (X[base_col].astype(str) + "|" + hl).map(hour_counts[map_name]).fillna(0).astype("float32")
        for lead in range(1, 13):
            hl = (hour + lead).clip(0, 23).astype(str)
            X[f"{tag}_lead{lead}"] = (X[base_col].astype(str) + "|" + hl).map(hour_counts[map_name]).fillna(0).astype("float32")
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# bagged XGBoost: average models that actually differ (row/column subsampling + seed), a variance
# reduction that is what should transfer to the hidden holdout
ENSEMBLE = [
    dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=1.0, colsample_bytree=0.7, random_state=42),
    dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.7, colsample_bytree=1.0, random_state=7),
    dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=2024),
    dict(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9, random_state=1337),
    dict(n_estimators=500, max_depth=7, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, random_state=99),
    dict(n_estimators=800, max_depth=6, learning_rate=0.03, subsample=0.9, colsample_bytree=0.7, random_state=2025),
]
COMMON = dict(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)

t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
models = []
for p in ENSEMBLE:
    m = xgb.XGBClassifier(**COMMON, **p)
    m.fit(Xtr, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = prepare(df)
    return np.mean([m.predict_proba(P)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
