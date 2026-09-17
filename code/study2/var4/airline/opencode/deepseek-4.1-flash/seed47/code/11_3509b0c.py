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

# label-free frequency / congestion encodings, fit on training data only
_cnt_cols = ["Origin", "Dest", "UniqueCarrier"]
cnt_maps = {c: train[c].astype(str).value_counts().to_dict() for c in _cnt_cols}
_train_hour = np.clip(train["DepTime"].to_numpy() // 100, 0, 23)
_oh = pd.DataFrame({"k": train["Origin"].astype(str) + "_" + _train_hour.astype(str)})
oh_maps = _oh["k"].value_counts().to_dict()
_dh = pd.DataFrame({"k": train["Dest"].astype(str) + "_" + _train_hour.astype(str)})
dh_maps = _dh["k"].value_counts().to_dict()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()

    dep = df["DepTime"].to_numpy()
    hour = np.clip(dep // 100, 0, 23)
    X["dep_hour"] = hour.astype("int16")
    X["dep_minofday"] = (hour * 60 + dep % 100).astype("int32")
    X["is_weekend"] = df["DayOfWeek"].isin(["c-6", "c-7"]).astype("int8")

    for c in _cnt_cols:
        X["cnt_" + c] = df[c].astype(str).map(cnt_maps[c]).fillna(0).astype("float32")
    X["cnt_origin_hour"] = (
        (df["Origin"].astype(str) + "_" + hour.astype(str)).map(oh_maps).fillna(0).astype("float32")
    )
    X["cnt_dest_hour"] = (
        (df["Dest"].astype(str) + "_" + hour.astype(str)).map(dh_maps).fillna(0).astype("float32")
    )
    X["frac_origin_hour"] = X["cnt_origin_hour"] / (X["cnt_Origin"] + 1.0)
    X["frac_dest_hour"] = X["cnt_dest_hour"] / (X["cnt_Dest"] + 1.0)

    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def make_model(spec: dict) -> xgb.XGBClassifier:
    params = dict(
        n_estimators=450,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        tree_method="hist",
        enable_categorical=True,
        n_jobs=N_JOBS,
    )
    params.update(spec)
    return xgb.XGBClassifier(**params)


SPECS = [
    dict(seed=42, max_depth=5),
    dict(seed=7, max_depth=4),
    dict(seed=123, max_depth=4),
    dict(seed=2024, max_depth=5),
    dict(seed=99, max_depth=4),
    dict(seed=1, max_depth=5),
    dict(seed=77, max_depth=3),
    dict(seed=55, max_depth=4),
    dict(seed=11, max_depth=6, min_child_weight=50, reg_lambda=20.0, subsample=0.7, colsample_bytree=0.7, n_estimators=500, learning_rate=0.02),
    dict(seed=21, max_depth=6, min_child_weight=50, reg_lambda=20.0, subsample=0.7, colsample_bytree=0.7, n_estimators=500, learning_rate=0.02),
    dict(seed=31, max_depth=7, min_child_weight=50, reg_lambda=20.0, subsample=0.7, colsample_bytree=0.7, n_estimators=500, learning_rate=0.02),
    dict(seed=41, max_depth=5, min_child_weight=50, reg_lambda=20.0, subsample=0.7, colsample_bytree=0.7, n_estimators=500, learning_rate=0.02),
    dict(seed=51, grow_policy="lossguide", max_depth=0, max_leaves=31, min_child_weight=30, reg_lambda=10.0, n_estimators=400, learning_rate=0.03),
    dict(seed=61, grow_policy="lossguide", max_depth=0, max_leaves=63, min_child_weight=50, reg_lambda=20.0, n_estimators=400, learning_rate=0.03),
]
X_train = prepare(train)
y_train = to_y(train)

t0 = time.time()
models = []
for spec in SPECS:
    m = make_model(spec)
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.zeros(len(X), dtype=float)
    for m in models:
        p += m.predict_proba(X)[:, 1]
    return p / len(models)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
