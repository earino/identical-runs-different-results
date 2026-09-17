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
feature_cols = [c for c in feature_cols if c not in ("Month", "DayofMonth")]  # poor year-to-year transfer
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# baseline: treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _freq_map(s: pd.Series) -> pd.Series:
    return s.astype(str).value_counts()


_route_tr = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
freq_maps = {
    "OriginFreq": _freq_map(train["Origin"]),
    "DestFreq": _freq_map(train["Dest"]),
    "CarrierFreq": _freq_map(train["UniqueCarrier"]),
    "RouteFreq": _freq_map(_route_tr),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    X["DepHour"] = (dep // 100).clip(0, 23)
    X["DepMinutes"] = X["DepHour"] * 60 + (dep % 100)
    X["LogDistance"] = np.log1p(pd.to_numeric(df["Distance"], errors="coerce"))
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["OriginFreq"] = np.log1p(df["Origin"].astype(str).map(freq_maps["OriginFreq"]).fillna(0))
    X["DestFreq"] = np.log1p(df["Dest"].astype(str).map(freq_maps["DestFreq"]).fillna(0))
    X["CarrierFreq"] = np.log1p(df["UniqueCarrier"].astype(str).map(freq_maps["CarrierFreq"]).fillna(0))
    X["RouteFreq"] = np.log1p(route.map(freq_maps["RouteFreq"]).fillna(0))
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=8,
    learning_rate=0.03,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
