"""XGBoost binary classifier for airline delay prediction.

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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
NUM_COLS = [
    "Distance", "logdist", "dep_hour", "dep_min", "dep_sin", "dep_cos", "DepTime_raw",
    "month", "dom", "dow", "is_weekend",
    "carrier_f", "origin_f", "dest_f", "route_f",
    "origin_hour_f", "carrier_hour_f", "dest_hour_f",
]


def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


def _features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["UniqueCarrier", "Origin", "Dest", "Distance"]].copy()
    dt = df["DepTime"].fillna(0).astype(int)
    X["logdist"] = np.log1p(X["Distance"])
    X["dep_hour"] = (dt // 100).clip(0, 23)
    X["dep_min"] = X["dep_hour"] * 60 + (dt % 100).clip(0, 59)
    X["dep_sin"] = np.sin(2 * np.pi * X["dep_min"] / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * X["dep_min"] / 1440.0)
    X["DepTime_raw"] = dt
    X["month"] = _cnum(df["Month"]).fillna(0).astype(int)
    X["dom"] = _cnum(df["DayofMonth"]).fillna(0).astype(int)
    X["dow"] = _cnum(df["DayOfWeek"]).fillna(0).astype(int)
    X["is_weekend"] = X["dow"].isin([6, 7]).astype(int)
    X["Route"] = _route(df)
    return X


# Categorical levels + frequency maps fit on TRAIN ONLY.
_train_feat = _features(train)
cat_levels = {c: pd.Index(sorted(_train_feat[c].dropna().unique())) for c in CAT_COLS}
_freq_src = {
    "carrier_f": train["UniqueCarrier"],
    "origin_f": train["Origin"],
    "dest_f": train["Dest"],
    "route_f": _train_feat["Route"],
    "origin_hour_f": _train_feat["Origin"].astype(str) + "_" + _train_feat["dep_hour"].astype(str),
    "carrier_hour_f": _train_feat["UniqueCarrier"].astype(str) + "_" + _train_feat["dep_hour"].astype(str),
    "dest_hour_f": _train_feat["Dest"].astype(str) + "_" + _train_feat["dep_hour"].astype(str),
}
_freq_map = {k: v.value_counts() for k, v in _freq_src.items()}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here so predict_proba() reproduces it on unseen rows.
    X = _features(df)
    X["carrier_f"] = np.log1p(X["UniqueCarrier"].map(_freq_map["carrier_f"]).fillna(0.0))
    X["origin_f"] = np.log1p(X["Origin"].map(_freq_map["origin_f"]).fillna(0.0))
    X["dest_f"] = np.log1p(X["Dest"].map(_freq_map["dest_f"]).fillna(0.0))
    X["route_f"] = np.log1p(X["Route"].map(_freq_map["route_f"]).fillna(0.0))
    oh = X["Origin"].astype(str) + "_" + X["dep_hour"].astype(str)
    X["origin_hour_f"] = np.log1p(oh.map(_freq_map["origin_hour_f"]).fillna(0.0))
    ch = X["UniqueCarrier"].astype(str) + "_" + X["dep_hour"].astype(str)
    X["carrier_hour_f"] = np.log1p(ch.map(_freq_map["carrier_hour_f"]).fillna(0.0))
    dh = X["Dest"].astype(str) + "_" + X["dep_hour"].astype(str)
    X["dest_hour_f"] = np.log1p(dh.map(_freq_map["dest_hour_f"]).fillna(0.0))
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X[CAT_COLS + NUM_COLS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: 3-seed ensemble of leaf-wise (lossguide) XGBoost trees ------------
ENSEMBLE = [
    dict(
        n_estimators=600,
        learning_rate=0.03,
        colsample_bytree=0.30,
        min_child_weight=5,
        grow_policy="lossguide",
        max_leaves=512,
        max_depth=0,
        random_state=s,
    )
    for s in range(3)
]

t0 = time.time()
_X_train = prepare(train)
_y_train = to_y(train)
models = []
for spec in ENSEMBLE:
    m = xgb.XGBClassifier(
        subsample=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        n_jobs=N_JOBS,
        **spec,
    )
    m.fit(_X_train, _y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
