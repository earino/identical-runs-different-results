"""XGBoost binary classifier for flight delay prediction. THE ONLY FILE THE AGENT EDITS.

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


def _cint(s: pd.Series) -> pd.Series:
    return s.str.slice(2).astype(int)


def base_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    month = _cint(df["Month"])
    dom = _cint(df["DayofMonth"])
    dow = _cint(df["DayOfWeek"])
    dep = df["DepTime"].astype(int)
    X["month"] = month
    X["dom"] = dom
    X["dow"] = dow
    X["is_weekend"] = (dow >= 6).astype(int)
    X["dep_hour"] = dep // 100
    X["dep_minute"] = dep % 100
    X["dep_tod"] = X["dep_hour"] * 60 + X["dep_minute"]
    X["dep_tod_sin"] = np.sin(2 * np.pi * X["dep_tod"] / 1440.0)
    X["dep_tod_cos"] = np.cos(2 * np.pi * X["dep_tod"] / 1440.0)
    X["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    X["doy"] = pd.Series([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]).reindex(month).to_numpy() + dom
    X["distance"] = df["Distance"].astype(float)
    X["carrier"] = df["UniqueCarrier"].astype(str)
    X["origin"] = df["Origin"].astype(str)
    X["dest"] = df["Dest"].astype(str)
    return X


cat_cols = ["carrier", "origin", "dest"]

raw_train = base_frame(train)
cat_levels = {c: pd.Index(sorted(raw_train[c].dropna().unique())) for c in cat_cols}
freq_maps = {c: raw_train[c].value_counts() for c in cat_cols}
pair_maps = {
    "orig_hour": (raw_train["origin"] + "_" + raw_train["dep_hour"].astype(str)).value_counts(),
    "dest_hour": (raw_train["dest"] + "_" + raw_train["dep_hour"].astype(str)).value_counts(),
    "carrier_hour": (raw_train["carrier"] + "_" + raw_train["dep_hour"].astype(str)).value_counts(),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = base_frame(df)
    X["freq_orig_hour"] = (X["origin"] + "_" + X["dep_hour"].astype(str)).map(pair_maps["orig_hour"]).astype(float)
    X["freq_dest_hour"] = (X["dest"] + "_" + X["dep_hour"].astype(str)).map(pair_maps["dest_hour"]).astype(float)
    X["freq_carrier_hour"] = (X["carrier"] + "_" + X["dep_hour"].astype(str)).map(pair_maps["carrier_hour"]).astype(float)
    for c in cat_cols:
        X["freq_" + c] = X[c].map(freq_maps[c]).astype(float)
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    eps = 1e-6
    X["orig_hour_frac"] = X["freq_orig_hour"] / (X["freq_origin"] + eps)
    X["dest_hour_frac"] = X["freq_dest_hour"] / (X["freq_dest"] + eps)
    X["carrier_hour_frac"] = X["freq_carrier_hour"] / (X["freq_carrier"] + eps)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=18,
    learning_rate=0.02,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.3,
    reg_lambda=1.0,
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
