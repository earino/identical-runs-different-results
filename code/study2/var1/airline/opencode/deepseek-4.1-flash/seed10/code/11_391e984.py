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
freq_cols = ["UniqueCarrier", "Origin", "Dest"]
freq_maps = {c: train[c].value_counts() for c in freq_cols}
freq_maps["Route"] = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).value_counts()
_oh = train["Origin"].astype(str) + "_" + (pd.to_numeric(train["DepTime"], errors="coerce") // 100).astype("Int64").astype(str)
freq_maps["OriginHour"] = _oh.value_counts()
_dh = train["Dest"].astype(str) + "_" + (pd.to_numeric(train["DepTime"], errors="coerce") // 100).astype("Int64").astype(str)
freq_maps["DestHour"] = _dh.value_counts()


def _doy(df: pd.DataFrame) -> pd.Series:
    mon = df["Month"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    dom = df["DayofMonth"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    cum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=float)
    return pd.Series(cum[(mon - 1).clip(0, 11).astype(int)] + dom, index=df.index)


_od = train["Origin"].astype(str) + "_" + _doy(train).astype(str)
freq_maps["OriginDay"] = _od.value_counts()
_dd = train["Dest"].astype(str) + "_" + _doy(train).astype(str)
freq_maps["DestDay"] = _dd.value_counts()
_ch = train["UniqueCarrier"].astype(str) + "_" + (pd.to_numeric(train["DepTime"], errors="coerce") // 100).astype("Int64").astype(str)
freq_maps["CarrierHour"] = _ch.value_counts()
_cday = train["UniqueCarrier"].astype(str) + "_" + _doy(train).astype(str)
freq_maps["CarrierDay"] = _cday.value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    X["DepHour"] = (dt // 100).clip(0, 23)
    X["DepMin"] = (dt % 100).clip(0, 59)
    X["DepMinOfDay"] = X["DepHour"] * 60 + X["DepMin"]
    X["DepSin"] = np.sin(2 * np.pi * X["DepMinOfDay"] / 1440)
    X["DepCos"] = np.cos(2 * np.pi * X["DepMinOfDay"] / 1440)
    mon = df["Month"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    dom = df["DayofMonth"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    dow = df["DayOfWeek"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    cum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=float)
    X["DayOfYear"] = cum[(mon - 1).clip(0, 11).astype(int)] + dom
    X["IsWeekend"] = (dow >= 6).astype(float)
    for c in freq_cols:
        X[c + "_freq"] = df[c].map(freq_maps[c]).fillna(0).astype(float)
    X["Route_freq"] = (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).map(freq_maps["Route"]).fillna(0).astype(float)
    _oh = df["Origin"].astype(str) + "_" + (dt // 100).astype("Int64").astype(str)
    X["OriginHour_freq"] = _oh.map(freq_maps["OriginHour"]).fillna(0).astype(float)
    _dh = df["Dest"].astype(str) + "_" + (dt // 100).astype("Int64").astype(str)
    X["DestHour_freq"] = _dh.map(freq_maps["DestHour"]).fillna(0).astype(float)
    _od = df["Origin"].astype(str) + "_" + X["DayOfYear"].astype("Int64").astype(str)
    X["OriginDay_freq"] = _od.map(freq_maps["OriginDay"]).fillna(0).astype(float)
    _dd = df["Dest"].astype(str) + "_" + X["DayOfYear"].astype("Int64").astype(str)
    X["DestDay_freq"] = _dd.map(freq_maps["DestDay"]).fillna(0).astype(float)
    X["OriginHour_ratio"] = X["OriginHour_freq"] / X["Origin_freq"].replace(0, np.nan)
    X["DestHour_ratio"] = X["DestHour_freq"] / X["Dest_freq"].replace(0, np.nan)
    X["OriginDay_ratio"] = X["OriginDay_freq"] / X["Origin_freq"].replace(0, np.nan)
    _ch = df["UniqueCarrier"].astype(str) + "_" + (dt // 100).astype("Int64").astype(str)
    X["CarrierHour_freq"] = _ch.map(freq_maps["CarrierHour"]).fillna(0).astype(float)
    _cday = df["UniqueCarrier"].astype(str) + "_" + X["DayOfYear"].astype("Int64").astype(str)
    X["CarrierDay_freq"] = _cday.map(freq_maps["CarrierDay"]).fillna(0).astype(float)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
Xtr = prepare(train)
ytr = to_y(train)

PARAMS = dict(
    n_estimators=3000,
    max_depth=4,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=30,
    reg_lambda=20.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
SEEDS = [42, 7, 2024]
models = []
t0 = time.time()
for s in SEEDS:
    m = xgb.XGBClassifier(random_state=s, **PARAMS)
    m.fit(Xtr, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  n_models={len(models)}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return p


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
