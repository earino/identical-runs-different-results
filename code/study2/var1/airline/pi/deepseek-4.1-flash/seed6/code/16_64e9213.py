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
TOP_ORIGINS = set(train["Origin"].value_counts().head(25).index)
TOP_DESTS = set(train["Dest"].value_counts().head(25).index)
train["OriginTop"] = train["Origin"].where(train["Origin"].isin(TOP_ORIGINS), "OTHER")
evald["OriginTop"] = evald["Origin"].where(evald["Origin"].isin(TOP_ORIGINS), "OTHER")
train["DestTop"] = train["Dest"].where(train["Dest"].isin(TOP_DESTS), "OTHER")
evald["DestTop"] = evald["Dest"].where(evald["Dest"].isin(TOP_DESTS), "OTHER")
DIST_BINS = list(np.unique(train["Distance"].quantile(np.linspace(0, 1, 9)).to_numpy()))
train["DistBin"] = pd.cut(train["Distance"], bins=DIST_BINS, labels=False, include_lowest=True).astype(str)
for _name, _series in [("CarrierHour", train["UniqueCarrier"] + "_" + (train["DepTime"] // 100).astype(str)),
                       ("DowHour", train["DayOfWeek"] + "_" + (train["DepTime"] // 100).astype(str)),
                       ("OriginTopHour", train["OriginTop"] + "_" + (train["DepTime"] // 100).astype(str)),
                       ("DestTopHour", train["DestTop"] + "_" + (train["DepTime"] // 100).astype(str)),
                       ("OriginTopCarrier", train["OriginTop"] + "_" + train["UniqueCarrier"]),
                       ("TopRoute", train["OriginTop"] + "_" + train["DestTop"]),
                       ("CarrierLogDist", train["UniqueCarrier"] + "_" + train["DistBin"]),
                       ("DestTopCarrier", train["DestTop"] + "_" + train["UniqueCarrier"]),
                       ("DistBinHour", train["DistBin"] + "_" + (train["DepTime"] // 100).astype(str)),
                       ("OriginTopDistBin", train["OriginTop"] + "_" + train["DistBin"]),
                       ("DestTopDistBin", train["DestTop"] + "_" + train["DistBin"])]:
    train[_name] = _series
    cat_levels[_name] = pd.Index(sorted(_series.unique()))
    cat_cols = cat_cols + [_name]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    dep = X["DepTime"].astype(np.int32)
    X["hour"] = (dep // 100).astype(np.int16)
    X["minute"] = (dep % 100).astype(np.int16)
    X["tod"] = (X["hour"] * 60 + X["minute"]).astype(np.int16)
    X["tod_sin"] = np.sin(2 * np.pi * X["tod"] / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * X["tod"] / 1440.0)
    X["logdist"] = np.log1p(X["Distance"]).astype(np.float32)
    X = X.drop(columns=["DepTime"])
    X["CarrierHour"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "_" + (df["DepTime"].astype(np.int32) // 100).astype(str),
        categories=cat_levels["CarrierHour"],
    )
    X["DowHour"] = pd.Categorical(
        df["DayOfWeek"].astype(str) + "_" + (df["DepTime"].astype(np.int32) // 100).astype(str),
        categories=cat_levels["DowHour"],
    )
    _ot = df["Origin"].where(df["Origin"].isin(TOP_ORIGINS), "OTHER")
    X["OriginTopHour"] = pd.Categorical(
        _ot.astype(str) + "_" + (df["DepTime"].astype(np.int32) // 100).astype(str),
        categories=cat_levels["OriginTopHour"])
    _dt = df["Dest"].where(df["Dest"].isin(TOP_DESTS), "OTHER")
    X["DestTopHour"] = pd.Categorical(
        _dt.astype(str) + "_" + (df["DepTime"].astype(np.int32) // 100).astype(str),
        categories=cat_levels["DestTopHour"])
    X["OriginTopCarrier"] = pd.Categorical(
        _ot.astype(str) + "_" + df["UniqueCarrier"].astype(str),
        categories=cat_levels["OriginTopCarrier"])
    X["TopRoute"] = pd.Categorical(
        _ot.astype(str) + "_" + _dt.astype(str), categories=cat_levels["TopRoute"])
    _dbin = pd.cut(df["Distance"], bins=DIST_BINS, labels=False, include_lowest=True).astype("Int64").astype(str)
    X["CarrierLogDist"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "_" + _dbin, categories=cat_levels["CarrierLogDist"])
    X["DestTopCarrier"] = pd.Categorical(
        _dt.astype(str) + "_" + df["UniqueCarrier"].astype(str),
        categories=cat_levels["DestTopCarrier"])
    X["DistBinHour"] = pd.Categorical(
        _dbin + "_" + (df["DepTime"].astype(np.int32) // 100).astype(str),
        categories=cat_levels["DistBinHour"])
    X["OriginTopDistBin"] = pd.Categorical(
        _ot.astype(str) + "_" + _dbin, categories=cat_levels["OriginTopDistBin"])
    X["DestTopDistBin"] = pd.Categorical(
        _dt.astype(str) + "_" + _dbin, categories=cat_levels["DestTopDistBin"])
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# ensemble of XGBoost models (different seeds/depths) averaged for variance reduction
MEMBER_SPECS = [(5, 42), (5, 7), (5, 2024), (6, 42), (6, 7), (6, 2024), (7, 42), (7, 7), (7, 2024), (8, 42), (8, 7), (8, 2024)]
models = []
t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
for depth, seed in MEMBER_SPECS:
    m = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=depth,
        learning_rate=0.1,
        min_child_weight=2,
        reg_lambda=3.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.zeros(len(X))
    for m in models:
        p += m.predict_proba(X)[:, 1]
    return p / len(models)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
