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
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# frequency (count) encoding from train only — hub-ness, no target leakage
cnt_maps = {
    c: train[c].value_counts().astype("float32") for c in ("UniqueCarrier", "Origin", "Dest")
}
cnt_maps["Route"] = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).value_counts().astype("float32")
_tr_hour = (train["DepTime"] // 100).clip(0, 24)
cnt_maps["OriginHour"] = (train["Origin"].astype(str) + "_" + _tr_hour.astype(str)).value_counts().astype("float32")
cnt_maps["DestHour"] = (train["Dest"].astype(str) + "_" + _tr_hour.astype(str)).value_counts().astype("float32")
CNT_UNSEEN = 1.0  # count for levels never seen in train


def parse_c(s: pd.Series) -> pd.Series:
    """'c-4' -> 4 (float, NaN if malformed)."""
    return pd.to_numeric(s.astype("string").str.replace(r"^c-", "", regex=True), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    # numeric, cyclical versions of the calendar columns
    for c, period in (("Month", 12), ("DayofMonth", 31), ("DayOfWeek", 7)):
        n = parse_c(X[c])
        X[c] = n
        X[c + "_sin"] = np.sin(2 * np.pi * n / period)
        X[c + "_cos"] = np.cos(2 * np.pi * n / period)
    # departure time: hours + minutes + cyclical
    ht = (X["DepTime"] // 100).clip(0, 24)
    mt = X["DepTime"] % 100
    mins = ht * 60 + mt
    X["DepHour"] = ht
    X["DepMin"] = mt
    X["DepMinutes"] = mins
    X["DepMinutes_sin"] = np.sin(2 * np.pi * mins / 1440)
    X["DepMinutes_cos"] = np.cos(2 * np.pi * mins / 1440)
    # frequency encoding (unseen levels -> 1)
    route = X["Origin"].astype(str) + "_" + X["Dest"].astype(str)
    X["RouteCnt"] = route.map(cnt_maps["Route"]).fillna(CNT_UNSEEN).astype("float32")
    for c in ("UniqueCarrier", "Origin", "Dest"):
        X[c + "Cnt"] = X[c].map(cnt_maps[c]).fillna(CNT_UNSEEN).astype("float32")
    # airport-hour congestion: flight volume at that airport around that hour
    X["OriginHourCnt"] = (X["Origin"].astype(str) + "_" + ht.astype(str)).map(cnt_maps["OriginHour"]).fillna(CNT_UNSEEN).astype("float32")
    X["DestHourCnt"] = (X["Dest"].astype(str) + "_" + ht.astype(str)).map(cnt_maps["DestHour"]).fillna(CNT_UNSEEN).astype("float32")
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- models: ensemble of 3 XGBs with different seeds/depths -------------------
SPECS = [
    dict(max_depth=10, subsample=0.8, colsample_bytree=0.8, random_state=1),
    dict(max_depth=8, subsample=0.7, colsample_bytree=0.7, random_state=2),
    dict(max_depth=12, subsample=0.9, colsample_bytree=0.9, random_state=3),
]
models = [
    xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        min_child_weight=3,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        n_jobs=N_JOBS,
        **spec,
    )
    for spec in SPECS
]

t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
for m in models:
    m.fit(Xtr, ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
