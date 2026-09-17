"""XGBoost bagged ensemble for airline delay prediction. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- feature definition -------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols
            if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# frequency features: traffic volume per carrier / airport / route (stable across years)
COUNT_COLS = ["UniqueCarrier", "Origin", "Dest", "Route", "carrier_origin", "origin_hour", "carrier_hour",
              "dest_hour", "route_hour", "carrier_route",
              "origin_dow", "carrier_dow", "origin_month", "carrier_month", "route_dow"]


def _count_key(df: pd.DataFrame, c: str) -> pd.Series:
    if c == "Route":
        return df["Origin"].astype("string") + "_" + df["Dest"].astype("string")
    if c == "carrier_origin":
        return df["UniqueCarrier"].astype("string") + "_" + df["Origin"].astype("string")
    if c == "origin_hour":
        return df["Origin"].astype("string") + "_" + (df["DepTime"] // 100).astype("string")
    if c == "carrier_hour":
        return df["UniqueCarrier"].astype("string") + "_" + (df["DepTime"] // 100).astype("string")
    if c == "dest_hour":
        return df["Dest"].astype("string") + "_" + (df["DepTime"] // 100).astype("string")
    if c == "route_hour":
        return _count_key(df, "Route") + "_" + (df["DepTime"] // 100).astype("string")
    if c == "carrier_route":
        return df["UniqueCarrier"].astype("string") + "_" + _count_key(df, "Route")
    if c == "carrier_origin_hour":
        return df["UniqueCarrier"].astype("string") + "_" + _count_key(df, "origin_hour")
    if c == "origin_dow_hour":
        return df["Origin"].astype("string") + "_" + df["DayOfWeek"].astype("string") + "_" + (df["DepTime"] // 100).astype("string")
    if c == "route_month":
        return _count_key(df, "Route") + "_" + df["Month"].astype("string")
    if c == "carrier_route_hour":
        return _count_key(df, "carrier_route") + "_" + (df["DepTime"] // 100).astype("string")
    if c == "route_dow_hour":
        return _count_key(df, "route_dow") + "_" + (df["DepTime"] // 100).astype("string")
    if c == "carrier_dest_hour":
        return df["UniqueCarrier"].astype("string") + "_" + _count_key(df, "dest_hour")
    if c == "carrier_route_dow":
        return _count_key(df, "carrier_route") + "_" + df["DayOfWeek"].astype("string")
    if c == "dest_dow_hour":
        return df["Dest"].astype("string") + "_" + df["DayOfWeek"].astype("string") + "_" + (df["DepTime"] // 100).astype("string")
    if c == "origin_dow":
        return df["Origin"].astype("string") + "_" + df["DayOfWeek"].astype("string")
    if c == "carrier_dow":
        return df["UniqueCarrier"].astype("string") + "_" + df["DayOfWeek"].astype("string")
    if c == "origin_month":
        return df["Origin"].astype("string") + "_" + df["Month"].astype("string")
    if c == "carrier_month":
        return df["UniqueCarrier"].astype("string") + "_" + df["Month"].astype("string")
    if c == "route_dow":
        return _count_key(df, "Route") + "_" + df["DayOfWeek"].astype("string")
    return df[c].astype("string")


# out-of-fold target (delay-rate) encoding of interactions; smooth toward the global prior
TE_KEYS = ["Route", "origin_hour", "carrier_hour", "carrier_origin", "route_hour",
           "dest_hour", "carrier_route", "origin_dow", "carrier_dow",
           "carrier_origin_hour", "origin_dow_hour", "route_month",
           "carrier_route_hour", "route_dow_hour", "carrier_dest_hour",
           "carrier_route_dow", "dest_dow_hour"]
COUNT_COLS = COUNT_COLS + [k for k in TE_KEYS if k not in COUNT_COLS]
FREQ = {c: _count_key(train, c).value_counts().to_dict() for c in COUNT_COLS}
TE_SMOOTH = 40.0
N_FOLDS = 5
_prior_te = float((train[TARGET] == POSITIVE).mean())


def _fit_te(keys: pd.DataFrame, y: np.ndarray) -> dict:
    out = {}
    for c in TE_KEYS:
        g = pd.DataFrame({"k": _count_key(keys, c).to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
        out[c] = ((g["sum"] + _prior_te * TE_SMOOTH) / (g["count"] + TE_SMOOTH)).to_dict()
    return out


ENCODERS = _fit_te(train, (train[TARGET] == POSITIVE).astype(int).to_numpy())


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dep = df["DepTime"].to_numpy()
    X["DepHour"] = (dep // 100).astype(float)
    X["DepMin"] = (dep % 100).astype(float)
    for c in COUNT_COLS:
        X["cnt_" + c] = _count_key(df, c).map(FREQ[c]).fillna(0).astype(float)
    for c in TE_KEYS:
        X["te_" + c] = _count_key(df, c).map(ENCODERS[c]).fillna(_prior_te).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: bagged ensemble of shallow, decorrelated trees --------------------
X_train = prepare(train)
y_train = to_y(train)
# leakage-free out-of-fold target encodings for the training rows
_fold = np.random.RandomState(SEED).randint(0, N_FOLDS, size=len(train))
for f in range(N_FOLDS):
    va_m = _fold == f
    enc = _fit_te(train.loc[~va_m], y_train[~va_m])
    for c in TE_KEYS:
        X_train.loc[va_m, "te_" + c] = _count_key(train.loc[va_m], c).map(enc[c]).fillna(_prior_te).to_numpy()
N_MODELS = 20

models = []
for i in range(N_MODELS):
    m = xgb.XGBClassifier(
        n_estimators=450,
        max_depth=5 + (i % 3),
        learning_rate=0.02,
        subsample=0.7 + 0.05 * (i % 4),
        colsample_bytree=0.2 + 0.1 * (i % 3),
        min_child_weight=20 + 8 * (i % 4),
        reg_lambda=8.0 + 4.0 * (i % 3),
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + 1000 * i,
        n_jobs=N_JOBS,
    )
    t0 = time.time()
    m.fit(X_train, y_train)
    models.append(m)
    print(f"model {i} trained in {time.time() - t0:.1f}s")


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
