"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

NUM_COLS = ["Distance", "DepTime"]  # numeric passthrough

# --- traffic-density statistics (target-free, fit on train only) -----------------
dens = {}


def _fit_density(df: pd.DataFrame) -> None:
    global dens
    hour = df["DepTime"].astype(int) // 100
    origin = df["Origin"].astype(str)
    dest = df["Dest"].astype(str)
    carrier = df["UniqueCarrier"].astype(str)
    hs = hour.astype(str)
    dens = {
        "origin": origin.value_counts().to_dict(),
        "origin_hour": (origin + "_" + hs).value_counts().to_dict(),
        "hour": hour.value_counts().to_dict(),
        "dest": dest.value_counts().to_dict(),
        "dest_hour": (dest + "_" + hs).value_counts().to_dict(),
        "carrier": carrier.value_counts().to_dict(),
        "carrier_hour": (carrier + "_" + hs).value_counts().to_dict(),
        "route": (origin + "_" + dest).value_counts().to_dict(),
    }


def _dens(name: str, keys: pd.Series) -> np.ndarray:
    return np.log1p(keys.map(dens[name]).fillna(0.0).to_numpy(dtype=float))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in cat_cols:
        if c == "DayofMonth" or c == "Month":
            continue  # keep only their int forms
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    for c in NUM_COLS:
        X[c] = df[c]
    # time-of-day engineering (delay risk rises through the day)
    dep = df["DepTime"].astype(int)
    minutes = (dep // 100) * 60 + dep % 100
    X["dep_min"] = minutes  # minutes since midnight
    X["dep_sin"] = np.sin(2 * np.pi * minutes / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * minutes / 1440.0)
    # calendar as ints (no DayofMonth: noise)
    for c in ("Month", "DayOfWeek"):
        X[c + "_i"] = df[c].str.slice(2).astype(int)
    # traffic density proxies (log counts from the 2005 training sample)
    hour = dep // 100
    origin = df["Origin"].astype(str)
    X["dens_origin"] = _dens("origin", origin)
    X["dens_origin_hour"] = _dens("origin_hour", origin + "_" + hour.astype(str))
    X["dens_hour"] = _dens("hour", hour)
    dest = df["Dest"].astype(str)
    X["dens_dest"] = _dens("dest", dest)
    X["dens_dest_hour"] = _dens("dest_hour", dest + "_" + hour.astype(str))
    carrier = df["UniqueCarrier"].astype(str)
    X["dens_carrier"] = _dens("carrier", carrier)
    X["dens_carrier_hour"] = _dens("carrier_hour", carrier + "_" + hour.astype(str))
    X["dens_route"] = _dens("route", origin + "_" + dest)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- models: small seed ensemble for variance reduction ------------------------
N_MODELS = 5
t0 = time.time()
ytr = to_y(train)
_fit_density(train)
Xtr = prepare(train)
models = []
for seed in range(SEED, SEED + N_MODELS):
    m = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.05,
        min_child_weight=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s for {N_MODELS} models")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
