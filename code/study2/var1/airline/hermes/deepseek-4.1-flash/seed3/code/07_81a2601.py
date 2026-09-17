"""XGBoost binary classifier for the airline task. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
All feature engineering lives in `prepare()`, which is the only code path `predict_proba()` uses, so the same
transformations apply to the hidden holdout. Anything fit on data (category levels, frequency tables) is fit on
train only.
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
# drop high-cardinality string columns, keep the rest as XGBoost categoricals
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _hour(df: pd.DataFrame) -> pd.Series:
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    return (dep // 100).clip(0, 23)


def _tod(df: pd.DataFrame) -> pd.Series:
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    return _hour(df) * 60 + (dep % 100)


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


# train-only statistics / level sets used by prepare()
route_freq = _route(train).value_counts()
carrier_freq = train["UniqueCarrier"].value_counts()
car_hour_levels = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_" + _hour(train).astype(str)).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything
    # computed on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    h = _hour(df)
    tod = _tod(df)
    # departure hour is the dominant delay driver: interaction with carrier lets each airline have its own
    # intraday delay profile (peak congestion); frequency counts and log-distance add size/route-strength.
    X["car_hour"] = pd.Categorical(df["UniqueCarrier"].astype(str) + "_" + h.astype(str), categories=car_hour_levels)
    X["route_cnt"] = _route(df).map(route_freq).fillna(0).to_numpy()
    X["carrier_cnt"] = df["UniqueCarrier"].map(carrier_freq).fillna(0).to_numpy()
    X["Dist_log"] = np.log1p(df["Distance"].to_numpy(dtype=float))
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# A single XGBoost is high-variance on this small feature set; averaging several seeds is a cheap,
# robust variance reduction (and cannot leak: each member sees the same training data).
BASE = dict(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)
MEMBER_PARAMS = [
    dict(n_estimators=1000, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, min_child_weight=10),
    dict(n_estimators=600, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, min_child_weight=10),
    dict(n_estimators=800, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bynode=0.7, min_child_weight=10),
    dict(n_estimators=400, max_depth=5, learning_rate=0.04, subsample=0.9, colsample_bytree=0.8, min_child_weight=20),
]
SEEDS = [42, 777, 2025]
DRAWS = 2
N_MEMBERS = len(MEMBER_PARAMS) * len(SEEDS) * DRAWS

X_train = prepare(train)
y_train = to_y(train)

# random subspace: each member also sees a different 70% subset of the columns, which decorrelates the members
rng = np.random.default_rng(SEED)
all_cols = list(X_train.columns)
member_cols = [list(rng.choice(all_cols, size=int(rng.integers(8, len(all_cols) + 1)), replace=False))
               for _ in range(N_MEMBERS)]

t0 = time.time()
models = []
for i in range(N_MEMBERS):
    params = MEMBER_PARAMS[i % len(MEMBER_PARAMS)]
    s = SEEDS[(i // len(MEMBER_PARAMS)) % len(SEEDS)]
    # alternate categorical encoding strategy per member: partition splits vs one-hot, for extra diversity
    enc = dict(max_cat_to_onehot=16) if i % 2 == 0 else dict(max_cat_to_onehot=1)
    m = xgb.XGBClassifier(random_state=s, **BASE, **enc, **params)
    m.fit(X_train[member_cols[i]], y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  members={len(models)}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X[cols])[:, 1] for m, cols in zip(models, member_cols)], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
