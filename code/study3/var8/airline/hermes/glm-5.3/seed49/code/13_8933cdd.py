"""XGBoost binary classifier for airline delay. THE ONLY FILE THE AGENT EDITS.

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

# --- features ----------------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
# interaction categoricals: airport/carrier x hour-of-day (delay risk is strongly time-of-day dependent)
HALF = lambda d: ((d["DepTime"] // 100) * 2 + (d["DepTime"] % 100) // 30).astype(str)
INTERACTIONS = {
    "OriginHour": lambda d: d["Origin"].astype(str) + "_" + (d["DepTime"] // 100).astype(str),
    "CarrierHour": lambda d: d["UniqueCarrier"].astype(str) + "_" + (d["DepTime"] // 100).astype(str),
    "DestHour": lambda d: d["Dest"].astype(str) + "_" + (d["DepTime"] // 100).astype(str),
    "OriginHalf": lambda d: d["Origin"].astype(str) + "_" + HALF(d),
    "DestHalf": lambda d: d["Dest"].astype(str) + "_" + HALF(d),
}
interaction_levels = {name: pd.Index(sorted(fn(train).unique())) for name, fn in INTERACTIONS.items()}


def _int_c(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("c-", "", regex=False).astype(int)


# volume stats computed on the training set only (airport congestion proxies)
_oh_keys = train["Origin"].astype(str) + "_" + (train["DepTime"] // 100).astype(str)
_dh_keys = train["Dest"].astype(str) + "_" + (train["DepTime"] // 100).astype(str)
_oh_counts = _oh_keys.value_counts()
_dh_counts = _dh_keys.value_counts()
_o_totals = train["Origin"].value_counts()
_d_totals = train["Dest"].value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["Month"] = _int_c(df["Month"])
    X["DayofMonth"] = _int_c(df["DayofMonth"])
    X["DayOfWeek"] = _int_c(df["DayOfWeek"])
    dep = df["DepTime"]
    X["DepTime"] = dep
    X["Hour"] = dep // 100
    X["Minute"] = dep % 100
    X["DepMin"] = (dep // 100) * 60 + dep % 100
    X["Distance"] = df["Distance"]
    X["dist_x_hour"] = df["Distance"] * (df["DepTime"] // 100)
    X["sin_depmin"] = np.sin(2 * np.pi * X["DepMin"] / 1440)
    X["cos_depmin"] = np.cos(2 * np.pi * X["DepMin"] / 1440)
    oh = df["Origin"].astype(str) + "_" + (df["DepTime"] // 100).astype(str)
    dh = df["Dest"].astype(str) + "_" + (df["DepTime"] // 100).astype(str)
    X["vol_oh"] = oh.map(_oh_counts).fillna(0).to_numpy()
    X["vol_dh"] = dh.map(_dh_counts).fillna(0).to_numpy()
    X["vol_oh_norm"] = X["vol_oh"] / df["Origin"].map(_o_totals).fillna(1).to_numpy()
    X["vol_dh_norm"] = X["vol_dh"] / df["Dest"].map(_d_totals).fillna(1).to_numpy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])  # unseen -> NaN
    for name, fn in INTERACTIONS.items():
        X[name] = pd.Categorical(fn(df), categories=interaction_levels[name])  # unseen -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# sample weights: upweight the last three months of the training year (train=2005, eval/holdout=2006)
w_train = np.where(_int_c(train["Month"]).to_numpy() >= 10, 2.0, 1.0)

# --- ensemble: 2 full-feature models + 8 feature-bagged models ----------------
# feature bags: all core numeric/time features + a random 4 of the categorical/interaction features.
# Diverse feature subsets decorrelate the trees more than seeds or hyperparameters do.
ALL_FEATURES = list(prepare(train).columns)
CORE = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "Hour", "Minute", "DepMin",
        "Distance", "sin_depmin", "cos_depmin", "vol_oh", "vol_oh_norm", "vol_dh", "vol_dh_norm"]
NONCORE = [c for c in ALL_FEATURES if c not in CORE]

ENSEMBLE = [(SEED, None), (SEED + 1, None)]
rng = np.random.RandomState(0)
for i in range(8):
    bag = list(rng.choice(NONCORE, size=4, replace=False))
    ENSEMBLE.append((SEED + 100 + i, CORE + bag))

X_train = prepare(train)
y_train = to_y(train)
models = []
t0 = time.time()
for seed, cols in ENSEMBLE:
    # full-feature members get lighter L1 (they see all signals); bagged members keep stronger L1
    alpha = 2 if cols is None else 5
    m = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.05,
        reg_alpha=alpha,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    feats = cols if cols is not None else list(X_train.columns)
    m.fit(X_train[feats], y_train, sample_weight=w_train)
    models.append((m, feats))
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X[feats])[:, 1] for m, feats in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
