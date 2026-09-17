"""XGBoost binary classifier for airline delays. THE ONLY FILE THE AGENT EDITS.

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


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


y_train = to_y(train)
PRIOR = float(y_train.mean())
SMOOTH = 30.0

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "hour_c", "slot_c"]
TE_KEYS = ["Month", "DayofMonth", "DayOfWeek"]


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row derived features. No fitted statistics in here."""
    out = pd.DataFrame(index=df.index)
    out["DepTime"] = pd.to_numeric(df["DepTime"], errors="coerce")
    dt = out["DepTime"]
    hour = dt // 100
    out["hour_c"] = (hour % 24).astype("Int64").astype(str)
    out["dep_min"] = hour * 60 + dt % 100
    out["slot_c"] = (out["dep_min"] // 15).astype("Int64").astype(str)
    out["dep_sin"] = np.sin(2 * np.pi * out["dep_min"] / 1440.0)
    out["dep_cos"] = np.cos(2 * np.pi * out["dep_min"] / 1440.0)
    out["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    out["log_distance"] = np.log1p(out["Distance"])
    out["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    out["Origin"] = df["Origin"].astype(str)
    out["Dest"] = df["Dest"].astype(str)
    out["Month"] = df["Month"].astype(str)
    out["DayofMonth"] = df["DayofMonth"].astype(str)
    out["DayOfWeek"] = df["DayOfWeek"].astype(str)
    return out


feat_train = base_features(train)
cat_levels = {c: pd.Index(sorted(feat_train[c].unique())) for c in CAT_COLS}

_month_num = pd.to_numeric(feat_train["Month"].str.replace("c-", "", regex=False))
sample_w = (1.0 + 0.5 * (_month_num - 1) / 11.0).to_numpy()

te_maps = {}
for key in TE_KEYS:
    g = pd.DataFrame({"k": feat_train[key], "y": y_train}).groupby("k")["y"].agg(["size", "sum"])
    te_maps[key] = ((g["sum"] + SMOOTH * PRIOR) / (g["size"] + SMOOTH)).to_dict()

NUMERIC = ["DepTime", "dep_min", "dep_sin", "dep_cos", "Distance", "log_distance"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    f = base_features(df)
    X = pd.DataFrame(index=df.index)
    for c in NUMERIC:
        X[c] = f[c]
    for c in CAT_COLS:
        X[c] = pd.Categorical(f[c], categories=cat_levels[c])  # unseen levels -> NaN
    for key in TE_KEYS:
        X["te_" + key] = f[key].map(te_maps[key]).astype(float).fillna(PRIOR)
    return X


def fit_one(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.033,
        max_depth=9,
        min_child_weight=50,
        gamma=5.0,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


SEEDS = [42, 7, 2026, 1, 2, 3, 4, 5, 6, 8, 9, 10]

t0 = time.time()
X_full = prepare(train)
models = []
for seed in SEEDS:
    m = fit_one(seed)
    m.fit(X_full, y_train, sample_weight=sample_w)
    models.append(m)
print(f"features={X_full.shape[1]} models={len(models)}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = np.mean([m.predict_proba(prepare(df))[:, 1] for m in models], axis=0)
    return P


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
