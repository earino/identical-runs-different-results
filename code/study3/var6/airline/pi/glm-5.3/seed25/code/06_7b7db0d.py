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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]


def _num_from_c(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering: called by prepare(), which predict_proba() uses on unseen rows."""
    X = pd.DataFrame(index=df.index)
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 24)
    minute = dep % 100
    tod = (hour * 60 + minute) / 1440.0
    X["DepTime"] = dep
    X["hour"] = hour
    X["minute"] = minute
    X["tod_sin"] = np.sin(2 * np.pi * tod)
    X["tod_cos"] = np.cos(2 * np.pi * tod)

    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["Distance"] = dist
    X["log_distance"] = np.log1p(dist)

    X["Month"] = _num_from_c(df["Month"])
    X["DayofMonth"] = _num_from_c(df["DayofMonth"])
    X["DayOfWeek"] = _num_from_c(df["DayOfWeek"])

    X["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    X["Origin"] = df["Origin"].astype(str)
    X["Dest"] = df["Dest"].astype(str)
    X["route"] = X["Origin"] + "_" + X["Dest"]
    return X


# statistics fitted on TRAIN ONLY
def _fit_stats(F: pd.DataFrame):
    return {
        "origin_cnt": F["Origin"].value_counts(),
        "dest_cnt": F["Dest"].value_counts(),
        "route_cnt": F["route"].value_counts(),
        "carrier_cnt": F["UniqueCarrier"].value_counts(),
    }


def _add_counts(X: pd.DataFrame) -> pd.DataFrame:
    X["origin_cnt"] = np.log1p(X["Origin"].map(stats["origin_cnt"]).fillna(0))
    X["dest_cnt"] = np.log1p(X["Dest"].map(stats["dest_cnt"]).fillna(0))
    X["route_cnt"] = np.log1p(X["route"].map(stats["route_cnt"]).fillna(0))
    X["carrier_cnt"] = np.log1p(X["UniqueCarrier"].map(stats["carrier_cnt"]).fillna(0))
    return X


# categorical levels + count maps fitted on TRAIN ONLY
_feat_train = build_features(train)
cat_levels = {c: pd.Index(sorted(_feat_train[c].unique())) for c in CAT_COLS}
stats = _fit_stats(_feat_train)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = build_features(df)
    X = _add_counts(X)
    X = X.drop(columns=["route"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=1500,
    max_depth=24,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
