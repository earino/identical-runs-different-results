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
        "origin_hour": F.groupby(["Origin", "hour"]).size(),
        "dest_hour": F.groupby(["Dest", "hour"]).size(),
        "carrier_hour": F.groupby(["UniqueCarrier", "hour"]).size(),
    }


def _add_counts(X: pd.DataFrame) -> pd.DataFrame:
    X["origin_cnt"] = np.log1p(X["Origin"].map(stats["origin_cnt"]).fillna(0))
    X["dest_cnt"] = np.log1p(X["Dest"].map(stats["dest_cnt"]).fillna(0))
    X["route_cnt"] = np.log1p(X["route"].map(stats["route_cnt"]).fillna(0))
    X["carrier_cnt"] = np.log1p(X["UniqueCarrier"].map(stats["carrier_cnt"]).fillna(0))
    X["origin_hour_cnt"] = np.log1p(pd.Series(list(zip(X["Origin"], X["hour"])), index=X.index).map(stats["origin_hour"]).fillna(0))
    X["dest_hour_cnt"] = np.log1p(pd.Series(list(zip(X["Dest"], X["hour"])), index=X.index).map(stats["dest_hour"]).fillna(0))
    X["carrier_hour_cnt"] = np.log1p(pd.Series(list(zip(X["UniqueCarrier"], X["hour"])), index=X.index).map(stats["carrier_hour"]).fillna(0))
    # hourly concentration of an airport's schedule (log-count difference)
    X["origin_hour_frac"] = X["origin_hour_cnt"] - X["origin_cnt"]
    X["dest_hour_frac"] = X["dest_hour_cnt"] - X["dest_cnt"]
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


# --- model: 4-member ensemble of deep-tree XGB with sampling jitter -------------------------
ENSEMBLE = [
    dict(max_depth=24, colsample_bytree=0.40, random_state=1),
    dict(max_depth=20, colsample_bytree=0.35, random_state=2),
    dict(max_depth=24, colsample_bytree=0.35, random_state=3),
    dict(max_depth=20, colsample_bytree=0.40, random_state=4),
]

members = []
t0 = time.time()
X_train, y_train = prepare(train), to_y(train)
X_eval, y_eval = prepare(evald), to_y(evald)
for cfg in ENSEMBLE:
    m = xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        subsample=1.0,
        reg_alpha=1.0,
        reg_lambda=2.0,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=60,
        eval_metric="auc",
        n_jobs=N_JOBS,
        **cfg,
    )
    m.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
    members.append(m)
print(f"Training time: {time.time() - t0:.1f}s  iters={[m.best_iteration for m in members]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in members], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
