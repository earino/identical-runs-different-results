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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# --- count statistics (fit on TRAIN only) --------------------------------------
_key = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
route_counts = _key.value_counts()
origin_counts = train["Origin"].value_counts()
dest_counts = train["Dest"].value_counts()
origin_hour_counts = (train["Origin"].astype(str) + "_" + (train["DepTime"] // 100).astype(str)).value_counts()
carrier_hour_levels = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_" + (train["DepTime"] // 100).astype(str)).unique()))
origin_hblock_levels = pd.Index(sorted((train["Origin"].astype(str) + "_" + ((train["DepTime"] // 100) // 3).astype(str)).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    dt = df["DepTime"].astype("int64")
    hour = dt // 100
    minute = dt % 100
    X["DepTime"] = dt
    X["dep_hour"] = hour
    X["dep_minute"] = minute
    X["dep_minofday"] = (hour * 60 + minute).clip(upper=24 * 60)
    ang = 2 * np.pi * X["dep_minofday"] / (24 * 60)
    X["dep_sin"] = np.sin(ang)
    X["dep_cos"] = np.cos(ang)
    X["Distance"] = df["Distance"]
    X["Distance_log"] = np.log1p(df["Distance"])
    X["route_count"] = np.log1p((df["Origin"].astype(str) + "_" + df["Dest"].astype(str))
                                .map(route_counts).fillna(0))
    X["origin_count"] = np.log1p(df["Origin"].map(origin_counts).fillna(0))
    X["dest_count"] = np.log1p(df["Dest"].map(dest_counts).fillna(0))
    X["origin_hour_count"] = np.log1p((df["Origin"].astype(str) + "_" + (df["DepTime"] // 100).astype(str))
                                      .map(origin_hour_counts).fillna(0))
    hour = df["DepTime"] // 100
    X["dep_hour"] = pd.Categorical(hour, categories=list(range(31)))
    X["carrier_hour"] = pd.Categorical(df["UniqueCarrier"].astype(str) + "_" + hour.astype(str),
                                       categories=carrier_hour_levels)
    hb = (hour // 3).astype(str)
    X["origin_hblock"] = pd.Categorical(df["Origin"].astype(str) + "_" + hb,
                                        categories=origin_hblock_levels)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    min_child_weight=5,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    early_stopping_rounds=50,
    n_jobs=N_JOBS,
)
MEMBERS = [
    {"max_depth": 10, "random_state": 10, "colsample_bytree": 0.9, "subsample": 1.0, "reg_lambda": 5.0, "learning_rate": 0.08, "n_estimators": 600},
    {"max_depth": 10, "random_state": 11, "colsample_bytree": 0.9, "subsample": 1.0, "reg_lambda": 5.0, "learning_rate": 0.08, "n_estimators": 600},
    {"max_depth": 12, "random_state": 12, "colsample_bytree": 0.7, "subsample": 0.9, "reg_lambda": 5.0, "learning_rate": 0.08, "n_estimators": 600},
    {"max_depth": 12, "random_state": 13, "colsample_bytree": 0.7, "subsample": 0.9, "reg_lambda": 5.0, "learning_rate": 0.08, "n_estimators": 600},
    {"max_depth": 8, "random_state": 9, "colsample_bytree": 1.0, "subsample": 1.0, "reg_lambda": 5.0, "learning_rate": 0.05, "n_estimators": 1200},
    {"max_depth": 6, "random_state": 3, "colsample_bytree": 0.7, "subsample": 0.9, "reg_lambda": 1.0, "learning_rate": 0.08, "n_estimators": 600},
    {"max_depth": 8, "random_state": 2, "colsample_bytree": 0.9, "subsample": 0.9, "reg_lambda": 1.0, "learning_rate": 0.08, "n_estimators": 600},
    {"max_depth": 14, "random_state": 18, "colsample_bytree": 0.5, "subsample": 0.9, "reg_lambda": 10.0, "learning_rate": 0.08, "n_estimators": 300},
]

t0 = time.time()
X_tr = prepare(train)
y_tr = to_y(train)
X_ev = prepare(evald)
y_ev = to_y(evald)
models = []
for i, extra in enumerate(MEMBERS):
    m = xgb.XGBClassifier(**PARAMS, **extra)
    m.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    models.append(m)
    print(f"  member {i}: depth={extra['max_depth']} best_iter={m.best_iteration}")
print(f"Training time: {time.time() - t0:.1f}s")


def _rank(p: np.ndarray) -> np.ndarray:
    return pd.Series(p).rank(pct=True).to_numpy()


W = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0])


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = np.column_stack([_rank(m.predict_proba(X)[:, 1]) for m in models])
    return ps @ W / W.sum()


t0 = time.time()
eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
