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

# --- target encodings (fit on training data only) ------------------------------
y_all = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = y_all.mean()
TE_M = 20  # smoothing: shrinks rare levels toward the prior


def te_map(keys: pd.Series) -> pd.Series:
    s = pd.Series(y_all).groupby(keys).agg(["sum", "count"])
    return (s["sum"] + PRIOR * TE_M) / (s["count"] + TE_M)


def route_of(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


TE = {
    "route": te_map(route_of(train)),
    "carrier": te_map(train["UniqueCarrier"]),
    "origin": te_map(train["Origin"]),
    "dest": te_map(train["Dest"]),
    "hour": te_map(train["DepTime"].astype(float) // 100),
}
carrier_levels = pd.Index(sorted(train["UniqueCarrier"].dropna().unique()))
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in ["UniqueCarrier", "Origin", "Dest"]}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["Month"] = df["Month"].astype(str).str.replace("c-", "", "").astype(float)
    X["DayofMonth"] = df["DayofMonth"].astype(str).str.replace("c-", "", "").astype(float)
    X["DayOfWeek"] = df["DayOfWeek"].astype(str).str.replace("c-", "", "").astype(float)
    dep = df["DepTime"].astype(float)
    X["DepTime"] = dep
    X["DepHour"] = (dep // 100)
    X["DepMinute"] = (dep % 100)
    X["DepFrac"] = dep / 2400.0
    X["Distance"] = df["Distance"].astype(float)
    for c in ["UniqueCarrier", "Origin", "Dest"]:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    X["TE_route"] = route_of(df).map(TE["route"]).fillna(PRIOR)
    X["TE_carrier"] = df["UniqueCarrier"].map(TE["carrier"]).fillna(PRIOR)
    X["TE_origin"] = df["Origin"].map(TE["origin"]).fillna(PRIOR)
    X["TE_dest"] = df["Dest"].map(TE["dest"]).fillna(PRIOR)
    X["TE_hour"] = (dep // 100).map(TE["hour"]).fillna(PRIOR)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=4000,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=80,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
