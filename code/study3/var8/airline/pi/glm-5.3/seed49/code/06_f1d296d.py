"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, module-level `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
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

# --- feature engineering -------------------------------------------------------
C_CAT = ["UniqueCarrier", "Origin", "Dest", "Month", "DayofMonth", "DayOfWeek"]
NUM_COLS = ["Distance", "DepTime"]

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in C_CAT}

# smoothed target encoding, fit on train ONLY (module level = fit on training data)
y_all = (train[TARGET] == POSITIVE).astype(int)
PRIOR = float(y_all.mean())


def _te_map(keys: pd.Series, m: float) -> pd.Series:
    g = y_all.groupby(keys)
    agg = pd.DataFrame({"s": g.sum(), "n": g.size()})
    te = (agg["s"] + PRIOR * m) / (agg["n"] + m)
    return te


TE_M = {"UniqueCarrier": 20.0, "Origin": 50.0, "Dest": 50.0, "route": 100.0}
te_maps = {c: _te_map(train[c], TE_M[c]) for c in ["UniqueCarrier", "Origin", "Dest"]}
te_maps["route"] = _te_map(train["Origin"].astype(str) + "_" + train["Dest"].astype(str), TE_M["route"])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here; predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(int)
    hour = (dep // 100).clip(0, 23)
    minute = (dep % 100).clip(0, 59)
    X["hour"] = hour / 23.0
    X["min_of_day"] = (hour * 60 + minute) / 1439.0
    for c in NUM_COLS:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    for c in C_CAT:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["te_carrier"] = df["UniqueCarrier"].map(te_maps["UniqueCarrier"]).fillna(PRIOR)
    X["te_origin"] = df["Origin"].map(te_maps["Origin"]).fillna(PRIOR)
    X["te_dest"] = df["Dest"].map(te_maps["Dest"]).fillna(PRIOR)
    X["te_route"] = route.map(te_maps["route"]).fillna(PRIOR)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=3000,
    max_depth=6,
    learning_rate=0.1,
    min_child_weight=1,
    subsample=1.0,
    colsample_bytree=1.0,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train),
          eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s, best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
