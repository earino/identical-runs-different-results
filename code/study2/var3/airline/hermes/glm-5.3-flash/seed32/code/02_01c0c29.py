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


def _num(s):
    return pd.to_numeric(s, errors="coerce")


# --- stateless feature construction (applied identically to any raw df) -------
CAT_FEATURES = ["month", "dom", "dow", "carrier", "origin", "dest", "route",
                "car_orig", "dow_hour", "hour"]
NUM_FEATURES = ["dep_num", "dep_small", "dep_large", "dist", "log_dist",
                "hour_sin", "hour_cos", "month_num", "dom_num", "dow_num"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    F = pd.DataFrame(index=df.index)
    F["month"] = df["Month"].astype(str)
    F["dom"] = df["DayofMonth"].astype(str)
    F["dow"] = df["DayOfWeek"].astype(str)
    F["carrier"] = df["UniqueCarrier"].astype(str)
    F["origin"] = df["Origin"].astype(str)
    F["dest"] = df["Dest"].astype(str)
    F["route"] = F["origin"] + ">" + F["dest"]
    F["car_orig"] = F["carrier"] + "|" + F["origin"]
    dep = _num(df["DepTime"])
    hh = (dep // 100).clip(0, 25)
    mm = dep - (dep // 100) * 100
    minutes = (hh * 60 + mm) % 1440
    ang = 2 * np.pi * minutes / 1440.0
    F["hour"] = hh
    F["dow_hour"] = F["dow"] + "|" + hh.fillna(-1).astype(int).astype(str)
    F["hour_sin"] = np.sin(ang)
    F["hour_cos"] = np.cos(ang)
    F["dep_num"] = dep
    F["dep_small"] = (dep < 100).astype(float)
    F["dep_large"] = (dep >= 2400).astype(float)
    F["month_num"] = _num(F["month"].str.slice(2))
    F["dom_num"] = _num(F["dom"].str.slice(2))
    F["dow_num"] = _num(F["dow"].str.slice(2))
    dist = _num(df["Distance"])
    F["dist"] = dist
    F["log_dist"] = np.log1p(dist)
    return F


CAT_LEVELS = {c: pd.Index(sorted(add_features(train)[c].dropna().unique()))
              for c in CAT_FEATURES}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering applied to unseen rows must go through here.
    F = add_features(df)
    X = pd.DataFrame(index=df.index)
    for c in CAT_FEATURES:
        X[c] = pd.Categorical(F[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    for c in NUM_FEATURES:
        X[c] = pd.to_numeric(F[c], errors="coerce")
    return X


# --- model --------------------------------------------------------------------
PARAMS = dict(
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=50.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_all = prepare(train)
y01 = (train[TARGET] == POSITIVE).astype(int)
va = np.random.RandomState(SEED).rand(len(train)) < 0.1
m_es = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=100,
                         eval_metric="auc", **PARAMS)
m_es.fit(X_all[~va], y01[~va], eval_set=[(X_all[va], y01[va])], verbose=False)
best_it = int(getattr(m_es, "best_iteration", 1999)) + 1
print(f"ES best iteration: {best_it} ({time.time() - t0:.1f}s)")

model = xgb.XGBClassifier(n_estimators=best_it, **PARAMS)
t0 = time.time()
model.fit(X_all, y01)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


eval_auc = roc_auc_score((evald[TARGET] == POSITIVE).astype(int), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
