"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. Module-level `predict_proba(df)`: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives inside prepare(); statistics fit on train.csv only.

Model: diverse 5-model XGBoost ensemble (varied depth/lr/lambda/colsample/seeds),
probability-averaged. Fixed rounds per model, tuned on the flat post-peak region.
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
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- statistics fit on the training data only ----------------------------------
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in ["UniqueCarrier", "Origin", "Dest"]}
_tr_hour = (train["DepTime"].astype(int) // 100) % 24
HOUR_CARR_CATS = pd.Index(sorted((_tr_hour.astype(str) + "_" + train["UniqueCarrier"].astype(str)).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(int)
    hour = (dep // 100) % 24
    mins = hour * 60 + dep % 100
    X["DepTime"] = dep
    X["Distance"] = df["Distance"].astype(float)
    X["logDist"] = np.log1p(X["Distance"])
    X["Month"] = df["Month"].astype(str).str.replace("c-", "", regex=False).astype(int)
    X["DayofMonth"] = df["DayofMonth"].astype(str).str.replace("c-", "", regex=False).astype(int)
    X["DayOfWeek"] = df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False).astype(int)
    X["DepHour"] = hour
    X["DepMinute"] = dep % 100
    X["MinsOfDay"] = mins
    X["sin_tod"] = np.sin(2 * np.pi * mins / 1440.0)
    X["cos_tod"] = np.cos(2 * np.pi * mins / 1440.0)
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"], categories=cat_levels["UniqueCarrier"])
    X["Origin"] = pd.Categorical(df["Origin"], categories=cat_levels["Origin"])
    X["Dest"] = pd.Categorical(df["Dest"], categories=cat_levels["Dest"])
    X["HourCat"] = pd.Categorical(hour, categories=pd.Index(range(24)))
    X["TodCat"] = pd.Categorical((mins // 15).astype(int), categories=pd.Index(range(96)))
    X["HourCarr"] = pd.Categorical(
        hour.astype(str) + "_" + df["UniqueCarrier"].astype(str), categories=HOUR_CARR_CATS
    )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- diverse 5-model ensemble ---------------------------------------------------
BASE = dict(min_child_weight=20, subsample=0.7, colsample_bytree=0.7, reg_lambda=100, gamma=0.5)
MODELS = [
    (dict(n_estimators=650, learning_rate=0.05, max_depth=8, **BASE), 42),
    (dict(n_estimators=800, learning_rate=0.05, max_depth=6, **BASE), 1),
    (dict(n_estimators=650, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.6, reg_lambda=100, gamma=0.5, min_child_weight=20), 2),
    (dict(n_estimators=950, learning_rate=0.03, max_depth=8, **BASE), 3),
    (dict(n_estimators=350, learning_rate=0.1, max_depth=10, min_child_weight=20, subsample=0.7, colsample_bytree=0.7, reg_lambda=30, gamma=0.5), 4),
]

Xtrain = prepare(train)
ytrain = to_y(train)
models = []
t0 = time.time()
for params, seed in MODELS:
    m = xgb.XGBClassifier(
        tree_method="hist", enable_categorical=True, random_state=seed, n_jobs=N_JOBS, **params
    )
    m.fit(Xtrain, ytrain)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s for {len(models)} models")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
