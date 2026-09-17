"""Experiment 13: exp7 features, lr 0.03 + ES patience 200 (more rounds, finer shrinkage).
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


def _cat_levels(df: pd.DataFrame) -> dict[str, pd.Index]:
    lv = {c: pd.Index(sorted(df[c].dropna().unique())) for c in CAT_COLS}
    hour = (df["DepTime"] // 100).dropna().unique()
    lv["hour_cat"] = pd.Index(sorted(int(h) for h in hour))
    hc = (df["DepTime"] // 100).astype(int).astype(str) + "_" + df["UniqueCarrier"].astype(str)
    lv["hour_carrier"] = pd.Index(sorted(hc.dropna().unique()))
    dh = df["DayOfWeek"].astype(str) + "_" + (df["DepTime"] // 100).astype(int).astype(str)
    lv["dow_hour"] = pd.Index(sorted(dh.dropna().unique()))
    return lv


CAT_LEVELS = _cat_levels(train)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["month"] = df["Month"].str[2:].astype(int)
    X["day"] = df["DayofMonth"].str[2:].astype(int)
    X["dow"] = df["DayOfWeek"].str[2:].astype(int)
    dt = df["DepTime"]
    dep_min = (dt // 100) * 60 + dt % 100
    X["deptime"] = dt
    X["dep_min"] = dep_min
    X["dep_sin"] = np.sin(2 * np.pi * dep_min / 1440)
    X["dep_cos"] = np.cos(2 * np.pi * dep_min / 1440)
    X["hour"] = dt // 100
    X["distance"] = df["Distance"]
    X["log_dist"] = np.log1p(df["Distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])
    X["hour_cat"] = pd.Categorical(X["hour"].astype(int).astype(str), categories=CAT_LEVELS["hour_cat"].astype(str))
    hc = X["hour"].astype(int).astype(str) + "_" + df["UniqueCarrier"].astype(str)
    X["hour_carrier"] = pd.Categorical(hc, categories=CAT_LEVELS["hour_carrier"].astype(str))
    dh = df["DayOfWeek"].astype(str) + "_" + X["hour"].astype(int).astype(str)
    X["dow_hour"] = pd.Categorical(dh, categories=CAT_LEVELS["dow_hour"].astype(str))
    X["redeye"] = ((dep_min < 300) | (dep_min >= 1260)).astype(int)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


model = xgb.XGBClassifier(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=200,
    eval_metric="auc",
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s (best_iter={model.best_iteration})")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
