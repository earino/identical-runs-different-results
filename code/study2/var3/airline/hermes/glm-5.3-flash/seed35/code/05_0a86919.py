"""XGBoost binary classifier for flight delays. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Features (screened): Month/DayofMonth/DayOfWeek/UniqueCarrier/Origin/Dest as categoricals with train-fixed
appearance-order levels; DepTime -> dep_min + sin/cos (cyclic); carrier x 30-min-block categorical
(carrier delay patterns are strongly time-of-day dependent). Distance excluded from the base set (it hurt)
but one ensemble member re-adds it for diversity.
Model: small ensemble of XGBoost classifiers (depth / feature-set diversity), probability average.
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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

cat_levels = {c: pd.Index(train[c].dropna().unique()) for c in CAT_COLS}  # appearance order: screened best


def prepare(df: pd.DataFrame, with_dist: bool = False) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[CAT_COLS].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    m = (df["DepTime"] // 100) * 60 + (df["DepTime"] % 100)
    X["dep_min"] = m.astype(float)
    X["dep_sin"] = np.sin(2 * np.pi * m / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * m / 1440.0)
    # carrier x 30-min block
    blk = (m // 30).astype(int)
    blk_train = ((train["DepTime"] // 100) * 60 + train["DepTime"] % 100) // 30
    X["car_blk"] = pd.Categorical(
        df["UniqueCarrier"] + "_" + blk.astype(str),
        categories=pd.Index((train["UniqueCarrier"] + "_" + blk_train.astype(int).astype(str)).unique()),
    )
    if with_dist:
        X["Distance"] = df["Distance"].astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: small diverse ensemble --------------------------------------------
def make_model(**kw):
    p = dict(n_estimators=1200, learning_rate=0.03, max_depth=6, tree_method="hist",
             enable_categorical=True, random_state=SEED, n_jobs=N_JOBS)
    p.update(kw)
    return xgb.XGBClassifier(**p)


MEMBERS = [
    ({}, False),                       # d6, base features
    ({}, True),                        # d6, + Distance
    ({"max_depth": 5}, False),         # d5
    ({"max_depth": 7}, True),          # d7, + Distance
    ({"max_depth": 5}, True),          # d5, + Distance
]

t0 = time.time()
y = to_y(train)
models = []
for kw, wd in MEMBERS:
    m = make_model(**kw)
    m.fit(prepare(train, with_dist=wd), y)
    models.append((m, wd))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    ps = [m.predict_proba(prepare(df, with_dist=wd))[:, 1] for m, wd in models]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
