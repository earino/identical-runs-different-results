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

# --- features -----------------------------------------------------------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
NUM_COLS = ["DepTime", "dep_min", "sin_t", "cos_t", "Distance"]
INTER_COLS = ["car_m24", "hh_dist"]
FEATURES = NUM_COLS + CAT_COLS + INTER_COLS


def _hh_str(df: pd.DataFrame) -> pd.Series:
    return ((((df["DepTime"].astype(float) // 100) % 24) * 60 + df["DepTime"].astype(float) % 100) // 30).astype(int).astype(str)


def _m24_str(df: pd.DataFrame) -> pd.Series:
    return ((((df["DepTime"].astype(float) // 100) % 24) * 60 + df["DepTime"].astype(float) % 100) // 24).astype(int).astype(str)


# train-fitted lookups (never fit on the dataframe passed to predict_proba)
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
_car_m24_train = train["UniqueCarrier"].astype(str) + "@" + _m24_str(train)
cm24_levels = pd.Index(sorted(_car_m24_train.unique()))
DIST_BINS = pd.qcut(train["Distance"], 8, retbins=True)[1]
_hhd_train = _hh_str(train) + "@" + pd.cut(train["Distance"], bins=DIST_BINS, include_lowest=True).astype(str)
hhd_levels = pd.Index(sorted(_hhd_train.unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(float)
    X["DepTime"] = dep
    X["dep_min"] = ((dep // 100) % 24) * 60 + dep % 100
    X["sin_t"] = np.sin(2 * np.pi * X["dep_min"] / 1440.0)
    X["cos_t"] = np.cos(2 * np.pi * X["dep_min"] / 1440.0)
    X["Distance"] = df["Distance"].astype(float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    X["car_m24"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "@" + _m24_str(df), categories=cm24_levels)
    _db = pd.cut(df["Distance"].astype(float), bins=DIST_BINS, include_lowest=True).astype(str)
    X["hh_dist"] = pd.Categorical(_hh_str(df) + "@" + _db, categories=hhd_levels)
    return X[FEATURES]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
models = [
    xgb.XGBClassifier(
        n_estimators=1500,
        max_depth=5,
        learning_rate=0.05,
        colsample_bytree=0.5,
        tree_method="hist",
        enable_categorical=True,
        random_state=s,
        n_jobs=N_JOBS,
    )
    for s in (1, 2, 3, 4, 5, 6, 7)
]

t0 = time.time()
_X_tr = prepare(train)
_y_tr = to_y(train)
for m in models:
    m.fit(_X_tr, _y_tr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
