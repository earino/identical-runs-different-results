"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

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
cat_levels = {c: pd.Index(sorted(train[c].unique())) for c in CAT_COLS}

# target-encoded keys: (feature name, grouping columns)
TE_KEYS = [
    ("teOrigin", ["Origin"]),
    ("teDest", ["Dest"]),
    ("teRoute", ["Origin", "Dest"]),
    ("teCarrier", ["UniqueCarrier"]),
    ("teMonth", ["Month"]),
]
_SMOOTH = 20.0
_y = (train[TARGET] == POSITIVE).astype(float)
_PRIOR = float(_y.mean())


def _fit_maps(df: pd.DataFrame, y: pd.Series) -> dict:
    maps = {}
    for name, keys in TE_KEYS:
        t = df[keys].assign(_y=y.values)
        g = t.groupby(keys, observed=True)["_y"].agg(["sum", "count"])
        maps[name] = {
            k if len(keys) > 1 else k[0]: v
            for k, v in (((g["sum"] + _SMOOTH * _PRIOR) / (g["count"] + _SMOOTH)).items())
        }
    return maps


_MAPS = _fit_maps(train, _y)


def _te(df: pd.DataFrame, name: str, keys: list) -> pd.Series:
    m = _MAPS[name]
    if len(keys) == 1:
        return df[keys[0]].map(m)
    key = list(zip(df[keys[0]], df[keys[1]]))
    return pd.Series([m.get(k, np.nan) for k in key], index=df.index)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[CAT_COLS + ["DepTime", "Distance"]].copy()
    for name, keys in TE_KEYS:
        X[name] = _te(df, name, keys).fillna(_PRIOR).astype(float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
