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

RAW_COLS = [c for c in train.columns if c not in ID_COLS + [TARGET]]
OBJ_COLS = [c for c in RAW_COLS if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]

# --- fitted state (training data only) ----------------------------------------
# string/categorical columns -> integer codes; levels fixed from train
CAT_LEVELS = {}
for c in RAW_COLS:
    if c in OBJ_COLS:
        CAT_LEVELS[c] = pd.Index(sorted(train[c].dropna().astype(str).unique()))
# route levels are assembled in prepare() from origin x dest strings


def _to_num(s: pd.Series) -> pd.Series:
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")
    return pd.to_numeric(s, errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw airflow-style row -> model matrix. All feature engineering lives here."""
    X = pd.DataFrame(index=df.index)

    mon = _to_num(df["Month"])
    dom = _to_num(df["DayofMonth"])
    dow = _to_num(df["DayOfWeek"])
    X["month"] = mon
    X["dayofmonth"] = dom
    X["dayofweek"] = dow
    X["is_weekend"] = (dow >= 6).astype(float)

    dt = _to_num(df["DepTime"])
    hour = (dt // 100).clip(upper=23).fillna(-1)
    minute = (dt % 100).fillna(0)
    X["dep_hour"] = hour
    X["dep_minute"] = minute
    tod = hour * 60 + minute
    X["dep_tod"] = tod
    X["dep_tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["dep_tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)

    X["distance"] = pd.to_numeric(df["Distance"], errors="coerce")

    carrier = df["UniqueCarrier"].astype(str)
    origin = df["Origin"].astype(str)
    dest = df["Dest"].astype(str)
    X["carrier"] = pd.Categorical(carrier, categories=CAT_LEVELS["UniqueCarrier"])
    X["origin"] = pd.Categorical(origin, categories=CAT_LEVELS["Origin"])
    X["dest"] = pd.Categorical(dest, categories=CAT_LEVELS["Dest"])

    return X


CAT_COLS = ["carrier", "origin", "dest"]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")
print(f"Train AUC: {roc_auc_score(to_y(train), model.predict_proba(prepare(train))[:, 1]):.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
