"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
All feature engineering lives in prepare(), which predict_proba() calls on unseen rows. Counts/encoders are
fitted on train only.
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

RAW_CAT = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in RAW_CAT}


def _num_c(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace(r"[^0-9]", "", regex=True), errors="coerce")


def _dep(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["DepTime"], errors="coerce").astype("float64")


def _hour(df: pd.DataFrame) -> pd.Series:
    return np.floor(_dep(df) / 100.0)


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    K = pd.DataFrame(index=df.index)
    K["carrier"] = df["UniqueCarrier"].astype("string")
    K["origin"] = df["Origin"].astype("string")
    K["dest"] = df["Dest"].astype("string")
    K["route"] = K["origin"] + "_" + K["dest"]
    K["hour"] = _hour(df).astype("float64").astype("string")
    K["origin_hour"] = K["origin"] + "_" + K["hour"]
    K["route_hour"] = K["route"] + "_" + K["hour"]
    return K


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leak-free row-wise features (no fitted statistics)."""
    X = pd.DataFrame(index=df.index)
    dep = _dep(df)
    X["dep_hour"] = _hour(df)
    X["dep_min"] = dep % 100.0
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").astype("float64")
    X["month"] = _num_c(df["Month"])
    X["dom"] = _num_c(df["DayofMonth"])
    X["dow"] = _num_c(df["DayOfWeek"])
    for c in RAW_CAT:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


# --- frequency counts fitted on train only --------------------------------------
FREQ_COLS = ["carrier", "origin", "dest", "route", "origin_hour", "route_hour"]
_kt = _keys(train)
freq_maps = {c: _kt[c].value_counts() for c in FREQ_COLS}


def add_freq(X: pd.DataFrame, K: pd.DataFrame) -> pd.DataFrame:
    for c in FREQ_COLS:
        X["cnt_" + c] = np.log1p(K[c].map(freq_maps[c]).fillna(0.0).to_numpy(dtype="float64"))
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = base_features(df)
    K = _keys(df)
    for c in RAW_CAT:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return add_freq(X, K)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
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
