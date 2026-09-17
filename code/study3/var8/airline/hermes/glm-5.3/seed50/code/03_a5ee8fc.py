"""XGBoost binary classifier: airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
obj_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]
            and (pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]))]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _num(s):  # 'c-7' -> 7
    return pd.to_numeric(s.astype(str).str.replace(r"^[a-zA-Z]-", "", regex=True), errors="coerce")


# --- smoothed target statistics fit on training data only ---------------------
_y = (train[TARGET] == POSITIVE).astype(float)
_global_mean = float(_y.mean())
SMOOTH_M = 20.0


def _stats_by(key_series: pd.Series) -> dict:
    g = pd.DataFrame({"k": key_series, "y": _y}).groupby("k")["y"].agg(["mean", "count"])
    return {k: (float(r["mean"]), float(r["count"])) for k, r in g.iterrows()}


carrier_stats = _stats_by(train["UniqueCarrier"])
origin_stats = _stats_by(train["Origin"])
dest_stats = _stats_by(train["Dest"])


def _smoothed(mapping: dict, key) -> float:
    if key in mapping:
        mean, n = mapping[key]
        return (mean * n + _global_mean * SMOOTH_M) / (n + SMOOTH_M)
    return _global_mean


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["MonthN"] = _num(df["Month"])
    X["DayN"] = _num(df["DayofMonth"])
    X["DowN"] = _num(df["DayOfWeek"])
    dep = df["DepTime"].astype(float)
    hh = (dep // 100) % 24
    mm = dep % 100
    mins = hh * 60 + mm
    X["DepHour"] = hh
    X["DepMinute"] = mm
    X["DepMins"] = mins
    X["DepSin"] = np.sin(2 * np.pi * mins / 1440.0)
    X["DepCos"] = np.cos(2 * np.pi * mins / 1440.0)
    X["Distance"] = df["Distance"].astype(float)
    X["CarrierRate"] = df["UniqueCarrier"].map(lambda k: _smoothed(carrier_stats, k))
    X["OriginRate"] = df["Origin"].map(lambda k: _smoothed(origin_stats, k))
    X["DestRate"] = df["Dest"].map(lambda k: _smoothed(dest_stats, k))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s, best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
