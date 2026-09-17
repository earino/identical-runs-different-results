"""XGBoost binary classifier for airline delay. THE ONLY FILE THE AGENT EDITS.

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

# --- features ----------------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
# interaction categoricals: airport/carrier x hour-of-day (delay risk is strongly time-of-day dependent)
INTERACTIONS = {
    "OriginHour": lambda d: d["Origin"].astype(str) + "_" + (d["DepTime"] // 100).astype(str),
    "CarrierHour": lambda d: d["UniqueCarrier"].astype(str) + "_" + (d["DepTime"] // 100).astype(str),
    "DestHour": lambda d: d["Dest"].astype(str) + "_" + (d["DepTime"] // 100).astype(str),
}
# half-hour resolution variants (finer time buckets)
HALF = lambda d: ((d["DepTime"] // 100) * 2 + (d["DepTime"] % 100) // 30).astype(str)
INTERACTIONS["OriginHalf"] = lambda d: d["Origin"].astype(str) + "_" + HALF(d)
INTERACTIONS["DestHalf"] = lambda d: d["Dest"].astype(str) + "_" + HALF(d)
interaction_levels = {name: pd.Index(sorted(fn(train).unique())) for name, fn in INTERACTIONS.items()}


def _int_c(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("c-", "", regex=False).astype(int)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["Month"] = _int_c(df["Month"])
    X["DayofMonth"] = _int_c(df["DayofMonth"])
    X["DayOfWeek"] = _int_c(df["DayOfWeek"])
    dep = df["DepTime"]
    X["DepTime"] = dep
    X["Hour"] = dep // 100
    X["Minute"] = dep % 100
    X["DepMin"] = (dep // 100) * 60 + dep % 100
    X["Distance"] = df["Distance"]
    X["dist_x_hour"] = df["Distance"] * (df["DepTime"] // 100)
    X["sin_depmin"] = np.sin(2 * np.pi * X["DepMin"] / 1440)
    X["cos_depmin"] = np.cos(2 * np.pi * X["DepMin"] / 1440)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])  # unseen -> NaN
    for name, fn in INTERACTIONS.items():
        X[name] = pd.Categorical(fn(df), categories=interaction_levels[name])  # unseen -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# sample weights: upweight the last three months of the training year (train=2005, eval/holdout=2006)
w_train = np.where(_int_c(train["Month"]).to_numpy() >= 10, 2.0, 1.0)

# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    reg_alpha=5,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), sample_weight=w_train)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
