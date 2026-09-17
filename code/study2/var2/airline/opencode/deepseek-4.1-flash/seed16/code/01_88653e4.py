"""XGBoost binary classifier for flight-departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives in prepare(), and all fitted statistics are derived from data/train.csv only.
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

NUM_COLS = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "Distance"]
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]

# Frequency maps fit on training data only. Unseen levels map to 0 in prepare().
freq_maps = {}
for c in CAT_COLS:
    freq_maps[c] = train[c].value_counts()
_train_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
freq_maps["route"] = _train_route.value_counts()
del _train_route


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    out = pd.DataFrame(index=df.index)
    out["DepTime"] = dep
    out["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    out["hour"] = (dep // 100).clip(0, 23)
    out["minute"] = dep % 100
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    for c in CAT_COLS:
        out[c + "_freq"] = df[c].map(freq_maps[c]).fillna(0).astype(np.float32)
    out["route_freq"] = route.map(freq_maps["route"]).fillna(0).astype(np.float32)
    return out


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=10,
    learning_rate=0.1,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
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
