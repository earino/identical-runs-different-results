"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Key ideas (chosen by local screening):
  - Month/DayofMonth/DayOfWeek parsed to numeric ints (mixed encoding) + hour/minute from DepTime.
  - Origin/Dest/UniqueCarrier as native categoricals; Origin_Dest_hour route x hour categorical
    (stable congestion effect; categories from train only, unseen -> NaN).
  - Linear recency sample weights (recent 2005 months weigh more; eval/holdout are 2006).
  - Shallow (d=4), many slow (lr=0.05) trees with colsample 0.6-0.7.
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

# --- features (all stats fitted on train only) ----------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _route_hour(df: pd.DataFrame) -> pd.Series:
    return (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
            + "_" + (df["DepTime"] // 100).astype(str))


route_hour_levels = pd.Index(_route_hour(train).value_counts().index)  # frequency order


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype(float)
    X["Distance"] = df["Distance"].astype(float)
    X["Month"] = df["Month"].astype(str).str.slice(2).astype(float)
    X["DayofMonth"] = df["DayofMonth"].astype(str).str.slice(2).astype(float)
    X["DayOfWeek"] = df["DayOfWeek"].astype(str).str.slice(2).astype(float)
    X["hour"] = (df["DepTime"].astype(int) // 100).astype(float)
    X["minute"] = (df["DepTime"].astype(int) % 100).astype(float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["route_hour"] = pd.Categorical(_route_hour(df), categories=route_hour_levels)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# recency weights: linear ramp over months of 2005 (train slice)
_month = train["Month"].astype(str).str.slice(2).astype(float).to_numpy()
W_RECENCY = (0.5 + _month / 11.0)

# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=800,
    max_depth=4,
    learning_rate=0.03,
    min_child_weight=5,
    colsample_bytree=0.8,
    colsample_bynode=0.5,
    max_bin=512,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), sample_weight=W_RECENCY)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
