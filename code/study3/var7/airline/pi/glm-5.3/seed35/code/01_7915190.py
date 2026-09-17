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

# --- features -----------------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Route"]
cat_levels = {}

for df in (train,):  # fit statistics on training data only
    pass
_route_levels = pd.Index(sorted((train["Origin"] + "-" + train["Dest"]).unique()))
_carrier_levels = pd.Index(sorted(train["UniqueCarrier"].unique()))
_origin_levels = pd.Index(sorted(train["Origin"].unique()))
_dest_levels = pd.Index(sorted(train["Dest"].unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    # calendar: c-N strings -> integers
    X["month"] = df["Month"].str.replace("c-", "", regex=False).astype(int)
    X["day"] = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int)
    X["dow"] = df["DayOfWeek"].str.replace("c-", "", regex=False).astype(int)
    # scheduled departure time -> hour/minute/minutes since midnight
    dep = df["DepTime"].astype(int)
    X["dep_hour"] = dep // 100
    X["dep_min"] = dep % 100
    X["dep_ssm"] = X["dep_hour"] * 60 + X["dep_min"]
    X["dep_sin"] = np.sin(2 * np.pi * X["dep_ssm"] / 1440)
    X["dep_cos"] = np.cos(2 * np.pi * X["dep_ssm"] / 1440)
    X["distance"] = df["Distance"].astype(float)
    X["dist_log"] = np.log1p(X["distance"])
    # categoricals
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"], categories=_carrier_levels)
    X["Origin"] = pd.Categorical(df["Origin"], categories=_origin_levels)
    X["Dest"] = pd.Categorical(df["Dest"], categories=_dest_levels)
    X["Route"] = pd.Categorical(df["Origin"] + "-" + df["Dest"], categories=_route_levels)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
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
