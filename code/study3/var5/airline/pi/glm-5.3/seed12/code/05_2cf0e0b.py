"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Approach: light feature engineering (DepTime parsing into hour/minute + cyclic, native categoricals for
carrier/origin/dest fit on train only) + an ensemble of deep, heavily column-subsampled XGBoost models.
The deep+bagged regime is far more robust to the 2005->2006 covariate shift than shallow boosted trees.
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

# --- feature engineering -------------------------------------------------------
# categorical levels fit on the training data only; unseen levels in future data -> NaN -> missing
carrier_levels = pd.Index(sorted(train["UniqueCarrier"].unique()))
origin_levels = pd.Index(sorted(train["Origin"].unique()))
dest_levels = pd.Index(sorted(train["Dest"].unique()))
# airport traffic (structural, stable across years) from the training slice only
origin_count = train["Origin"].value_counts()
dest_count = train["Dest"].value_counts()
origin_routes = train.groupby("Origin")["Dest"].nunique()
dest_routes = train.groupby("Dest")["Origin"].nunique()


def _num(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("c-", "", regex=False).astype(int)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["month"] = _num(df["Month"])
    X["day"] = _num(df["DayofMonth"])
    X["dow"] = _num(df["DayOfWeek"])
    dt = df["DepTime"].astype(int)
    hour = dt // 100
    mins = hour * 60 + dt % 100
    X["deptime"] = dt
    X["hour"] = hour
    X["mins"] = mins
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    X["bin15"] = mins // 15
    X["dist"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(X["dist"])
    X["carrier"] = pd.Categorical(df["UniqueCarrier"], categories=carrier_levels)
    X["origin"] = pd.Categorical(df["Origin"], categories=origin_levels)
    X["dest"] = pd.Categorical(df["Dest"], categories=dest_levels)
    X["oc"] = np.log(df["Origin"].map(origin_count).fillna(1))
    X["dc"] = np.log(df["Dest"].map(dest_count).fillna(1))
    X["o_hub"] = df["Origin"].map(origin_routes).fillna(1)
    X["d_hub"] = df["Dest"].map(dest_routes).fillna(1)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: ensemble of deep, column-subsampled XGB models ---------------------
N_MEMBERS = 10
PARAMS = dict(
    n_estimators=60,
    max_depth=24,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.35,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)

# recency weighting: later months of 2005 are temporally closer to the 2006 evaluation period
_w_month = _num(train["Month"]).to_numpy()
SAMPLE_W = 1.0 + 1.0 * (_w_month - 1) / 11.0

t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
members = []
for i in range(N_MEMBERS):
    m = xgb.XGBClassifier(random_state=SEED + i, **PARAMS)
    m.fit(Xtr, ytr, sample_weight=SAMPLE_W)
    members.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.zeros(len(X))
    for m in members:
        p += m.predict_proba(X)[:, 1]
    return p / len(members)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
