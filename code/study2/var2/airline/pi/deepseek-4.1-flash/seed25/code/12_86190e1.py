"""XGBoost binary classifier for airline departure delay. Only file the agent edits.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives in prepare() and all fitted state comes from train only.
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

# --- fitted state (computed from train ONLY) ---------------------------------
DATE_CAT = ["DayOfWeek"]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
HOUR_LEVELS = pd.Index([str(i) for i in range(24)])
cat_levels = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in DATE_CAT + CAT_COLS}


def _ordinal(s: pd.Series) -> pd.Series:
    """'c-4' -> 4.0 ; numeric passes through."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _hour(df: pd.DataFrame) -> pd.Series:
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    return (dep // 100).clip(0, 23).astype(int)


# airport x scheduled-hour traffic volume (no labels involved)
_tr_hour = _hour(train)
_tr_org_hour = (train["Origin"].astype(str) + "|" + _tr_hour.astype(str)).value_counts()
_tr_dst_hour = (train["Dest"].astype(str) + "|" + _tr_hour.astype(str)).value_counts()
# origin traffic in the neighbouring hours
_tr_org_hour_prev = (train["Origin"].astype(str) + "|" + ((_tr_hour - 1) % 24).astype(str)).value_counts()
_tr_org_hour_next = (train["Origin"].astype(str) + "|" + ((_tr_hour + 1) % 24).astype(str)).value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    X["DepTime"] = dep
    X["DepMin"] = dep % 100
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["Month"] = _ordinal(df["Month"])
    X["DayofMonth"] = _ordinal(df["DayofMonth"])
    X["DayOfWeek"] = pd.Categorical(df["DayOfWeek"].astype(str), categories=cat_levels["DayOfWeek"])
    hour = _hour(df)
    X["DepHourCat"] = pd.Categorical(hour.astype(str), categories=HOUR_LEVELS)
    org = df["Origin"].astype(str)
    dst = df["Dest"].astype(str)
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"].astype(str), categories=cat_levels["UniqueCarrier"])
    X["Origin"] = pd.Categorical(org, categories=cat_levels["Origin"])
    X["Dest"] = pd.Categorical(dst, categories=cat_levels["Dest"])
    X["org_hour_traffic"] = (org + "|" + hour.astype(str)).map(_tr_org_hour).fillna(0.0)
    X["dst_hour_traffic"] = (dst + "|" + hour.astype(str)).map(_tr_dst_hour).fillna(0.0)
    X["org_hour_prev"] = (org + "|" + ((hour - 1) % 24).astype(str)).map(_tr_org_hour_prev).fillna(0.0)
    X["org_hour_next"] = (org + "|" + ((hour + 1) % 24).astype(str)).map(_tr_org_hour_next).fillna(0.0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# small ensemble of deep trees at different depths (variance reduction)
DEPTHS = [14, 16, 18]
models = [
    xgb.XGBClassifier(
        n_estimators=400,
        max_depth=d,
        learning_rate=0.02,
        subsample=0.9,
        colsample_bytree=0.6,
        min_child_weight=0,
        reg_lambda=0.0,
        reg_alpha=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + i,
        n_jobs=N_JOBS,
    )
    for i, d in enumerate(DEPTHS)
]

t0 = time.time()
Xtr = prepare(train)
ytr = to_y(train)
for m in models:
    m.fit(Xtr, ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
