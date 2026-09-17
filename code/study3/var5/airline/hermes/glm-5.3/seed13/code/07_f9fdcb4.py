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
# Baseline feature set. Diagnostics showed added TE/route/time features overfit 2005 -> 2006,
# so we keep the raw columns and spend complexity on an ensemble of colsample-jittered models.
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
# traffic counts fit on train only (2005); used by prepare() on unseen rows
route_full = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
route_cnt = route_full.value_counts().to_dict()
origin_cnt = train["Origin"].value_counts().to_dict()
dest_cnt = train["Dest"].value_counts().to_dict()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    def _num(col):
        return pd.to_numeric(df[col].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    X["Month"] = _num("Month")
    X["DayofMonth"] = _num("DayofMonth")
    X["DayOfWeek"] = _num("DayOfWeek")
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"], categories=cat_levels["UniqueCarrier"])
    X["Origin"] = pd.Categorical(df["Origin"], categories=cat_levels["Origin"])
    X["Dest"] = pd.Categorical(df["Dest"], categories=cat_levels["Dest"])
    dt = X["DepTime"].fillna(0)
    X["hour"] = (dt // 100) % 24
    X["min_of_day"] = ((dt // 100) % 24) * 60 + (dt % 100)
    mod = X["min_of_day"].astype(float)
    X["sin_day"] = np.sin(2 * np.pi * mod / 1440.0)
    X["cos_day"] = np.cos(2 * np.pi * mod / 1440.0)
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["route_cnt"] = route.map(route_cnt).fillna(0)
    X["origin_cnt"] = df["Origin"].map(origin_cnt).fillna(0)
    X["dest_cnt"] = df["Dest"].map(dest_cnt).fillna(0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: 10-member colsample ensemble ---------------------------------------
X_train = prepare(train)
y_train = to_y(train)

BASE_PARAMS = dict(
    n_estimators=30,
    max_depth=24,
    learning_rate=0.1,
    colsample_bytree=0.7,
    colsample_bynode=0.8,
    reg_alpha=1.0,
    gamma=0.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
N_MEMBERS = 10

t0 = time.time()
members = []
for s in range(N_MEMBERS):
    m = xgb.XGBClassifier(random_state=s, **BASE_PARAMS)
    m.fit(X_train, y_train)
    members.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in members], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
