"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "route"]
cat_levels = {}
# route levels from training data only
_route_levels = pd.Index(sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique()))
for c in ["UniqueCarrier", "Origin", "Dest"]:
    cat_levels[c] = pd.Index(sorted(train[c].dropna().unique()))
cat_levels["route"] = _route_levels


def _unc(x: pd.Series) -> pd.Series:
    return x.astype(str).str.replace("c-", "", regex=False).astype(float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    month = _unc(df["Month"])
    dom = _unc(df["DayofMonth"])
    dow = _unc(df["DayOfWeek"])
    dep = df["DepTime"].astype(float)
    dep_h = np.floor(dep / 100.0)
    minutes = dep_h * 60.0 + (dep - dep_h * 100.0)

    X["month"] = month
    X["sin_month"] = np.sin(2 * np.pi * month / 12.0)
    X["cos_month"] = np.cos(2 * np.pi * month / 12.0)
    X["dom"] = dom
    X["sin_dom"] = np.sin(2 * np.pi * dom / 31.0)
    X["cos_dom"] = np.cos(2 * np.pi * dom / 31.0)
    X["dow"] = dow
    X["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    X["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)
    X["dep_minutes"] = minutes
    X["sin_min"] = np.sin(2 * np.pi * minutes / 1440.0)
    X["cos_min"] = np.cos(2 * np.pi * minutes / 1440.0)
    X["dep_hour"] = dep_h
    X["distance"] = df["Distance"].astype(float)
    X["log_distance"] = np.log1p(X["distance"])
    X["route"] = (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).values
    X["UniqueCarrier"] = df["UniqueCarrier"].values
    X["Origin"] = df["Origin"].values
    X["Dest"] = df["Dest"].values
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=8,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    min_child_weight=5.0,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_all = prepare(train)
y_all = to_y(train)
rng = np.random.RandomState(SEED)
val_idx = rng.rand(len(X_all)) < 0.1
X_tr, y_tr = X_all[~val_idx], y_all[~val_idx]
X_val, y_val = X_all[val_idx], y_all[val_idx]
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df), iteration_range=(0, model.best_iteration + 1))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
