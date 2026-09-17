"""XGBoost airline-delay classifier.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. Module-level `predict_proba(df)`: raw DataFrame (same columns as train.csv, target may be absent)
     -> 1-D numpy array of P(positive). All feature engineering lives in prepare().
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
def _cnum(s: pd.Series) -> pd.Series:
    """'c-7' -> 7 (NaN-safe)."""
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")

CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "route"]
_route_train = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
_cat_levels = {
    "UniqueCarrier": pd.Index(sorted(train["UniqueCarrier"].dropna().unique())),
    "Origin": pd.Index(sorted(train["Origin"].dropna().unique())),
    "Dest": pd.Index(sorted(train["Dest"].dropna().unique())),
    "route": pd.Index(sorted(_route_train.dropna().unique())),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    month = _cnum(df["Month"])
    day = _cnum(df["DayofMonth"])
    dow = _cnum(df["DayOfWeek"])
    X["month"] = month
    X["day"] = day
    X["dow"] = dow
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100) % 24
    minute = dep % 100
    tod = hour * 60 + minute
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440)

    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["distance"] = dist
    X["log_distance"] = np.log1p(dist)

    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"], categories=_cat_levels["UniqueCarrier"])
    X["Origin"] = pd.Categorical(df["Origin"], categories=_cat_levels["Origin"])
    X["Dest"] = pd.Categorical(df["Dest"], categories=_cat_levels["Dest"])
    X["route"] = pd.Categorical(route, categories=_cat_levels["route"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model ----------------------------------------------------------------------
X_all = prepare(train)
y_all = to_y(train)

rng = np.random.RandomState(SEED)
idx = rng.permutation(len(X_all))
n_val = 12000
val_idx, tr_idx = idx[:n_val], idx[n_val:]

t0 = time.time()
model = xgb.XGBClassifier(
    n_estimators=3000,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=60,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)
model.fit(X_all.iloc[tr_idx], y_all[tr_idx], eval_set=[(X_all.iloc[val_idx], y_all[val_idx])], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
