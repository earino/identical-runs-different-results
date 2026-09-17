"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives inside prepare(), which predict_proba() calls on unseen rows. Every lookup
     table / statistic used by prepare() is fitted on the training split only (never on the frame passed in).
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

feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
cat_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _code(s: pd.Series) -> pd.Series:
    """'c-4' -> 4 (the raw values are integer codes rendered as strings)."""
    return pd.to_numeric(s.astype(str).str.split("-").str[-1], errors="coerce")


def _key(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "route":
        return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    return df[name].astype(str)


# --- lookup tables fitted on the training split only ---------------------------
COUNT_KEYS = ["Origin", "Dest", "UniqueCarrier", "route"]
FREQ = {k: _key(train, k).value_counts() for k in COUNT_KEYS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    month = _code(df["Month"])
    dom = _code(df["DayofMonth"])
    dow = _code(df["DayOfWeek"])
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100) % 24
    minute = dep % 100
    X["month"] = month
    X["dom"] = dom
    X["dow"] = dow
    X["DepTime"] = dep
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = hour + minute / 60.0
    X["is_weekend"] = (dow >= 6).astype(float)
    X["redeye"] = ((dep >= 2300) | (dep < 500)).astype(float)
    X["hour_cat"] = pd.Categorical(hour.fillna(-1).astype(int), categories=list(range(24)))
    X["tod_30"] = pd.Categorical((X["tod"] .floordiv(0.5)).astype(int), categories=list(range(48)))
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["log_distance"] = np.log1p(X["Distance"])
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    for k in COUNT_KEYS:
        X[f"{k}_freq"] = np.log1p(_key(df, k).map(FREQ[k]))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
ytr = to_y(train)
Xtr_all = prepare(train)

model = xgb.XGBClassifier(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.02,
    min_child_weight=20,
    reg_lambda=5.0,
    subsample=0.8,
    colsample_bytree=0.7,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(Xtr_all, ytr)
print(f"Training time: {time.time() - t0:.1f}s")
print(f"Train AUC: {roc_auc_score(ytr, model.predict_proba(Xtr_all)[:, 1]):.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
yeval = to_y(evald)
p_eval = predict_proba(evald)
eval_auc = roc_auc_score(yeval, p_eval)
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")

# robustness diagnostics: a change worth keeping should help both random halves
half = np.random.RandomState(0).rand(len(yeval)) < 0.5
print(f"Eval half AUCs: {roc_auc_score(yeval[half], p_eval[half]):.4f} "
      f"{roc_auc_score(yeval[~half], p_eval[~half]):.4f}")
