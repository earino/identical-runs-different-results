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
from sklearn.model_selection import train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature definitions (fit on TRAIN only; reused for any unseen dataframe) ------------------
RAW_CAT = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in RAW_CAT}
route_levels = pd.Index(sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique()))


def _cnum(s: pd.Series) -> pd.Series:
    """'c-7' -> 7 (Month/DayofMonth/DayOfWeek are encoded as c-<n> strings)."""
    return pd.to_numeric(s.astype(str).str.slice(start=2), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["Month"] = _cnum(df["Month"])
    X["DayofMonth"] = _cnum(df["DayofMonth"])
    X["DayOfWeek"] = _cnum(df["DayOfWeek"])
    X["hour"] = df["DepTime"] // 100
    X["minute"] = df["DepTime"] % 100
    X["DepTime"] = df["DepTime"]
    X["Distance"] = df["Distance"]
    X["log_dist"] = np.log1p(pd.to_numeric(df["Distance"], errors="coerce"))
    X["route"] = pd.Categorical(df["Origin"].astype(str) + "_" + df["Dest"].astype(str), categories=route_levels)
    for c in RAW_CAT:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------------------------
PARAMS = dict(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    max_depth=8,
    learning_rate=0.1,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    seed=SEED,
    nthread=N_JOBS,
)

t0 = time.time()
X_all = prepare(train)
y_all = to_y(train)
X_tr, X_va, y_tr, y_va = train_test_split(X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all)
dtr = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
dva = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
early = xgb.train(PARAMS, dtr, num_boost_round=2000, evals=[(dva, "val")], early_stopping_rounds=50, verbose_eval=False)
best_round = max(early.best_iteration, 50)
print(f"Best round (internal val): {best_round}, internal AUC {early.best_score:.4f}")

dall = xgb.DMatrix(X_all, label=y_all, enable_categorical=True)
model = xgb.train(PARAMS, dall, num_boost_round=best_round)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict(xgb.DMatrix(prepare(df), enable_categorical=True))


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
