"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- feature engineering (fitted on TRAIN only, applied inside prepare) --------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
y_train = (train[TARGET] == POSITIVE).astype(int)
GLOBAL_MEAN = float(y_train.mean())


def _add_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic per-row features (no fitted state)."""
    X = df[["DepTime", "Distance"] + CAT_COLS].copy()
    dt = df["DepTime"].astype(int)
    X["hour"] = dt // 100
    X["minute"] = dt % 100
    X["depm_frac"] = (X["hour"] * 60 + X["minute"]) / 1440.0
    X["redeye"] = ((dt < 600) | (dt >= 2100)).astype(int)
    X["month_n"] = df["Month"].str[2:].astype(int)
    X["dow_n"] = df["DayOfWeek"].str[2:].astype(int)
    X["dom_n"] = df["DayofMonth"].str[2:].astype(int)
    return X


# smoothed target encodings, fitted on train only
def _te_map(keys: pd.Series) -> dict:
    g = pd.DataFrame({"k": keys, "y": y_train}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] / (g["count"] + 25) + GLOBAL_MEAN * 25 / (g["count"] + 25)).to_dict()


def _h3(df: pd.DataFrame) -> pd.Series:
    return (df["DepTime"].astype(int) // 300).astype(str)


train_route = train["Origin"] + "-" + train["Dest"]
TE_ORIGIN = _te_map(train["Origin"])
TE_DEST = _te_map(train["Dest"])
TE_CARRIER = _te_map(train["UniqueCarrier"])
TE_ROUTE = _te_map(train_route)
TE_O_H3 = _te_map(train["Origin"] + "|" + _h3(train))
TE_D_H3 = _te_map(train["Dest"] + "|" + _h3(train))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = _add_basic(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    X["te_origin"] = df["Origin"].map(TE_ORIGIN).fillna(GLOBAL_MEAN)
    X["te_dest"] = df["Dest"].map(TE_DEST).fillna(GLOBAL_MEAN)
    X["te_carrier"] = df["UniqueCarrier"].map(TE_CARRIER).fillna(GLOBAL_MEAN)
    X["te_route"] = (df["Origin"] + "-" + df["Dest"]).map(TE_ROUTE).fillna(GLOBAL_MEAN)
    X["te_o_h3"] = (df["Origin"] + "|" + _h3(df)).map(TE_O_H3).fillna(GLOBAL_MEAN)
    X["te_d_h3"] = (df["Dest"] + "|" + _h3(df)).map(TE_D_H3).fillna(GLOBAL_MEAN)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X = prepare(train)
y = to_y(train)
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

PARAMS = dict(
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
)

t0 = time.time()
es_model = xgb.XGBClassifier(
    n_estimators=3000,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
    early_stopping_rounds=50,
    **PARAMS,
)
es_model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
best_n = int(es_model.best_iteration) + 1
print(f"ES fit: {time.time() - t0:.1f}s, best_iteration={best_n}, va_auc={es_model.best_score:.4f}")

model = xgb.XGBClassifier(
    n_estimators=best_n,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    **PARAMS,
)

t0 = time.time()
model.fit(X, y)
print(f"Final fit ({best_n} trees): {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
