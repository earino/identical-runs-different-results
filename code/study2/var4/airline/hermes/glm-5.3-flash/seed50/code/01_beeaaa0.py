"""XGBoost binary classifier for airline delays. THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

v2: base categoricals (train-fitted levels) + Origin/Carrier/Dest x dep-hour interaction
categoricals (train-fitted levels) + DepTime/Distance numerics. Regularized shallow-ish
trees (depth 5, min_child_weight 20, lambda 50, 150 trees) — heavy regularization helps
because train is 2005 and eval is 2006 (concept drift).
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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]

# encoders fit on TRAIN ONLY ----------------------------------------------------
LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _hour(df: pd.DataFrame) -> pd.Series:
    return (df["DepTime"] // 100).clip(0, 24).astype(int)


def _inter_levels(cat_tr: pd.Series, num_tr: pd.Series) -> pd.Index:
    return pd.Index(sorted((cat_tr.astype(str) + "_" + num_tr.astype(str)).unique()))


_htr = _hour(train)
LEV_OH = _inter_levels(train["Origin"], _htr)
LEV_CH = _inter_levels(train["UniqueCarrier"], _htr)
LEV_DH = _inter_levels(train["Dest"], _htr)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows.
    # Categoricals use train-fitted levels; unseen levels become NaN automatically.
    X = pd.DataFrame(index=df.index)
    h = _hour(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=LEVELS[c])
    X["Orig_hour"] = pd.Categorical(df["Origin"].astype(str) + "_" + h.astype(str), categories=LEV_OH)
    X["Car_hour"] = pd.Categorical(df["UniqueCarrier"].astype(str) + "_" + h.astype(str), categories=LEV_CH)
    X["Dest_hour"] = pd.Categorical(df["Dest"].astype(str) + "_" + h.astype(str), categories=LEV_DH)
    X["DepTime"] = df["DepTime"].astype(float)
    X["Distance"] = df["Distance"].astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    min_child_weight=20,
    reg_lambda=50.0,
    tree_method="hist",
    enable_categorical=True,
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
