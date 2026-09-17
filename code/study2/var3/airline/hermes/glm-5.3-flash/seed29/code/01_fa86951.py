"""XGBoost binary classifier: airline departure delay. Only file the agent edits.

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

# --- feature layout (fitted on train only) -------------------------------------
CAT_COLS = ["Month", "DayOfWeek", "DayofMonth", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _to_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    # DepTime: hhmm integer; values 1..99 are 00:xx (leading zero dropped); 2400+ roll past midnight
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    X["DepTime_hour"] = (dt % 2400) // 100
    X["DepTime_min"] = dt % 100
    X["DepTime_ge2400"] = (dt >= 2400).astype(float)
    # calendar as numbers (trees split them fine) plus categoricals for Origin/Dest/carrier
    X["Month_num"] = _to_int(df["Month"])
    X["Dom_num"] = _to_int(df["DayofMonth"])
    X["Dow_num"] = _to_int(df["DayOfWeek"])
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
Xev = prepare(evald)
model.fit(Xtr, ytr, eval_set=[(Xev, to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
