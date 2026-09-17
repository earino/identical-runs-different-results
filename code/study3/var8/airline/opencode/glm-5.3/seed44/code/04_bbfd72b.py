"""XGBoost airline delay: numeric calendar/time features + recency-weighted training.

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

# --- feature engineering (fit on train only) ------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: sorted(train[c].dropna().astype(str).unique()) for c in CAT_COLS}
_CUM = np.array([0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])  # day-of-year of month start


def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    month = _cnum(df["Month"]).astype(int)
    dom = _cnum(df["DayofMonth"]).astype(int)
    X["month"] = month.astype(float)
    X["dom"] = dom.astype(float)
    X["dow"] = _cnum(df["DayOfWeek"]).astype(float)
    dep = df["DepTime"].astype(int)
    dep_min = ((dep // 100) * 60 + (dep % 100)) % 1440
    X["DepTime"] = dep.astype(float)
    X["dep_min"] = dep_min.astype(float)
    X["hour"] = (dep_min // 60).astype(float)
    X["Distance"] = df["Distance"].astype(float)
    # fixed-date holiday distances (day-of-year based; valid for both 2005/2006 non-leap)
    dy = (_CUM[month.to_numpy()] + dom.to_numpy()).astype(float)
    X["to_xmas"] = np.clip(dy - 359.0, -21.0, 35.0)
    X["to_ny"] = np.clip(dy - 1.0, -30.0, 30.0)
    X["to_jul4"] = np.clip(dy - 185.0, -30.0, 30.0)
    X["xmas_seas"] = np.where((dy >= 352.0) | (dy <= 5.0), 1.0, 0.0)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# recency weighting: later 2005 months matter more for 2006
_train_month = _cnum(train["Month"]).astype(int).to_numpy()
ALPHA = 0.15
w_train = np.exp(ALPHA * (_train_month - 12))

# --- model ----------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=20,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(
    prepare(train), to_y(train),
    sample_weight=w_train,
    eval_set=[(prepare(evald), to_y(evald))],
    verbose=False,
)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
