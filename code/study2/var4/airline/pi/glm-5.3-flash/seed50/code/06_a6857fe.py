"""XGBoost binary classifier on airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- features -----------------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _num_part(s: pd.Series) -> pd.Series:
    """c-<n> string -> int n."""
    return s.str[2:].astype(int)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    # --- scheduled departure time ---
    dt = df["DepTime"]
    valid = (dt <= 2359) & (dt % 100 <= 59)
    minutes = (dt // 100 * 60 + dt % 100).where(valid)  # invalid hhmm -> NaN
    X["dep_minutes"] = minutes
    X["dep_hour"] = minutes // 60
    X["dep_hour_cat"] = pd.Categorical((minutes // 60).astype("Int64"), categories=pd.Index(range(24)))
    X["dep_bin30"] = pd.Categorical((minutes // 30).astype("Int64"), categories=pd.Index(range(48)))
    X["dep_sin"] = np.sin(2 * np.pi * minutes / 1440)
    X["dep_cos"] = np.cos(2 * np.pi * minutes / 1440)
    X["red_eye"] = ((minutes < 360) | (minutes >= 1320)).astype(float)
    # --- calendar ---
    month = _num_part(df["Month"])
    dow = _num_part(df["DayOfWeek"])
    X["month"] = month
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    X["day"] = _num_part(df["DayofMonth"])
    X["dow"] = dow
    X["weekend"] = (dow >= 6).astype(float)
    # --- distance ---
    X["distance"] = df["Distance"]
    X["log_distance"] = np.log1p(df["Distance"])
    # --- categoricals (native xgb categorical; unseen levels -> NaN) ---
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)
EARLY_STOP = 50

t0 = time.time()
# internal split for early stopping (model selection only; final model refits on all rows)
X_tr, X_val, y_tr, y_val = train_test_split(
    prepare(train), to_y(train), test_size=0.2, random_state=SEED, stratify=to_y(train)
)
es_model = xgb.XGBClassifier(**PARAMS, early_stopping_rounds=EARLY_STOP)
es_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
best_iter = int(es_model.best_iteration) + 1
print(f"ES best iteration: {best_iter} (val auc {es_model.best_score:.5f}), {time.time()-t0:.0f}s")

seeds = [SEED, SEED + 7]
models = []
for s in seeds:
    m = xgb.XGBClassifier(**{**PARAMS, "n_estimators": best_iter, "random_state": s})
    m.fit(prepare(train), to_y(train))
    models.append(m)
model = models
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Ps = [m.predict_proba(prepare(df))[:, 1] for m in model]
    return np.mean(Ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
