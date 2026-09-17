"""XGBoost binary classifier for the airline delay task.

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

# --- feature layout (fitted on train only) ------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
# native categoricals for ALL string columns (XGBoost hist handles them natively)
str_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
num_cols = [c for c in feature_cols if c not in str_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in str_cols}

N_MONTH = 12


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering lives here so predict_proba reproduces it on unseen data."""
    X = pd.DataFrame(index=df.index)
    for c in num_cols:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    dt = X["DepTime"]
    hh = (dt // 100).clip(0, 23)
    mm = (dt % 100).clip(0, 59)
    X["dep_hhmm_sin"] = np.sin(2 * np.pi * (hh * 60 + mm) / 1440.0)
    X["dep_hhmm_cos"] = np.cos(2 * np.pi * (hh * 60 + mm) / 1440.0)
    X["dep_hour"] = hh
    X["dep_minute"] = mm
    X["is_night"] = ((hh >= 21) | (hh <= 5)).astype(np.int8)
    X["is_redeye"] = ((hh >= 0) & (hh <= 4)).astype(np.int8)
    X["dep_valley"] = ((dt >= 600) & (dt <= 1900)).astype(np.int8)
    X["DepTime_missing"] = dt.isna().astype(np.int8)
    dom = pd.to_numeric(df["DayofMonth"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    X["day_sin"] = np.sin(2 * np.pi * (dom - 1) / 31.0)
    X["day_cos"] = np.cos(2 * np.pi * (dom - 1) / 31.0)
    mon = pd.to_numeric(df["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    X["month_sin"] = np.sin(2 * np.pi * mon / N_MONTH)
    X["month_cos"] = np.cos(2 * np.pi * mon / N_MONTH)
    for c in str_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=50,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
    early_stopping_rounds=50,
)

t0 = time.time()
X = prepare(train)
y = to_y(train)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
model = xgb.XGBClassifier(**PARAMS)
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
best_iter = int(getattr(model, "best_iteration", PARAMS["n_estimators"] - 1)) + 1
print(f"early-stop fit: best_iter={best_iter}  ({time.time() - t0:.1f}s)")

# refit on all of train with the chosen number of trees
t0 = time.time()
final_params = {k: v for k, v in PARAMS.items() if k not in ("n_estimators", "early_stopping_rounds")}
final_params["n_estimators"] = max(100, best_iter)
model = xgb.XGBClassifier(**final_params)
model.fit(X, y)
print(f"final refit ({time.time() - t0:.1f}s)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
