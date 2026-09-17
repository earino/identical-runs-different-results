"""XGBoost binary classifier for airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

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
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# baseline: treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # time-of-day features from DepTime (hhmm)
    dep = _to_int(df["DepTime"])
    hour = (dep // 100).astype(float)
    minute = (dep % 100).astype(float)
    dep_min = hour * 60 + minute                      # raw minutes since midnight (values >= 2400 stay extreme)
    dep_min_mod = dep_min % 1440                      # true time-of-day for next-day slots
    X["dep_hour"] = hour
    X["dep_min"] = dep_min
    X["dep_min_mod"] = dep_min_mod
    ang = 2 * np.pi * dep_min_mod / 1440.0
    X["dep_sin"] = np.sin(ang)
    X["dep_cos"] = np.cos(ang)
    # cyclical day-of-week / day-of-year-ish seasonality
    dow = _to_int(df["DayOfWeek"].str.replace("c-", "", regex=False))
    mon = _to_int(df["Month"].str.replace("c-", "", regex=False))
    dom = _to_int(df["DayofMonth"].str.replace("c-", "", regex=False))
    doy = (mon - 1) * 31 + dom
    for name, v, period in (("dow", dow, 7.0), ("doy", doy, 372.0)):
        a = 2 * np.pi * v / period
        X[f"{name}_sin"] = np.sin(a)
        X[f"{name}_cos"] = np.cos(a)
    X["distance_log"] = np.log1p(_to_int(df["Distance"]))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    max_depth=6,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_all, y_all = prepare(train), to_y(train)
# internal split for early stopping (keeps eval.csv untouched by model selection)
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(X_all))
cut = int(0.8 * len(X_all))
tr_idx, va_idx = idx[:cut], idx[cut:]

es_model = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=50, **PARAMS)
es_model.fit(X_all.iloc[tr_idx], y_all[tr_idx], eval_set=[(X_all.iloc[va_idx], y_all[va_idx])], verbose=False)
best_n = int(es_model.best_iteration) + 1
print(f"best rounds: {best_n}")

model = xgb.XGBClassifier(n_estimators=best_n, **PARAMS)
model.fit(X_all, y_all)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
