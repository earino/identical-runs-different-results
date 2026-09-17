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
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
DIST_BINS = np.unique(np.quantile(train["Distance"], np.linspace(0, 1, 11)))


def _to_num(s: pd.Series) -> pd.Series:
    """'c-<n>' -> <n> as float (NaN-safe)."""
    return pd.to_numeric(s.astype(str).str.slice(2), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # cyclical time features
    month = _to_num(df["Month"])
    dom = _to_num(df["DayofMonth"])
    dow = _to_num(df["DayOfWeek"])
    hour = np.floor(df["DepTime"] / 100.0)
    minute = df["DepTime"] % 100.0
    X["hour"] = hour
    X["minute"] = minute
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    X["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    X["dom_sin"] = np.sin(2 * np.pi * (dom - 1) / 31)
    X["dom_cos"] = np.cos(2 * np.pi * (dom - 1) / 31)
    X["dow_sin"] = np.sin(2 * np.pi * (dow - 1) / 7)
    X["dow_cos"] = np.cos(2 * np.pi * (dow - 1) / 7)
    X["log_dist"] = np.log1p(df["Distance"])
    X["hour_cat"] = pd.Categorical(pd.to_numeric(hour, errors="coerce").astype("Int64").astype(str), categories=[str(i) for i in range(24)])
    X["dist_bin"] = pd.Categorical(pd.cut(df["Distance"], bins=DIST_BINS, labels=False).astype("Int64").astype(str), categories=[str(i) for i in range(len(DIST_BINS) - 1)])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
BASE = dict(
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
SWEEP = [
    dict(max_depth=3, n_estimators=1200, learning_rate=0.05),
    dict(max_depth=4, n_estimators=800, learning_rate=0.05),
]

t0 = time.time()
X = prepare(train)
y = to_y(train)
Xe = prepare(evald)
ye = to_y(evald)
best_model, best_auc, best_cfg = None, -1.0, None
for cfg in SWEEP:
    t1 = time.time()
    m = xgb.XGBClassifier(**{**BASE, **cfg})
    m.fit(X, y)
    auc = roc_auc_score(ye, m.predict_proba(Xe)[:, 1])
    print(f"cfg={cfg} eval_auc={auc:.4f} ({time.time() - t1:.1f}s)")
    if auc > best_auc:
        best_model, best_auc, best_cfg = m, auc, cfg
model = best_model
print(f"best cfg: {best_cfg}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
