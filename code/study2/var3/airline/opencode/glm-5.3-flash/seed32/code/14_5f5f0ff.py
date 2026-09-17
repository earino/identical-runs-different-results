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
    hb = pd.to_numeric(hour, errors="coerce").astype("Int64").astype(str)
    db = pd.cut(df["Distance"], bins=DIST_BINS, labels=False).astype("Int64").astype(str)
    X["hour_cat"] = pd.Categorical(hb, categories=[str(i) for i in range(24)])
    X["dist_bin"] = pd.Categorical(db, categories=[str(i) for i in range(len(DIST_BINS) - 1)])
    inter = (hb + "_" + db).where(hb.notna() & db.notna())
    X["hour_x_dist"] = pd.Categorical(inter, categories=[f"{h}_{d}" for h in range(24) for d in range(len(DIST_BINS) - 1)])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
BASE = dict(
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    max_bin=512,
    random_state=SEED,
    n_jobs=N_JOBS,
)
SWEEP = [
    dict(max_depth=4, n_estimators=800, learning_rate=0.05),
    dict(max_depth=3, n_estimators=1200, learning_rate=0.05),
    dict(max_depth=2, n_estimators=1200, learning_rate=0.05),
    dict(max_depth=5, n_estimators=600, learning_rate=0.05),
    dict(max_depth=5, n_estimators=1200, learning_rate=0.03),
    dict(max_depth=6, n_estimators=200, learning_rate=0.1),
    dict(max_depth=4, n_estimators=1000, learning_rate=0.03),
    dict(max_depth=3, n_estimators=2000, learning_rate=0.03),
    dict(max_depth=2, n_estimators=2000, learning_rate=0.05),
    dict(max_depth=7, n_estimators=300, learning_rate=0.05),
    dict(max_depth=5, n_estimators=800, learning_rate=0.05),
    dict(max_depth=3, n_estimators=800, learning_rate=0.1),
    dict(grow_policy="lossguide", max_leaves=24, n_estimators=800, learning_rate=0.05),
    dict(grow_policy="lossguide", max_leaves=64, n_estimators=800, learning_rate=0.05),
    dict(grow_policy="lossguide", max_leaves=128, n_estimators=800, learning_rate=0.05),
]

t0 = time.time()
X = prepare(train)
y = to_y(train)
Xe = prepare(evald)
ye = to_y(evald)
ENS_MODELS = []
proba_sum = np.zeros(len(ye))
for cfg in SWEEP:
    t1 = time.time()
    m = xgb.XGBClassifier(**{**BASE, **cfg})
    m.fit(X, y)
    p = m.predict_proba(Xe)[:, 1]
    auc = roc_auc_score(ye, p)
    proba_sum += p
    ENS_MODELS.append(m)
    print(f"cfg={cfg} eval_auc={auc:.4f} ({time.time() - t1:.1f}s)")
ens_auc = roc_auc_score(ye, proba_sum / len(SWEEP))
print(f"ensemble eval_auc={ens_auc:.4f}")
model = ENS_MODELS[0]
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in ENS_MODELS], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
