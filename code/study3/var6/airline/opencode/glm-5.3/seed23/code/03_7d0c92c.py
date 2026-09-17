"""XGBoost binary classifier for airline departure delay prediction.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Approach: multi-view bagged ensemble. Each "view" is a different feature representation of the
same data; each view is fit by several shallow XGBoost models (row/col subsampling, seed diversity);
final prediction is the average across all views and models. Views deliberately disagree
(e.g. some views use only stable year-over-year features, some use target encodings) — the
average generalizes far better to the time-shifted holdout year than any single view.
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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
y_all = (train[TARGET] == POSITIVE).astype(int).to_numpy()
prior = float(y_all.mean())

# --- feature construction ------------------------------------------------------
ALL_C = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
BASE_C = ["DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in ALL_C}


def _te_map(series, m):
    """Smoothed target mean per level, computed on the TRAINING data only."""
    g = pd.DataFrame({"k": series, "y": y_all}).groupby("k")["y"].agg(["sum", "count"])
    return ((g["sum"] + prior * m) / (g["count"] + m)).to_dict()


_hour_tr = (train["DepTime"] // 100).clip(0, 24)
te_oh = _te_map(train["Origin"].astype(str) + "_" + _hour_tr.astype(str), 600)
te_dh = _te_map(train["Dest"].astype(str) + "_" + _hour_tr.astype(str), 600)
te_or = _te_map(train["Origin"], 800)
te_de = _te_map(train["Dest"], 800)
te_ch = _te_map(train["UniqueCarrier"] + "_" + _hour_tr.astype(str), 300)


def prepare(df: pd.DataFrame, view: str) -> pd.DataFrame:
    """All feature engineering lives here so predict_proba() reproduces it on unseen rows."""
    h = (df["DepTime"] // 100).clip(0, 24)
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype(int)
    X["Distance"] = df["Distance"].astype(float)
    cats = ALL_C if view == "md" else (["DayOfWeek", "UniqueCarrier"] if view == "no_od" else BASE_C)
    for c in cats:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    if view in ("oh", "no_od"):
        X["te_oh"] = (df["Origin"].astype(str) + "_" + h.astype(str)).map(te_oh).fillna(prior)
        X["te_dh"] = (df["Dest"].astype(str) + "_" + h.astype(str)).map(te_dh).fillna(prior)
    if view == "ot":
        X["te_or"] = df["Origin"].map(te_or).fillna(prior)
        X["te_de"] = df["Dest"].map(te_de).fillna(prior)
    if view == "ch":
        X["te_ch"] = (df["UniqueCarrier"] + "_" + h.astype(str)).map(te_ch).fillna(prior)
    return X


VIEWS = ["base", "oh", "ot", "ch", "md", "no_od"]

# --- ensemble ------------------------------------------------------------------
BASE = dict(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)
CONFIGS = [
    dict(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8),
    dict(n_estimators=150, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8),
    dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.7),
    dict(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=1.0, colsample_bytree=0.9),
]
N_SEEDS = 4

models = []  # (view, model) pairs
t0 = time.time()
for view in VIEWS:
    X_tr = prepare(train, view)
    for cfg in CONFIGS:
        for seed in range(N_SEEDS):
            m = xgb.XGBClassifier(random_state=seed, **cfg, **BASE)
            m.fit(X_tr, y_all)
            models.append((view, m))
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X_by_view = {v: prepare(df, v) for v in VIEWS}
    out = None
    for view, m in models:
        p = m.predict_proba(X_by_view[view])[:, 1]
        out = p if out is None else out + p
    return out / len(models)


t0 = time.time()
eval_auc = roc_auc_score((evald[TARGET] == POSITIVE).astype(int).to_numpy(), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
