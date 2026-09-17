"""XGBoost ensemble for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of
     P(positive).  All feature engineering lives in prepare(), which predict_proba() calls, so the
     same transformation applies to the hidden holdout.  Encoders are fitted on train only.

Model: the average of 16 XGBoost classifiers over the tree-depth grid 1..16 (learning rate 0.16,
200 rounds, min_child_weight 0).  train is 2005 while eval and the hidden holdout are 2006, so any
single capacity / step-size setting is a year-transfer gamble; averaging over the whole grid is far
more robust than picking one point estimate off eval.csv.  See FINAL.md for the evidence.
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

# --- features -----------------------------------------------------------------
# Calendar columns (Month/DayofMonth/DayOfWeek) are dropped: ablating them cost nothing on the
# year-shifted eval set (0.7175 vs 0.7170), so they are noise here.  Airport identity, scheduled
# departure time and carrier are what carry signal; the frequency counts below summarise how busy an
# airport / carrier is and are fitted on train only (unseen keys -> 0).
FEATURES = ["DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
cat_cols = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(train[c])]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
freq_tables = {c: train[c].value_counts() for c in ("Origin", "Dest", "UniqueCarrier")}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["freq_origin"] = np.log1p(df["Origin"].map(freq_tables["Origin"]).fillna(0).to_numpy())
    X["freq_dest"] = np.log1p(df["Dest"].map(freq_tables["Dest"]).fillna(0).to_numpy())
    X["freq_carrier"] = np.log1p(df["UniqueCarrier"].map(freq_tables["UniqueCarrier"]).fillna(0).to_numpy())
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: averaged ensemble over the tree-depth grid -------------------------
COMMON = dict(learning_rate=0.16, n_estimators=200, min_child_weight=0,
              tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)

Xtr, ytr = prepare(train), to_y(train)

t0 = time.time()
models = []
for depth in range(1, 17):
    m = xgb.XGBClassifier(max_depth=depth, **COMMON)
    m.fit(Xtr, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
