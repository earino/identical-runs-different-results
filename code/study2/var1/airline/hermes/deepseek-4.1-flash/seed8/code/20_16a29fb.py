"""XGBoost ensemble for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Model: averaged ensemble of XGBoost classifiers over a wide capacity grid (tree depths 1-16,
learning rate 0.16, min_child_weight 1).  train is 2005 and eval/holdout are 2006, so any single
capacity or step-size setting is a year-transfer gamble; averaging over the grid is the robust choice
(see FINAL.md for the ablation evidence behind each setting).
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
# Calendar columns (Month/DayofMonth/DayOfWeek) are dropped: ablation showed they contribute nothing on
# the year-shifted eval set.  Airport identity + scheduled time + carrier carry the signal.
FEATURES = ["DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
cat_cols = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(train[c])]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
freq_tables = {c: train[c].value_counts() for c in ("Origin", "Dest", "UniqueCarrier")}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows.
    X = df[FEATURES].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["freq_origin"] = np.log1p(df["Origin"].map(freq_tables["Origin"]).fillna(0).to_numpy())
    X["freq_dest"] = np.log1p(df["Dest"].map(freq_tables["Dest"]).fillna(0).to_numpy())
    X["freq_carrier"] = np.log1p(df["UniqueCarrier"].map(freq_tables["UniqueCarrier"]).fillna(0).to_numpy())
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
DEPTHS = range(1, 17)
COMMON = dict(learning_rate=0.16, n_estimators=300, min_child_weight=1,
              tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)

Xtr, ytr = prepare(train), to_y(train)

t0 = time.time()
models = [xgb.XGBClassifier(max_depth=d, **COMMON) for d in DEPTHS]
for m in models:
    m.fit(Xtr, ytr)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
