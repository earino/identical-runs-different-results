"""XGBoost binary classifier for the airline delay task. THE ONLY FILE THE AGENT EDITS.

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
carrier_levels = pd.Index(sorted(train["UniqueCarrier"].dropna().unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    t = pd.to_numeric(df["DepTime"], errors="coerce")
    X["carrier_hour"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "_" + (t // 100).astype("Int64").astype(str),
        categories=[f"{c}_{h}" for c in carrier_levels for h in range(25)],
    )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
N_ESTIMATORS = 4000
model = xgb.XGBClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    subsample=0.9,
    colsample_bytree=0.9,
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
# internal early-stopping holdout carved from train (NOT eval.csv, to avoid tuning on eval)
rng = np.random.RandomState(SEED)
val_idx = rng.rand(len(train)) < 0.15
Xtr = prepare(train.loc[~val_idx])
ytr = to_y(train.loc[~val_idx])
Xva = prepare(train.loc[val_idx])
yva = to_y(train.loc[val_idx])
model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    m = model
    return m.predict_proba(prepare(df), iteration_range=(0, m.best_iteration + 1))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
