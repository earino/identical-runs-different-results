"""XGBoost bagged ensemble for airline delay prediction. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- feature definition -------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols
            if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# frequency features: traffic volume per carrier / airport / route (stable across years)
COUNT_COLS = ["UniqueCarrier", "Origin", "Dest", "Route"]


def _count_key(df: pd.DataFrame, c: str) -> pd.Series:
    if c == "Route":
        return df["Origin"].astype("string") + "_" + df["Dest"].astype("string")
    return df[c].astype("string")


FREQ = {c: _count_key(train, c).value_counts().to_dict() for c in COUNT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dep = df["DepTime"].to_numpy()
    X["DepHour"] = (dep // 100).astype(float)
    X["DepMin"] = (dep % 100).astype(float)
    for c in COUNT_COLS:
        X["cnt_" + c] = _count_key(df, c).map(FREQ[c]).fillna(0).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: bagged ensemble of shallow, decorrelated trees --------------------
X_train = prepare(train)
y_train = to_y(train)
N_MODELS = 20

models = []
for i in range(N_MODELS):
    m = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=5 + (i % 3),
        learning_rate=0.05,
        subsample=0.7 + 0.05 * (i % 4),
        colsample_bytree=0.6 + 0.1 * (i % 3),
        min_child_weight=3 + (i % 4),
        reg_lambda=1.0 + (i % 3),
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + 1000 * i,
        n_jobs=N_JOBS,
    )
    t0 = time.time()
    m.fit(X_train, y_train)
    models.append(m)
    print(f"model {i} trained in {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.zeros(len(X))
    for m in models:
        p += m.predict_proba(X)[:, 1]
    return p / len(models)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
