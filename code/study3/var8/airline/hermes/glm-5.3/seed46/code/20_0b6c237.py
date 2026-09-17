"""XGBoost binary classifier — airline dep_delayed_15min.

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
# all six string columns (Month, DayofMonth, DayOfWeek, UniqueCarrier, Origin, Dest) are true
# categoricals; XGBoost >=1.2 handles them natively, so keep the high-cardinality ones too.
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
num_cols = [c for c in feature_cols if c not in obj_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in obj_cols}


def _add_features(X: pd.DataFrame) -> pd.DataFrame:
    # cyclical hour-of-day: flight delays have a strong daily pattern; sin/cos let the trees
    # split time-of-day smoothly even with a coarse feature.
    dep = X["DepTime"]
    hour = (dep // 100).astype(float)
    X["DepHour"] = hour
    X["DepHour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    X["DepHour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    X["DepMinute"] = dep % 100
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    _add_features(X)
    for c in obj_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# ensemble of deep XGBoost models with different seeds; the averaged probability rank
# generalizes better to the shifted (2006) holdout than any single model.
# best_iteration was ~273-300 with AUC early stopping, so train a fixed 300 rounds and
# skip the per-round eval_set overhead -> 3 models fit inside the time cap.
N_MODELS = 3
N_ROUNDS = 300


def _make_model(seed: int, colsample: float, depth: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=N_ROUNDS,
        max_depth=depth,
        learning_rate=0.012,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=colsample,
        reg_lambda=1.0,
        gamma=0.1,
        tree_method="hist",
        max_cat_to_onehot=32,
        max_bin=512,
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


t0 = time.time()
models = []
for i, (cs, d) in enumerate([(0.8, 26), (0.7, 34)]):
    m = _make_model(SEED + 10 * i, cs, d)
    m.fit(prepare(train), to_y(train))
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
