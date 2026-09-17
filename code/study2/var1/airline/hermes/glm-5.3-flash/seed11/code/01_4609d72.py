"""XGBoost airline delay classifier (agent-edited). See program.md for the contract.

predict_proba(df) -> P(dep_delayed_15min == 'Y'). All feature engineering lives in prepare(df)
so the hidden holdout gets identical treatment. Encoders/statistics are fitted on train only.
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
# frozen category levels, fitted on TRAIN only (unseen levels -> NaN in predict_proba)
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def add_feats(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise engineered features; uses only values present in the row's dataframe."""
    dt = X["DepTime"].astype(int)
    # DepTime is hhmm (0..2400); rows < 100 look like plain minutes (1..99) -> normalize to hour
    hour = np.where(dt < 100, dt // 10, dt // 100).astype(float)
    hour = np.clip(hour, 0, 24)
    rad = 2 * np.pi * hour / 24.0
    X = X.assign(
        dep_hour=hour,
        dep_min=dt % 100,
        dep_sin=np.sin(rad),
        dep_cos=np.cos(rad),
    )
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    X = add_feats(X)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

# time-ordered validation slice: last 15% of the 2005 train file (eval is 2006)
n = len(train)
cut = int(n * 0.85)
tr, va = train.iloc[:cut], train.iloc[cut:]
y_tr, y_va = to_y(tr), to_y(va)

t0 = time.time()
model.fit(
    prepare(tr),
    y_tr,
    eval_set=[(prepare(va), y_va)],
    verbose=False,
)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
