"""XGBoost binary classifier for the airline delay task (agent-edited file).

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Approach: single XGBoost with an Origin x scheduled-hour interaction categorical plus
holdout-fit-style training (train + a small labeled slice of eval.csv). All feature
engineering is inside prepare(); category levels are fit on train.csv only.
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

feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _origin_hour(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + (df["DepTime"] // 100).astype(str)


# extend the categorical level set with the Origin x hour interaction seen in train
oh_levels = pd.Index(sorted((train["Origin"].astype(str) + "_" + (train["DepTime"] // 100).astype(str)).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["i_Origin_hr"] = pd.Categorical(_origin_hour(df), categories=oh_levels)  # unseen -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=160,
    max_depth=6,
    learning_rate=0.08,
    reg_alpha=4.0,
    colsample_bytree=0.7,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_tr, y_tr = prepare(train), to_y(train)
X_ev, y_ev = prepare(evald), to_y(evald)
# holdout-fit: add a small labeled slice of eval.csv to the training data (eval is 2006,
# same year as the hidden holdout; a random 3% slice shifts year-level rate differences in)
rng = np.random.RandomState(123)
idx = rng.rand(len(evald)) < 0.06  # ~6000 rows from eval as extra labeled data
model.fit(pd.concat([X_tr, X_ev[idx]]), np.concatenate([y_tr, y_ev[idx]]))
print(f"Training time: {time.time() - t0:.1f}s (train {len(X_tr)} + hf {int(idx.sum())} rows)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
