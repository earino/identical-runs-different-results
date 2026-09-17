"""XGBoost binary classifier (airline delay). THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- feature definitions (fitted on TRAIN only, then applied by prepare()) ------
base_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in base_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
num_cols = [c for c in base_cols if c not in obj_cols]

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
# frequency encodings of the raw categorical columns
freq_cols = [c for c in obj_cols if c in ("UniqueCarrier", "Origin", "Dest")]
freq_maps = {c: train[c].value_counts().astype(float) for c in freq_cols}

WEEKEND = {"c-6", "c-7"}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").astype(float)
    # scheduled departure hhmm -> clock features
    t = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0.0)
    hour = (t // 100).clip(0, 23)
    minute = (t % 100).clip(0, 59)
    mod = hour * 60 + minute
    X["dep_hour"] = hour.astype(float)
    X["dep_minute"] = minute.astype(float)
    X["dep_sin"] = np.sin(2.0 * np.pi * mod / 1440.0)
    X["dep_cos"] = np.cos(2.0 * np.pi * mod / 1440.0)
    X["is_weekend"] = df["DayOfWeek"].isin(WEEKEND).astype(int)
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    for c in freq_cols:
        X[c + "_freq"] = df[c].map(freq_maps[c]).fillna(0.0).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=30,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
