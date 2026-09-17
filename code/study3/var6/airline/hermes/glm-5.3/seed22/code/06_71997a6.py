"""XGBoost binary classifier for airline delay prediction.

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


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    # compact numeric encodings of the calendar columns (raw c-<n> strings are redundant)
    X["month"] = df["Month"].str.replace("c-", "", regex=False).astype(int)
    X["day"] = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int)
    X["dow"] = df["DayOfWeek"].str.replace("c-", "", regex=False).astype(int)
    X["dep_time"] = df["DepTime"].astype(int)
    X["hour"] = X["dep_time"] // 100
    X["minute"] = X["dep_time"] % 100
    # cyclic versions of month/dow (small domains; let trees see the wrap-around too)
    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12.0)
    X["dow_sin"] = np.sin(2 * np.pi * X["dow"] / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * X["dow"] / 7.0)
    X["distance"] = df["Distance"].astype(float)
    X["dist_log"] = np.log1p(X["distance"].clip(lower=0))
    # nominal categoricals, native XGBoost handling
    X["carrier"] = pd.Categorical(df["UniqueCarrier"], categories=carrier_levels)
    X["origin"] = pd.Categorical(df["Origin"], categories=origin_levels)
    X["dest"] = pd.Categorical(df["Dest"], categories=dest_levels)
    return X


# categorical levels from TRAIN only (module-level, before prepare is used)
carrier_levels = pd.Index(sorted(train["UniqueCarrier"].dropna().unique()))
origin_levels = pd.Index(sorted(train["Origin"].dropna().unique()))
dest_levels = pd.Index(sorted(train["Dest"].dropna().unique()))


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
models = []
t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
for seed in (42, 7, 123):
    m = xgb.XGBClassifier(
        n_estimators=4000,
        max_depth=10,
        learning_rate=0.03,
        tree_method="hist",
        enable_categorical=True,
        min_child_weight=2,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        early_stopping_rounds=150,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    ps = [m.predict_proba(prepare(df))[:, 1] for m in models]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
