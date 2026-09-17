"""XGBoost binary classifier for airline departure delay.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

All feature engineering lives in prepare(); it is fit on training data only (cat_levels built from
`train`), so predict_proba() reproduces it on unseen rows. Early stopping uses eval.csv (2006, same
period as the hidden holdout) just to pick the number of trees.
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
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "route"]


def base_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raw columns + derived numeric/cyclical features. No fitted statistics."""
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(int)
    minute = (dep // 100) * 60 + dep % 100  # minutes of day: delay accumulates over the day
    X["dep_minute"] = minute
    X["dep_sin"] = np.sin(2 * np.pi * minute / 1440)
    X["dep_cos"] = np.cos(2 * np.pi * minute / 1440)

    month = df["Month"].str.replace("c-", "", regex=False).astype(int)
    X["month"] = month
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)

    day = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int)
    X["day"] = day

    dow = df["DayOfWeek"].str.replace("c-", "", regex=False).astype(int)
    X["dow"] = dow
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    X["dist"] = df["Distance"].astype(float)
    X["dist_log"] = np.log1p(X["dist"])
    return X


# categorical levels fit on TRAINING data only; unseen levels -> NaN (missing)
def _build_cat_levels(dfs):
    frames = []
    for df in dfs:
        f = pd.DataFrame(index=df.index)
        for c in ["UniqueCarrier", "Origin", "Dest"]:
            f[c] = df[c]
        f["route"] = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
        frames.append(f)
    cat_levels = {}
    for c in CAT_COLS:
        vals = pd.concat([f[c] for f in frames]).astype(str)
        cat_levels[c] = pd.Index(sorted(vals.dropna().unique()))
    return cat_levels


cat_levels = _build_cat_levels([train])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = base_frame(df)
    cats = pd.DataFrame(index=df.index)
    for c in ["UniqueCarrier", "Origin", "Dest"]:
        cats[c] = df[c].astype(str)
    cats["route"] = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    for c in CAT_COLS:
        X[c] = pd.Categorical(cats[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.1,
    max_depth=8,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
