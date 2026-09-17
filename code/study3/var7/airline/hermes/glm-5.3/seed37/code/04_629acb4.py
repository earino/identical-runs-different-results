"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# raw numeric columns
num_cols = ["DepTime", "Distance"]
# categorical string columns
cyc_cols = ["Month", "DayofMonth", "DayOfWeek"]  # keep as c-<n> strings, categorical
hcc_cols = ["UniqueCarrier", "Origin", "Dest"]

# --- feature engineering (all inside prepare: it must apply to the hidden holdout) ---
# time-of-day blocks and cyclic encodings
# time-of-day blocks



def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering: called by prepare() on both train and hidden holdout."""
    X = pd.DataFrame(index=df.index)
    dt = df["DepTime"].astype(int)
    h, m = np.divmod(dt, 100)
    hour = (h + m / 60.0).astype(float)
    X["hour"] = hour
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    # block one-hots via sin/cos of block id

    blk = pd.Series(np.select(
        [dt < 600, dt < 1200, dt < 1800], [0, 1, 2], default=3), index=df.index)
    X["blk_sin"] = np.sin(2 * np.pi * blk / 4)
    X["blk_cos"] = np.cos(2 * np.pi * blk / 4)
    # day-of-week / month cyclic (parse c-<n> strings)
    dow = df["DayOfWeek"].str.extract(r"c-(\d+)", expand=False).astype(float)
    X["dow"] = dow
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    mon = df["Month"].str.extract(r"c-(\d+)", expand=False).astype(float)
    X["mon"] = mon
    X["mon_sin"] = np.sin(2 * np.pi * mon / 12)
    X["mon_cos"] = np.cos(2 * np.pi * mon / 12)
    dom = df["DayofMonth"].str.extract(r"c-(\d+)", expand=False).astype(float)
    X["dom"] = dom
    X["dom_sin"] = np.sin(2 * np.pi * dom / 31)
    X["dom_cos"] = np.cos(2 * np.pi * dom / 31)
    X["dist"] = df["Distance"].astype(float)
    X["dist_log"] = np.log1p(X["dist"])
    # flight/route identity: same-flight recurrence is handled by cat encoding below
    X["dep_blk_dist"] = X["blk_sin"] * X["dist_log"]  # crude interaction
    return X


cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cyc_cols + hcc_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = _add_features(df)
    for c in cyc_cols + hcc_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# The 2005->2006 shift punishes capacity; keep trees shallow and regularized.
model = xgb.XGBClassifier(
    n_estimators=1200,
    max_depth=4,
    learning_rate=0.02,
    min_child_weight=20,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=2.0,
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
