"""XGBoost binary classifier for the airline delay task (harness benchmark edition).

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
  All feature engineering lives in `prepare()`, which `predict_proba` uses; statistics are fit on the
  training data only (module-level constants set by `_prepare_frame(train, fit=True)`) and applied
  identically to any new dataframe passed to `predict_proba`.
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

# ------------------------------------------------------------- constants fit on TRAIN only
MONTH_NUM = {"c-1": 1, "c-2": 2, "c-3": 3, "c-4": 4, "c-5": 5, "c-6": 6,
             "c-7": 7, "c-8": 8, "c-9": 9, "c-10": 10, "c-11": 11, "c-12": 12}
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
ORIG_FQ_BINS = None      # quantile edges of train origin frequencies
DEST_FQ_BINS = None      # quantile edges of train dest frequencies
GLOBALS = {}             # shrink-encoded per-group delay rates + global mean
ORIG_TOP = None          # top-40 train airports (one-hots)
DEST_TOP = None


def _shrink(tab: pd.DataFrame, gmean: float, k: float = 20.0) -> pd.DataFrame:
    """Empirical-Bayes shrinkage of per-group means toward the global mean; k = pseudo-count."""
    post = (tab["count"] * tab["mean"] + k * gmean) / (tab["count"] + k)
    return pd.DataFrame({"mean": post, "count": tab["count"]})


def _prepare_frame(df: pd.DataFrame, fit: bool) -> pd.DataFrame:
    """Shared feature preparation. fit=True (training data only) (re)fits the module-level
    statistics; fit=False applies the already-fit statistics to new rows."""
    global ORIG_FQ_BINS, DEST_FQ_BINS, GLOBALS, ORIG_TOP, DEST_TOP

    X = pd.DataFrame(index=df.index)
    month = df["Month"].map(MONTH_NUM).astype("float64")
    dow = df["DayOfWeek"].astype(str).str.lstrip("c").astype("float64")
    dom = df["DayofMonth"].astype(str).str.lstrip("c").astype("float64")
    deptime = pd.to_numeric(df["DepTime"], errors="coerce").astype("float64")

    # ---- calendar
    X["month"] = month
    X["dow"] = dow
    X["dom"] = dom
    X["dom_edge"] = ((dom <= 7) | (dom >= 29)).astype("float64")

    # ---- deptime cyclic + coarse bands
    X["deptime"] = deptime
    X["hour"] = np.floor(deptime / 100.0)
    X["dep_sin1"] = np.sin(2 * np.pi * deptime / 1440.0)
    X["dep_cos1"] = np.cos(2 * np.pi * deptime / 1440.0)
    X["dep_sin2"] = np.sin(4 * np.pi * deptime / 1440.0)
    X["dep_cos2"] = np.cos(4 * np.pi * deptime / 1440.0)
    X["dep_sin3"] = np.sin(6 * np.pi * deptime / 1440.0)
    X["dep_cos3"] = np.cos(6 * np.pi * deptime / 1440.0)
    X["dep_group"] = np.where(deptime < 600, 0.0,
                     np.where(deptime < 1200, 1.0,
                     np.where(deptime < 1800, 2.0, 3.0)))

    # ---- distance
    dist = pd.to_numeric(df["Distance"], errors="coerce").astype("float64")
    X["dist"] = dist
    X["dist_log"] = np.log1p(dist)

    # ---- raw categorical passthrough (XGBoost native categoricals)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=CAT_LEVELS[c])

    return X.replace([np.inf, -np.inf], np.nan)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    return _prepare_frame(df, fit=False)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# ---- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(_prepare_frame(train, fit=True), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
