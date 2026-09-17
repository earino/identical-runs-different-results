"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- feature engineering (ALL of it lives in prepare(); stats below are fit on train only) ----
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
# carrier x scheduled-hour interaction: delay propensities are carrier-specific per time of day
_carhour = train.UniqueCarrier + "_" + ((train.DepTime // 100) % 24).astype(str)
CARHOUR_LEVELS = pd.Index(sorted(_carhour.unique()))


def _cal_parts(df: pd.DataFrame) -> dict:
    """Numeric + cyclic encodings of the c-<n> calendar columns."""
    month = df["Month"].str[2:].astype(np.int32)
    dom = df["DayofMonth"].str[2:].astype(np.int32)
    dow = df["DayOfWeek"].str[2:].astype(np.int32)
    mang = 2 * np.pi * (month - 1) / 12.0
    dang = 2 * np.pi * (dom - 1) / 31.0
    wang = 2 * np.pi * (dow - 1) / 7.0
    return {
        "Month_num": month,
        "Day_num": dom,
        "DOW_num": dow,
        "month_sin": np.sin(mang),
        "month_cos": np.cos(mang),
        "dom_sin": np.sin(dang),
        "dom_cos": np.cos(dang),
        "dow_sin": np.sin(wang),
        "dow_cos": np.cos(wang),
    }


def _time_parts(df: pd.DataFrame) -> dict:
    """DepTime is scheduled departure hhmm (values up to 2620 = after-midnight)."""
    dep = df["DepTime"].astype(np.int32)
    hour = (dep // 100) % 24
    minute = dep % 100
    ang = 2 * np.pi * (hour * 60 + minute) / 1440.0
    return {
        "DepTime": dep,
        "dep_hour": hour,
        "dep_minute": minute,
        "tod_sin": np.sin(ang),
        "tod_cos": np.cos(ang),
        "is_red_eye": (hour < 6).astype(np.int8),
        "is_evening": ((hour >= 16) & (hour < 22)).astype(np.int8),
    }


def _dist_parts(df: pd.DataFrame) -> dict:
    dist = df["Distance"].astype(np.float64)
    return {
        "Distance": dist,
        "log_dist": np.log1p(dist),
        "sqrt_dist": np.sqrt(dist),
    }


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for fn in (_time_parts, _cal_parts, _dist_parts):
        for k, v in fn(df).items():
            X[k] = v
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    hour = ((df["DepTime"].astype(np.int32) // 100) % 24).astype(str)
    X["carhour"] = pd.Categorical(df["UniqueCarrier"] + "_" + hour, categories=CARHOUR_LEVELS)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=320,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model = xgb.XGBClassifier(**PARAMS)
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s ({PARAMS['n_estimators']} trees)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
