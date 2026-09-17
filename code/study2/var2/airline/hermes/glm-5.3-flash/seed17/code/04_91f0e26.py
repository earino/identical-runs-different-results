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
# carrier x 30-minute departure slot: delay propensity is carrier-specific per time of day
_dep = train["DepTime"].astype(np.int32)
_carsl30 = train["UniqueCarrier"] + "_" + (((_dep // 100) % 24) * 2 + (_dep % 100) // 30).astype(str)
CARSL30_LEVELS = pd.Index(sorted(_carsl30.unique()))


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
        "doy": (month - 1) * 31 + dom,  # day-of-year proxy (31-day months)
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
    dep = df["DepTime"].astype(np.int32)
    slot30 = ((dep // 100) % 24) * 2 + (dep % 100) // 30
    X["carsl30"] = pd.Categorical(df["UniqueCarrier"] + "_" + slot30.astype(str), categories=CARSL30_LEVELS)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=700,
    learning_rate=0.03,
    max_depth=9,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
SEEDS = (42, 7, 2026, 99, 123)

X = prepare(train)
y = to_y(train)

_MODELS = []
t0 = time.time()
for sd in SEEDS:
    m = xgb.XGBClassifier(**{**PARAMS, "random_state": sd})
    m.fit(X, y)
    _MODELS.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(SEEDS)} seeds x {PARAMS['n_estimators']} trees)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xd = prepare(df)
    p = np.mean([m.predict_proba(Xd)[:, 1] for m in _MODELS], axis=0)
    return p


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
