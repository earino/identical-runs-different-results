"""XGBoost binary classifier for flight delay prediction (airline task).

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
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")


# --- feature engineering (statistics fit on train only) -----------------------
def _num_c(series: pd.Series) -> pd.Series:
    """'c-4' -> 4 (the categorical codes are 1-based integers)."""
    return series.str.replace("c-", "", regex=False).astype(int)


CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
# 30-minute time-of-day bins, treated as a categorical so the model can fit an arbitrary (non-monotone)
# delay-vs-time profile.
TOD_BIN_MIN = 30
TOD_BIN_LEVELS = [str(b) for b in range(1440 // TOD_BIN_MIN + 2)]


def _tod_minutes(df: pd.DataFrame) -> pd.Series:
    dep = df["DepTime"].astype(int)
    return ((dep // 100) * 60 + dep % 100) % 1440


def _tod_bin(df: pd.DataFrame) -> pd.Series:
    return (_tod_minutes(df) // TOD_BIN_MIN).astype(str)


# carrier x time-of-day interaction: carriers have distinct delay profiles over the day
CARR_TOD_LEVELS = pd.Index(sorted((train["UniqueCarrier"] + "_" + _tod_bin(train)).unique()))
# origin airport x hour interaction: captures airport congestion by time of day
ORIG_HR_LEVELS = pd.Index(
    sorted((train["Origin"] + "_" + (_tod_minutes(train) // 60).astype(str)).unique())
)
DIST_BIN_MI = 250
DIST_BIN_LEVELS = [str(i) for i in range(41)]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything
    # computed on `train`/`evald` outside this function would NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["Month"] = _num_c(df["Month"])
    X["DayofMonth"] = _num_c(df["DayofMonth"])
    dow = _num_c(df["DayOfWeek"])
    X["DayOfWeek"] = dow

    dep = df["DepTime"].astype(int)
    hour = dep // 100
    minute = dep % 100
    X["DepHour"] = hour                      # 0..26 (24+ = after-midnight departures)
    X["DepMinute"] = minute
    tod = (hour * 60 + minute) % 1440         # clock time of day
    X["TOD"] = tod
    X["DepTime"] = dep
    tb = _tod_bin(df)
    X["TODbinC"] = pd.Categorical(tb, categories=TOD_BIN_LEVELS)
    X["CarrTOD"] = pd.Categorical(df["UniqueCarrier"] + "_" + tb, categories=CARR_TOD_LEVELS)
    X["OrigHr"] = pd.Categorical(
        df["Origin"] + "_" + (tod // 60).astype(str), categories=ORIG_HR_LEVELS
    )
    X["IsWeekend"] = (dow >= 6).astype(np.int8)

    X["Distance"] = df["Distance"].astype(float)
    X["DistBin"] = pd.Categorical(
        (df["Distance"].astype(int) // DIST_BIN_MI).astype(str), categories=DIST_BIN_LEVELS
    )
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Small ensemble of depth-diverse XGBoost models; averaging predictions reduces variance.
MODEL_CFGS = [
    dict(max_depth=14, random_state=1),
    dict(max_depth=16, random_state=2),
    dict(max_depth=18, random_state=3),
]
COMMON = dict(
    n_estimators=300,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.5,
    reg_alpha=1.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_train = prepare(train)
y_train = to_y(train)
models = []
for cfg in MODEL_CFGS:
    m = xgb.XGBClassifier(**COMMON, **cfg)
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
