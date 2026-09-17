"""XGBoost binary classifier for the airline delay task (agent-edited file).

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
All feature engineering lives in prepare(); statistics (freq encodings, category levels) are fit on
training data only, at module level, and applied inside prepare().
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

# --- encoders / statistics fit on TRAINING data only --------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Route", "hour_cat", "dist_bin"]
cat_levels = {}
FREQ = {}  # column -> value -> count in train


def _num(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("c-", "", regex=False).astype(int)


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


# category levels from train only
tmp = train.assign(
    Route=_route(train),
    hour_cat=(train["DepTime"] // 100).astype(str),
    dist_bin=pd.cut(train["Distance"], [0, 250, 500, 750, 1000, 1500, 2500, 10000]).astype(str),
)
for c in CAT_COLS:
    cat_levels[c] = pd.Index(sorted(tmp[c].dropna().unique()))
for c in ["Origin", "Dest", "UniqueCarrier", "Route"]:
    FREQ[c] = tmp[c].value_counts()

FEATURE_COLS = [
    "Month", "DayofMonth", "DayOfWeek", "DepTime", "Distance",
    "hour", "minute", "sin_hr", "cos_hr", "is_redeye", "sin_doy", "cos_doy",
    "log_dist",
    "cnt_Origin", "cnt_Dest", "cnt_UniqueCarrier", "cnt_Route",
] + CAT_COLS


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering; safe on unseen rows (encoders fit on train only)."""
    X = pd.DataFrame(index=df.index)
    X["Month"] = _num(df["Month"])
    X["DayofMonth"] = _num(df["DayofMonth"])
    X["DayOfWeek"] = _num(df["DayOfWeek"])
    X["DepTime"] = df["DepTime"].astype(float)
    X["Distance"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(X["Distance"])

    hour = (df["DepTime"] // 100).astype(float)
    minute = (df["DepTime"] % 100).astype(float)
    X["hour"] = hour
    X["minute"] = minute
    ang = 2 * np.pi * hour / 24.0
    X["sin_hr"] = np.sin(ang)
    X["cos_hr"] = np.cos(ang)
    X["is_redeye"] = ((hour >= 24) | (hour <= 5)).astype(float)
    doy = (X["Month"] - 1) * 30.4 + X["DayofMonth"]
    ang2 = 2 * np.pi * doy / 365.0
    X["sin_doy"] = np.sin(ang2)
    X["cos_doy"] = np.cos(ang2)

    # train-fit frequency encodings (unseen -> 0)
    r = _route(df)
    X["cnt_Origin"] = df["Origin"].map(FREQ["Origin"]).fillna(0).astype(float)
    X["cnt_Dest"] = df["Dest"].map(FREQ["Dest"]).fillna(0).astype(float)
    X["cnt_UniqueCarrier"] = df["UniqueCarrier"].map(FREQ["UniqueCarrier"]).fillna(0).astype(float)
    X["cnt_Route"] = r.map(FREQ["Route"]).fillna(0).astype(float)

    X["hour_cat"] = hour.astype(int).astype(str)
    X["dist_bin"] = pd.cut(X["Distance"], [0, 250, 500, 750, 1000, 1500, 2500, 10000]).astype(str)
    X["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    X["Origin"] = df["Origin"].astype(str)
    X["Dest"] = df["Dest"].astype(str)
    X["Route"] = r
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X[FEATURE_COLS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------------------------
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
