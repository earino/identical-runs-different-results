"""Airline delay XGBoost. Feature engineering lives in prepare(); encoders fitted on train only."""
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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Route"]
FREQ_COLS = ["UniqueCarrier", "Origin", "Dest", "Route"]

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")


def _int_from_c(s: pd.Series) -> pd.Series:
    return s.astype(str).str.slice(2).astype(float)


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise transforms only (no fitted statistics)."""
    X = pd.DataFrame(index=df.index)
    X["hour"] = (df["DepTime"] // 100).astype(float)
    X["minute"] = (df["DepTime"] % 100).astype(float)
    X["month"] = _int_from_c(df["Month"])
    X["dayofmonth"] = _int_from_c(df["DayofMonth"])
    X["dayofweek"] = _int_from_c(df["DayOfWeek"])
    X["distance"] = df["Distance"].astype(float)
    X["log_distance"] = np.log1p(df["Distance"].astype(float))
    X["Origin"] = df["Origin"].astype(str)
    X["Dest"] = df["Dest"].astype(str)
    X["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    X["Route"] = X["Origin"] + "_" + X["Dest"]
    return X


# --- fitted-on-train statistics (module level; applied inside prepare) ---------
_train_bf = base_features(train)
cat_levels = {c: pd.Index(sorted(_train_bf[c].unique())) for c in CAT_COLS}
freq_maps = {c: _train_bf[c].value_counts().to_dict() for c in FREQ_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = base_features(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    for c in FREQ_COLS:
        X[c + "_freq"] = X[c].astype(str).map(freq_maps[c]).fillna(0.0).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def time_sort_key(df: pd.DataFrame) -> np.ndarray:
    bf = base_features(df)
    return (bf["month"] * 1000000 + bf["dayofmonth"] * 1000 + df["DepTime"]).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
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
