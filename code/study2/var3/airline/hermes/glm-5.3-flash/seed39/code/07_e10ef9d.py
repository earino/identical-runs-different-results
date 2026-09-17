"""XGBoost binary classifier for the airline delay task.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

All feature engineering lives in prepare(); all fitted statistics (category levels, frequency
maps) are computed from the training data at module level and only *applied* inside prepare().
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

# --- fitted statistics (training data only) -----------------------------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).dropna().unique())) for c in CAT_COLS}
freq_maps = {}
for c in ["Origin", "Dest", "UniqueCarrier"]:
    vc = train[c].astype(str).value_counts(normalize=True)
    freq_maps[c] = vc.to_dict()
FREQ_GLOBAL = 0.0


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + ">" + df["Dest"].astype(str)


route_freq = _route(train).value_counts(normalize=True).to_dict()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    dt = df["DepTime"].astype("int64")

    # time-of-day decomposition (DepTime is hhmm; 2400-26xx rows exist = very late scheduled deps)
    X["DepTime"] = dt
    tod = (dt % 2400) / 100.0  # continuous 0..24, wraps late-night rows to early morning
    X["tod"] = tod
    ang = 2 * np.pi * tod / 24.0
    X["tod_sin"] = np.sin(ang)
    X["tod_cos"] = np.cos(ang)
    X["hour24"] = dt // 100  # 0..26 keeps late rows separable
    X["minute"] = dt % 100
    X["late_night"] = (dt >= 2400).astype("int64")

    # calendar as numbers
    X["month_num"] = df["Month"].astype(str).str.slice(2).astype("int64")
    X["dom_num"] = df["DayofMonth"].astype(str).str.slice(2).astype("int64")
    X["dow_num"] = df["DayOfWeek"].astype(str).str.slice(2).astype("int64")
    X["weekend"] = X["dow_num"].isin([1, 7]).astype("int64")

    # distance
    dist = df["Distance"].astype("float64")
    X["Distance"] = dist
    X["log_dist"] = np.log1p(dist)

    # frequencies (fit on train)
    for c in ["Origin", "Dest", "UniqueCarrier"]:
        X[f"{c}_freq"] = df[c].astype(str).map(freq_maps[c]).astype("float64").fillna(FREQ_GLOBAL)
    X["route_freq"] = _route(df).map(route_freq).astype("float64").fillna(FREQ_GLOBAL)

    # native categoricals
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.02,
    max_depth=8,
    min_child_weight=5,
    subsample=0.5,
    colsample_bytree=0.5,
    reg_alpha=0.0,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
EARLY_STOP = 250
VALID_FRAC = 0.15

t0 = time.time()
X = prepare(train)
y = to_y(train)
model = xgb.XGBClassifier(**PARAMS, early_stopping_rounds=EARLY_STOP)
model.fit(X, y, eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
