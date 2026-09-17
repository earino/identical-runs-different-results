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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
RAW_NUM = ["DepTime", "DepMinutes", "Distance"]
RAW_CAT = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in RAW_CAT}


def to_y_of(frame):
    return (frame[TARGET] == POSITIVE).astype(float).to_numpy()


# hour categorical + smoothed hourly delay-rate prior (train-only; 2005->2006 rate corr 0.96)
hr_tr = (train["DepTime"].astype(int) // 100)
hr_levels = pd.Index(sorted(hr_tr.unique()))
hr_rate = (hr_tr.to_frame("hr").assign(_y=to_y_of(train)).groupby("hr")["_y"].agg(["mean", "count"]))
HR_SMOOTH = 100.0
_glob_rate = float(to_y_of(train).mean())

# volume / congestion features (train-only counts; 2005->2006 log-vol corr 0.90-0.98)
vol_origin = train["Origin"].value_counts()
vol_dest = train["Dest"].value_counts()
vol_origin_hour = (train["Origin"].astype(str) + "_" + hr_tr.astype(str)).value_counts()
VOL_DEFAULT = 1.0


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype(float)
    dep = df["DepTime"].astype(int)
    X["DepMinutes"] = (dep // 100) * 60 + (dep % 100)
    X["Distance"] = df["Distance"].astype(float)
    # hour structure
    hr = dep // 100
    X["DepHour"] = pd.Categorical(hr, categories=hr_levels)
    m = hr.map(hr_rate["mean"]).fillna(_glob_rate).to_numpy(dtype=float)
    c = hr.map(hr_rate["count"]).fillna(0.0).to_numpy(dtype=float)
    X["hr_rate"] = (m * c + HR_SMOOTH * _glob_rate) / (c + HR_SMOOTH)
    # congestion: train-derived flight volumes at origin/dest/origin-by-hour
    X["vol_origin"] = np.log(df["Origin"].map(vol_origin).fillna(VOL_DEFAULT).astype(float))
    X["vol_dest"] = np.log(df["Dest"].map(vol_dest).fillna(VOL_DEFAULT).astype(float))
    oh = df["Origin"].astype(str) + "_" + hr.astype(str)
    X["vol_origin_hour"] = np.log(oh.map(vol_origin_hour).fillna(VOL_DEFAULT).astype(float))
    for c in RAW_CAT:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Xtr, ytr = prepare(train), to_y(train)

# --- bagged native-API ensemble ------------------------------------------------
# Member params from offline 3-fold CV on train (2005), chosen under the 120s run cap:
# d12/lr.05/n400/cs.3/mcw5/ss.9 -> CV 0.7866; ~100s for 7 members incl. 100k-row predict.
PARAMS = dict(
    max_depth=12,
    min_child_weight=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.3,
    tree_method="hist",
    eval_metric="auc",
    nthread=N_JOBS,
)
N_ROUNDS = 400
N_MEMBERS = 10

dtrain = xgb.DMatrix(Xtr, label=ytr, enable_categorical=True)
boosters = []
t0 = time.time()
for seed in range(N_MEMBERS):
    b = xgb.train(dict(PARAMS, seed=seed), dtrain, num_boost_round=N_ROUNDS, verbose_eval=False)
    boosters.append(b)
print(f"Training time: {time.time() - t0:.1f}s")


def _ranks(p: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(p)) / (len(p) - 1)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    dm = xgb.DMatrix(X, enable_categorical=True)
    total = np.zeros(len(X))
    for b in boosters:
        total += _ranks(b.predict(dm, iteration_range=(0, N_ROUNDS)))
    return total / len(boosters)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
