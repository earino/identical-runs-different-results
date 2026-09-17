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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "dep_hour_cat", "dist_bin", "dep_half_cat", "dep_qtr_cat"]
NUM_COLS = [
    "Month", "DayofMonth", "DayOfWeek",
    "dep_min", "dep_sin", "dep_cos", "dep_wrap",
    "Distance", "log_distance", "org_cnt", "dest_cnt", "car_med_hour", "car_med_dist", "org_hr_cnt", "dest_hr_cnt",
]
FEATURE_COLS = CAT_COLS + NUM_COLS
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in ["UniqueCarrier", "Origin", "Dest"]}
HOUR_LEVELS = pd.Index(list(range(24)))
HALF_LEVELS = pd.Index(list(range(48)))
QTR_LEVELS = pd.Index(list(range(96)))
ORG_CNT = train["Origin"].value_counts()
DEST_CNT = train["Dest"].value_counts()
_tr_hour = (train["DepTime"].astype(int) // 100) % 24
CAR_MED_HOUR = train.assign(_h=_tr_hour).groupby("UniqueCarrier")["_h"].median()
CAR_MED_DIST = train.groupby("UniqueCarrier")["Distance"].median()
ORG_HR = train.assign(_h=_tr_hour).groupby(["Origin", "_h"]).size()
DEST_HR = train.assign(_h=_tr_hour).groupby(["Dest", "_h"]).size()
DIST_EDGES = [0, 300, 600, 1000, 1500, 2500, 7000]
DIST_LEVELS = pd.Index([str(i) for i in pd.IntervalIndex.from_breaks(DIST_EDGES)])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[["UniqueCarrier", "Origin", "Dest"]].copy()
    for c in ["UniqueCarrier", "Origin", "Dest"]:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN

    month = df["Month"].astype(str).str.slice(2).astype(int)
    dom = df["DayofMonth"].astype(str).str.slice(2).astype(int)
    dow = df["DayOfWeek"].astype(str).str.slice(2).astype(int)
    dt = df["DepTime"].astype(int)
    hour = (dt // 100) % 24
    minute = dt % 100
    wrap = (dt >= 2400).astype(int)
    ang = 2.0 * np.pi * (hour * 60.0 + minute) / 1440.0

    X["Month"] = month
    X["DayofMonth"] = dom
    X["DayOfWeek"] = dow
    X["dep_min"] = minute
    X["dep_sin"] = np.sin(ang)
    X["dep_cos"] = np.cos(ang)
    X["dep_wrap"] = wrap
    X["Distance"] = df["Distance"].astype(float)
    X["log_distance"] = np.log1p(X["Distance"])
    X["org_cnt"] = np.log1p(df["Origin"].map(ORG_CNT).fillna(0.0))
    X["dest_cnt"] = np.log1p(df["Dest"].map(DEST_CNT).fillna(0.0))
    _dh = (df["DepTime"].astype(int) // 100) % 24
    X["car_med_hour"] = df["UniqueCarrier"].map(CAR_MED_HOUR).fillna(float(CAR_MED_HOUR.median())) - _dh
    X["car_med_dist"] = df["UniqueCarrier"].map(CAR_MED_DIST).fillna(float(CAR_MED_DIST.median()))
    _hr = (df["DepTime"].astype(int) // 100) % 24
    oh_idx = pd.MultiIndex.from_arrays([df["Origin"].to_numpy(), _hr.to_numpy()])
    dh_idx = pd.MultiIndex.from_arrays([df["Dest"].to_numpy(), _hr.to_numpy()])
    X["org_hr_cnt"] = np.log1p(ORG_HR.reindex(oh_idx).fillna(0.0).to_numpy())
    X["dest_hr_cnt"] = np.log1p(DEST_HR.reindex(dh_idx).fillna(0.0).to_numpy())
    X["dep_hour_cat"] = pd.Categorical(hour, categories=HOUR_LEVELS)
    X["dep_half_cat"] = pd.Categorical(hour * 2 + (minute >= 30).astype(int), categories=HALF_LEVELS)
    X["dep_qtr_cat"] = pd.Categorical((hour * 60 + minute) // 15, categories=QTR_LEVELS)
    X["dist_bin"] = pd.Categorical(
        pd.Categorical(pd.cut(X["Distance"], bins=DIST_EDGES, right=False)).astype(str), categories=DIST_LEVELS
    )
    return X[FEATURE_COLS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def make_model(seed: int, col: float, ss: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=0,
    max_leaves=512,
    grow_policy="lossguide",
    min_child_weight=10.0,
    subsample=0.8,
    colsample_bytree=0.6,
    colsample_bynode=0.8,
    reg_lambda=5.0,
    reg_alpha=1.0,
    tree_method="hist",
    max_bin=512,
    enable_categorical=True,
    early_stopping_rounds=150,
    eval_metric="auc",
    random_state=seed,
    n_jobs=N_JOBS,
)


t0 = time.time()
Xtr = prepare(train)
ytr = to_y(train)
Xev = prepare(evald)
models = []
for seed, col, ss in [(42, 0.6, 0.8), (7, 0.75, 0.9)]:
    m = make_model(seed, col, ss)
    m.fit(Xtr, ytr, eval_set=[(Xev, to_y(evald))], verbose=False)
    models.append(m)
    print(f"model seed={seed} col={col} ss={ss} trees={m.best_iteration + 1}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
