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
SEEDS = (42, 7, 13, 101, 202)

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- encoders fitted on TRAIN only, applied inside prepare ---------------------
CAT_COLS = ["Month", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]  # DayofMonth dropped: pure year-overfit noise
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
hh_tr = (train["DepTime"].astype(int) // 100).astype(str)
_h30 = (train["DepTime"].astype(int) // 30).astype(str)
HLEVELS = pd.Index(sorted(hh_tr.unique())).astype(str)
H30LEVELS = pd.Index(sorted(_h30.unique())).astype(str)
CAR_H30_LEVELS = pd.Index(sorted((train["UniqueCarrier"] + "|" + _h30).unique())).astype(str)
# frequency counts fitted on train only (robust popularity signals)
ROUTE_CNT = (train["Origin"] + "-" + train["Dest"]).value_counts()
ORG_CNT = train["Origin"].value_counts()
DST_CNT = train["Dest"].value_counts()
DIST_BINS = pd.qcut(train["Distance"], 10, retbins=True)[1]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    dt = df["DepTime"].astype(int)
    hh = (dt // 100).astype(str)
    h30 = (dt // 30).astype(str)
    X = df[["DepTime", "Distance"]].copy()
    X["hcat"] = pd.Categorical(hh, categories=HLEVELS)  # unseen bins -> NaN
    X["h30"] = pd.Categorical(h30, categories=H30LEVELS)
    X["hour"] = dt // 100
    X["minute"] = dt % 100
    ang = 2 * np.pi * (dt % 1440) / 1440.0
    X["tsin"] = np.sin(ang)
    X["tcos"] = np.cos(ang)
    X["car_h"] = pd.Categorical(df["UniqueCarrier"] + "|" + h30, categories=CAR_H30_LEVELS)
    X["log_dist"] = np.log1p(df["Distance"])
    X["route_cnt"] = np.log1p((df["Origin"] + "-" + df["Dest"]).map(ROUTE_CNT).fillna(0))
    X["org_cnt"] = np.log1p(df["Origin"].map(ORG_CNT).fillna(0))
    X["dst_cnt"] = np.log1p(df["Dest"].map(DST_CNT).fillna(0))
    X["distbin"] = pd.Categorical(
        pd.cut(df["Distance"], bins=DIST_BINS, labels=False), categories=pd.Index(range(10)).astype(str)
    )
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X = prepare(train)
y = to_y(train)

PARAMS = dict(
    n_estimators=400,
    learning_rate=0.0375,
    max_depth=16,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.4,
    tree_method="hist",
    max_bin=512,
    enable_categorical=True,
    n_jobs=N_JOBS,
)

t0 = time.time()
models = []
for seed in SEEDS:
    m = xgb.XGBClassifier(random_state=seed, **PARAMS)
    m.fit(X, y)
    models.append(m)
print(f"Trained {len(SEEDS)} bagged models in {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = prepare(df)
    return np.mean([m.predict_proba(P)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
