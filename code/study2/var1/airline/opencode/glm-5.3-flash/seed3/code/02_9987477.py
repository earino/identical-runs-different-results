"""Airline delay XGBoost classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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

NUM_COLS = ["Month_i", "Day_i", "DoW_i", "DepTime", "DepTime_mod", "DepTime_sin", "DepTime_cos",
            "Distance", "LogDist"]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Route", "DepHour"]
TE_COLS = ["UniqueCarrier", "Origin", "Dest", "Route", "DepHour", "Date"]
FEATURE_COLS = NUM_COLS + CAT_COLS + [c + "_te" for c in TE_COLS] + ["Origin_cnt", "Dest_cnt", "Route_cnt"]


def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


# encoders fit on TRAINING DATA ONLY (module level is fine: predict_proba reuses these fixed stats)
Y_POS = (train[TARGET] == POSITIVE).astype(float)
PRIOR = float(Y_POS.mean())


def _te_map(keys: pd.Series) -> pd.Series:
    grp = pd.DataFrame({"k": keys.to_numpy(), "y": Y_POS.to_numpy()}).groupby("k")["y"].agg(["mean", "count"])
    m = 60.0
    return (grp["count"] * grp["mean"] + m * PRIOR) / (grp["count"] + m)


def _te_keys(df: pd.DataFrame) -> pd.DataFrame:
    k = pd.DataFrame(index=df.index)
    k["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    k["Origin"] = df["Origin"].astype(str)
    k["Dest"] = df["Dest"].astype(str)
    k["Route"] = k["Origin"] + "_" + k["Dest"]
    k["DepHour"] = "h" + ((pd.to_numeric(df["DepTime"], errors="coerce") % 2400 // 100).fillna(0).astype(int).clip(0, 23)).astype(str)
    k["Date"] = _to_int(df["Month"]).astype(str) + "_" + _to_int(df["DayofMonth"]).astype(str)
    return k


KEYS_TRAIN = _te_keys(train)
TE_MAPS = {c: _te_map(KEYS_TRAIN[c]) for c in TE_COLS}
CNT_MAPS = {c: KEYS_TRAIN[c].value_counts() for c in ["Origin", "Dest", "Route"]}
CAT_LEVELS = {c: pd.Index(sorted(KEYS_TRAIN[c].unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["Month_i"] = _to_int(df["Month"])
    X["Day_i"] = _to_int(df["DayofMonth"])
    X["DoW_i"] = _to_int(df["DayOfWeek"])
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    X["DepTime"] = dt
    dt_mod = dt % 2400
    X["DepTime_mod"] = dt_mod
    hh = (dt_mod // 100).fillna(0).astype(int).clip(0, 23)
    mm = (dt_mod % 100).clip(0, 59)
    minutes = hh * 60 + mm
    X["DepTime_sin"] = np.sin(2 * np.pi * minutes / 1440)
    X["DepTime_cos"] = np.cos(2 * np.pi * minutes / 1440)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["LogDist"] = np.log1p(X["Distance"])
    X["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    X["Origin"] = df["Origin"].astype(str)
    X["Dest"] = df["Dest"].astype(str)
    X["Route"] = X["Origin"] + "_" + X["Dest"]
    X["DepHour"] = "h" + hh.astype(int).astype(str).str.zfill(2)
    keys = _te_keys(df)
    for c in TE_COLS:
        X[c + "_te"] = keys[c].map(TE_MAPS[c])
    for c in ["Origin", "Dest", "Route"]:
        X[c + "_cnt"] = np.log1p(keys[c].map(CNT_MAPS[c]).fillna(0))
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    return X[FEATURE_COLS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


MODEL_KW = dict(
    learning_rate=0.05,
    max_depth=10,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="auc",
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
y_all = to_y(train)
rng = np.random.RandomState(SEED)
idx_val = rng.rand(len(train)) < 0.2
X_all = prepare(train)
m0 = xgb.XGBClassifier(n_estimators=1500, early_stopping_rounds=50, **MODEL_KW)
m0.fit(X_all[~idx_val], y_all[~idx_val], eval_set=[(X_all[idx_val], y_all[idx_val])], verbose=False)
best_n = int(m0.best_iteration) + 1
print(f"Early-stop best iteration: {best_n} (val auc {m0.best_score:.4f})")

model = xgb.XGBClassifier(n_estimators=best_n, **MODEL_KW)
model.fit(X_all, y_all)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
