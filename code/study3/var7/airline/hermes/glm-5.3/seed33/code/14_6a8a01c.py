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
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# --- train-only statistics ----------------------------------------------------
_h_tr = (train["DepTime"] // 100) % 24
_blk_tr = (train["DepTime"] // 100) * 2 + ((train["DepTime"] % 100) // 30)
_blk15_tr = (train["DepTime"] // 100) * 4 + ((train["DepTime"] % 100) // 15)
_blk5_tr = (train["DepTime"] // 100) * 12 + ((train["DepTime"] % 100) // 5)
SCHED_CNT = {}      # flights per (entity, hour)
SCHED30_CNT = {}    # flights per (entity, 30-min block)
SCHED15_CNT = {}    # flights per (entity, 15-min block)
SCHED5_CNT = {}     # flights per (entity, 5-min block)
for _c in ["UniqueCarrier", "Origin", "Dest"]:
    SCHED_CNT[_c] = (train[_c].astype(str) + "|" + _h_tr.astype(str)).value_counts()
    SCHED30_CNT[_c] = (train[_c].astype(str) + "|" + _blk_tr.astype(str)).value_counts()
    SCHED15_CNT[_c] = (train[_c].astype(str) + "|" + _blk15_tr.astype(str)).value_counts()
    SCHED5_CNT[_c] = (train[_c].astype(str) + "|" + _blk5_tr.astype(str)).value_counts()

_rt_tr = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
ROUTE_CNT = _rt_tr.value_counts()
ROUTE_H_CNT = (_rt_tr + "|" + _h_tr.astype(str)).value_counts()
ROUTE_B_CNT = (_rt_tr + "|" + _blk_tr.astype(str)).value_counts()
ROUTE_B15_CNT = (_rt_tr + "|" + _blk15_tr.astype(str)).value_counts()
ROUTE_B5_CNT = (_rt_tr + "|" + _blk5_tr.astype(str)).value_counts()

# flight-identity counts: (carrier, route, time-block) ~ a scheduled flight number
_fid_tr = train["UniqueCarrier"].astype(str) + "|" + _rt_tr
FID_H_CNT = (_fid_tr + "|" + _h_tr.astype(str)).value_counts()
FID_B15_CNT = (_fid_tr + "|" + _blk15_tr.astype(str)).value_counts()
RT_MEAN_TIME = train.groupby(_rt_tr)["DepTime"].mean()
RT_MEAN_TIME_DEFAULT = float(train["DepTime"].mean())
RT_MEAN_DIST = train.groupby(_rt_tr)["Distance"].mean()
RT_MEAN_DIST_DEFAULT = float(train["Distance"].mean())


def _cnum(s: pd.Series) -> pd.Series:
    return s.astype(str).str.slice(2).astype(float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    h = (X["DepTime"] // 100) % 24
    blk = (X["DepTime"] // 100) * 2 + ((X["DepTime"] % 100) // 30)
    blk15 = (X["DepTime"] // 100) * 4 + ((X["DepTime"] % 100) // 15)
    blk5 = (X["DepTime"] // 100) * 12 + ((X["DepTime"] % 100) // 5)
    # schedule-density counts (train-only stats)
    for key_col, name in [("UniqueCarrier", "ch"), ("Origin", "oh"), ("Dest", "dh")]:
        key = df[key_col].astype(str) + "|" + h.astype(str)
        X[f"sched_cnt_{name}"] = np.log1p(key.map(SCHED_CNT[key_col]).fillna(0))
        key = df[key_col].astype(str) + "|" + blk.astype(str)
        X[f"sched30_cnt_{name}"] = np.log1p(key.map(SCHED30_CNT[key_col]).fillna(0))
        key = df[key_col].astype(str) + "|" + blk15.astype(str)
        X[f"sched15_cnt_{name}"] = np.log1p(key.map(SCHED15_CNT[key_col]).fillna(0))
        key = df[key_col].astype(str) + "|" + blk5.astype(str)
        X[f"sched5_cnt_{name}"] = np.log1p(key.map(SCHED5_CNT[key_col]).fillna(0))
    # route-level features (train-only stats)
    rt = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["route_cnt"] = np.log1p(rt.map(ROUTE_CNT).fillna(0))
    X["route_hour_cnt"] = np.log1p((rt + "|" + h.astype(str)).map(ROUTE_H_CNT).fillna(0))
    X["route_blk_cnt"] = np.log1p((rt + "|" + blk.astype(str)).map(ROUTE_B_CNT).fillna(0))
    X["route_mean_deptime"] = rt.map(RT_MEAN_TIME).fillna(RT_MEAN_TIME_DEFAULT)
    X["route_mean_dist"] = rt.map(RT_MEAN_DIST).fillna(RT_MEAN_DIST_DEFAULT)
    X["dt_dev"] = df["DepTime"] - X["route_mean_deptime"]
    # flight-identity counts (train-only stats)
    fid = df["UniqueCarrier"].astype(str) + "|" + rt
    X["fid_hour_cnt"] = np.log1p((fid + "|" + h.astype(str)).map(FID_H_CNT).fillna(0))
    X["fid_blk15_cnt"] = np.log1p((fid + "|" + blk15.astype(str)).map(FID_B15_CNT).fillna(0))
    # distance transforms
    X["log_dist"] = np.log1p(X["Distance"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# ensemble of XGBoost models over a range of depths, with feature bagging:
# colsample_bytree=0.3 decorrelates members; shallow trees transfer better
# across the 2005->2006 time shift; averaging over depths adds robustness.
MEMBERS = [dict(max_depth=d, n_estimators=600, learning_rate=0.03, colsample_bytree=cs)
           for d in range(1, 11) for cs in (0.2, 0.3)]

t0 = time.time()
X_train = prepare(train)
y_train = to_y(train)
# upweight later months: closer to the eval year, less distribution shift
_w = 0.5 + 0.5 * (_cnum(train["Month"]).to_numpy() / 12.0)
models = []
for i, cfg in enumerate(MEMBERS):
    m = xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + i,
        n_jobs=N_JOBS,
        **cfg,
    )
    m.fit(X_train, y_train, sample_weight=_w)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  members={len(models)}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    out = np.zeros(len(X))
    for m in models:
        out += m.predict_proba(X)[:, 1] / len(models)
    return out


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
