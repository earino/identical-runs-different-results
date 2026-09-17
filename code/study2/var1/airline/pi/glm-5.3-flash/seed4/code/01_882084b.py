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
y_all = (train[TARGET] == POSITIVE).astype(int)

# --- encoders fitted on TRAIN only (module level); prepare() applies them to any raw df ------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
TE_M = 20.0  # smoothing pseudo-count for target encoding
PRIOR = float(y_all.mean())


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered string keys used for categoricals and target encodings (no target needed)."""
    k = pd.DataFrame(index=df.index)
    for c in ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]:
        k[c] = df[c].astype(str)
    k["hour"] = (df["DepTime"].astype(float) // 100).astype(int).astype(str)
    k["route"] = k["Origin"] + "_" + k["Dest"]
    k["carrier_hour"] = k["UniqueCarrier"] + "_" + k["hour"]
    return k


def _smooth_te(keys: pd.Series, m: float = TE_M) -> dict:
    g = y_all.groupby(keys).agg(["sum", "count"])
    return ((g["sum"] + m * PRIOR) / (g["count"] + m)).to_dict()


_k_train = _keys(train)
TE_COLS = ["Origin", "Dest", "UniqueCarrier", "route", "carrier_hour", "hour"]
te_maps = {c: _smooth_te(_k_train[c]) for c in TE_COLS}
cnt_maps = {c: _k_train[c].value_counts().to_dict() for c in ["route", "Origin", "Dest"]}
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
cat_levels["hour"] = pd.Index(sorted(_k_train["hour"].unique()))
cat_levels["route"] = pd.Index(sorted(_k_train["route"].unique()))
cat_levels["carrier_hour"] = pd.Index(sorted(_k_train["carrier_hour"].unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows."""
    k = _keys(df)
    X = pd.DataFrame(index=df.index)
    # time-of-day
    hour_num = k["hour"].astype(int)
    dep_min = hour_num * 60 + (df["DepTime"].astype(float) % 100)
    X["hour_num"] = hour_num
    X["hour_cat"] = pd.Categorical(k["hour"], categories=cat_levels["hour"])
    X["dep_sin"] = np.sin(2 * np.pi * dep_min / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * dep_min / 1440.0)
    # calendar
    X["month"] = pd.Categorical(k["Month"], categories=cat_levels["Month"])
    X["dayofmonth"] = pd.Categorical(k["DayofMonth"], categories=cat_levels["DayofMonth"])
    X["dayofweek"] = pd.Categorical(k["DayOfWeek"], categories=cat_levels["DayOfWeek"])
    X["month_num"] = k["Month"].str.slice(2).astype(int)
    X["day_num"] = k["DayofMonth"].str.slice(2).astype(int)
    X["dow_num"] = k["DayOfWeek"].str.slice(2).astype(int)
    # cats
    X["carrier"] = pd.Categorical(k["UniqueCarrier"], categories=cat_levels["UniqueCarrier"])
    X["origin"] = pd.Categorical(k["Origin"], categories=cat_levels["Origin"])
    X["dest"] = pd.Categorical(k["Dest"], categories=cat_levels["Dest"])
    X["route"] = pd.Categorical(k["route"], categories=cat_levels["route"])
    X["carrier_hour"] = pd.Categorical(k["carrier_hour"], categories=cat_levels["carrier_hour"])
    # distance
    dist = df["Distance"].astype(float)
    X["distance"] = dist
    X["dist_log"] = np.log1p(dist)
    # target encodings (maps fitted on train only; unseen -> prior)
    for c in TE_COLS:
        X["te_" + c] = k[c].map(te_maps[c]).fillna(PRIOR).astype(float)
    # frequency
    for c in ["route", "Origin", "Dest"]:
        X["cnt_" + c] = k[c].map(cnt_maps[c]).fillna(0).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def _dm(df: pd.DataFrame) -> xgb.DMatrix:
    return xgb.DMatrix(prepare(df), label=to_y(df), enable_categorical=True)


PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "enable_categorical": True,
    "max_depth": 8,
    "eta": 0.05,
    "min_child_weight": 10.0,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "seed": SEED,
    "n_jobs": N_JOBS,
}

t0 = time.time()
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(train))
n_val = max(1000, len(train) // 10)
val_idx, tr_idx = idx[:n_val], idx[n_val:]
train_in = train.iloc[tr_idx].reset_index(drop=True)
val_in = train.iloc[val_idx].reset_index(drop=True)
y_all_tr = y_all.iloc[tr_idx].reset_index(drop=True)
y_all_val = y_all.iloc[val_idx].reset_index(drop=True)
dtrain = xgb.DMatrix(prepare(train_in), label=y_all_tr.to_numpy(), enable_categorical=True)
dval = xgb.DMatrix(prepare(val_in), label=y_all_val.to_numpy(), enable_categorical=True)

bst = xgb.train(PARAMS, dtrain, num_boost_round=2000, evals=[(dval, "val")],
                early_stopping_rounds=60, verbose_eval=False)
best_it = bst.best_iteration
print(f"Early-stop fit: best_iter={best_it} val_auc={bst.best_score:.4f} time={time.time() - t0:.1f}s")

# refit on the full training set with the tuned number of trees
t1 = time.time()
n_full = int(round(best_it * 1.1)) + 1
dtrain_full = xgb.DMatrix(prepare(train), label=y_all.to_numpy(), enable_categorical=True)
model = xgb.train(PARAMS, dtrain_full, num_boost_round=n_full)
print(f"Refit ({n_full} trees) time={time.time() - t1:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict(xgb.DMatrix(prepare(df), enable_categorical=True))


t2 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t2:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
