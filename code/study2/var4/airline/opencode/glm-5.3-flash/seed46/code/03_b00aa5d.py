"""Airline delay XGBoost. v4: + OOF target encoding (Origin/Dest/Carrier/route) + counts.

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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
TE_KEYS = ["UniqueCarrier", "Origin", "Dest", "route"]
TE_M = 50
N_FOLDS = 5
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in CAT_COLS}

y_bin = (train[TARGET] == POSITIVE).astype(float)
PRIOR = float(y_bin.mean())


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


def _key(df: pd.DataFrame, k: str) -> pd.Series:
    return _route(df) if k == "route" else df[k].astype(str)


# --- encoding statistics, fit on TRAIN only -------------------------------------
_tr = train.copy()
_tr["route"] = _route(train)
_tr["_y"] = y_bin
TE_MAPS, CNT_MAPS = {}, {}
for k in TE_KEYS:
    g = _tr.groupby(k)["_y"].agg(["sum", "count"])
    TE_MAPS[k] = ((g["sum"] + PRIOR * TE_M) / (g["count"] + TE_M)).to_dict()
    CNT_MAPS[k] = g["count"].to_dict()


def _oof_te() -> pd.DataFrame:
    """Out-of-fold TE values for train rows (deterministic folds)."""
    folds = pd.Series(np.random.RandomState(SEED).randint(0, N_FOLDS, len(train)), index=train.index)
    out = {}
    for k in TE_KEYS:
        col = np.full(len(train), PRIOR)
        for f in range(N_FOLDS):
            tr_part = folds != f
            g = _tr.loc[tr_part].groupby(k)["_y"].agg(["sum", "count"])
            m = ((g["sum"] + PRIOR * TE_M) / (g["count"] + TE_M)).to_dict()
            idx = (folds == f).to_numpy()
            col[idx] = _key(train[idx], k).map(m).fillna(PRIOR).to_numpy()
        out[f"te_{k}"] = col
    return pd.DataFrame(out, index=train.index)


OOF_TE = _oof_te()
TE_COLS = [f"te_{k}" for k in TE_KEYS]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    # scheduled departure time (hhmm) -> time of day
    dt = _num(df["DepTime"])
    hour = (dt // 100).clip(0, 23)
    minute = dt - hour * 100
    tod = (hour * 60 + minute) / 1440.0
    X["tod_sin"] = np.sin(2 * np.pi * tod)
    X["tod_cos"] = np.cos(2 * np.pi * tod)
    X["hour"] = hour
    X["red_eye"] = ((hour >= 21) | (hour <= 5)).astype(np.int8)
    # calendar
    month = _num(df["Month"])
    dom = _num(df["DayofMonth"])
    dow = _num(df["DayOfWeek"])
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    X["dom"] = dom
    X["is_month_end"] = (dom >= 28).astype(np.int8)
    # distance
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["distance"] = dist
    X["log_dist"] = np.log1p(dist)
    # target encodings + counts (statistics fit on train only)
    for k in TE_KEYS:
        key = _key(df, k)
        X[f"te_{k}"] = key.map(TE_MAPS[k]).fillna(PRIOR)
        X[f"cnt_{k}"] = np.log1p(key.map(CNT_MAPS[k]).fillna(0.0))
    # categoricals (unseen levels -> NaN)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=CAT_LEVELS[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def make_model(n_estimators: int, **kw) -> xgb.XGBClassifier:
    params = dict(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=10,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    params.update(kw)
    return xgb.XGBClassifier(**params)


t0 = time.time()
# early stopping on a random 20% slice of train to find the best number of trees
rng = np.random.RandomState(SEED)
val_mask = rng.rand(len(train)) < 0.2
X_es = prepare(train[~val_mask])
X_es_val = prepare(train[val_mask])
es_model = make_model(4000, early_stopping_rounds=100)
es_model.fit(X_es, to_y(train[~val_mask]), eval_set=[(X_es_val, to_y(train[val_mask]))], verbose=False)
best_iter = int(es_model.best_iteration) + 1
print(f"ES best iteration: {best_iter}, val AUC: {es_model.best_score:.4f}")

# final fit on all train rows; TE columns replaced by out-of-fold values (no self-leakage)
X_tr = prepare(train)
X_tr[TE_COLS] = OOF_TE[TE_COLS].to_numpy()
model = make_model(best_iter)
model.fit(X_tr, to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
