"""XGBoost binary classifier for the airline task (dep_delayed_15min).

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. A module-level `predict_proba(df)` maps a raw DataFrame (same columns, target optional) to P(positive).
  3. ALL feature engineering lives in prepare(); every statistic (target/frequency encodings, category
     levels) is fitted on data/train.csv only, never on the frame passed in.

What matters here: train is 2005, eval/holdout are 2006, so a model that fits 2005 well (OOF AUC ~0.75)
only reaches ~0.73 on 2006. The design is therefore deliberately low-variance: shallow trees, strong L1
sparsity (reg_alpha), smoothed target encodings, and a small ensemble of diverse members.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- base columns --------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]     # Month/DayofMonth/DayOfWeek/carrier/origin/dest
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

Y_TRAIN = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(Y_TRAIN.mean())

# --- keys for target encoding (pure functions of the raw row) ------------------
def add_keys(df: pd.DataFrame) -> dict:
    """Raw -> string keys for target/frequency encoding (no train statistics involved)."""
    dep = df["DepTime"].astype(float)
    dep = dep.where(dep < 2400, dep - 2400.0)          # post-midnight departures fold back
    hour = (dep // 100.0).astype(int)
    hour_s = hour.astype(str)
    minute5 = ((dep - hour * 100.0) // 5.0).astype(int).astype(str)
    origin, dest = df["Origin"].astype(str), df["Dest"].astype(str)
    return {
        "Origin": origin,
        "Dest": dest,
        "UniqueCarrier": df["UniqueCarrier"].astype(str),
        "Origin_hour": origin + "|" + hour_s,
        "Origin_min5": origin + "|" + hour_s + "|" + minute5,
        "Hour": hour_s,
    }


# --- train-fitted statistics: smoothed target encoding + frequency -------------
TE_SPEC = {"Origin": 50.0, "Dest": 50.0, "UniqueCarrier": 50.0,
           "Origin_hour": 30.0, "Origin_min5": 100.0}
TE_COLS = list(TE_SPEC)
CNT_EXTRA = ["Hour"]                                  # frequency features without a target encoding
_KEYS = add_keys(train)


def _te_stats(keys, y, k):
    s = pd.Series(y, index=keys.index).groupby(keys.to_numpy()).agg(["sum", "count"])
    return ((s["sum"] + PRIOR * k) / (s["count"] + k)).to_dict()


te_maps, oof_te, cnt_maps = {}, {}, {}
for _c, _k in TE_SPEC.items():
    _ks = _KEYS[_c]
    te_maps[_c] = _te_stats(_ks, Y_TRAIN, _k)
    cnt_maps[_c] = _ks.value_counts()
    _oof = np.full(len(train), PRIOR)
    for _tr, _va in KFold(n_splits=5, shuffle=True, random_state=SEED).split(train):
        _m = _te_stats(_ks.iloc[_tr], Y_TRAIN[_tr], _k)
        _oof[_va] = _ks.iloc[_va].map(_m).fillna(PRIOR).to_numpy()
    oof_te[_c] = _oof                                  # out-of-fold values for the training rows
for _c in CNT_EXTRA:
    cnt_maps["cnt_" + _c] = _KEYS[_c].value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataframe -> model matrix. Uses only train-fitted statistics."""
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])       # unseen levels -> NaN
    dep = df["DepTime"].astype(float)
    dep = dep.where(dep < 2400, dep - 2400.0)
    hour = dep // 100.0
    tod = hour * 60.0 + (dep - hour * 100.0)
    mon = df["Month"].astype(str).str.slice(2).astype(float)
    dow = df["DayOfWeek"].astype(str).str.slice(2).astype(float)
    X["dep_hour"] = hour.to_numpy()
    X["dep_minute"] = (dep - hour * 100.0).to_numpy()
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0).to_numpy()
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0).to_numpy()
    X["month_sin"] = np.sin(2 * np.pi * mon / 12.0).to_numpy()
    X["month_cos"] = np.cos(2 * np.pi * mon / 12.0).to_numpy()
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7.0).to_numpy()
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7.0).to_numpy()
    keys = add_keys(df)
    for c in TE_COLS:
        X[f"te_{c}"] = keys[c].map(te_maps[c]).fillna(PRIOR).to_numpy()
        X[f"cnt_{c}"] = keys[c].map(cnt_maps[c]).fillna(0).to_numpy()
    for c in CNT_EXTRA:
        X["cnt_" + c] = keys[c].map(cnt_maps["cnt_" + c]).fillna(0).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Xtr = prepare(train)
for c in TE_COLS:                                      # training rows use out-of-fold encodings
    Xtr[f"te_{c}"] = oof_te[c]
ytr, Xev, yev = to_y(train), prepare(evald), to_y(evald)

# --- model: small ensemble of diverse L1-regularized XGBoost members -----------
BASE = dict(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS,
            learning_rate=0.02, reg_alpha=10.0)
MEMBERS = [
    dict(max_depth=6, n_estimators=1000),
    dict(max_depth=6, n_estimators=1000, reg_alpha=12.0),
    dict(max_depth=5, n_estimators=1500, reg_alpha=12.0),
    dict(max_depth=7, n_estimators=800),
]

t0 = time.time()
members = []
for _i, _p in enumerate(MEMBERS):
    _m = xgb.XGBClassifier(random_state=SEED + _i, **{**BASE, **_p})
    _m.fit(Xtr, ytr)
    members.append(_m)
    print(f"member{_i} {str(_p):52s} eval_auc={roc_auc_score(yev, _m.predict_proba(Xev)[:, 1]):.4f}")
print(f"Training time: {time.time() - t0:.1f}s ({len(members)} members)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in members], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
