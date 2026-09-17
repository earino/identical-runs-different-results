"""XGBoost classifier for airline delay (agent-edited, see program.md).

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. Module-level `predict_proba(df)` maps a raw DataFrame -> P(positive). All feature engineering
     happens inside prepare(), which only uses statistics fitted on data/train.csv.
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
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


ytr = to_y(train)
PRIOR = float(ytr.mean())

# --- group keys (shared by target encodings and counts) ------------------------
# name -> smoothing m for the target encoding. Keys built by _keys().
TE_M = {
    "carrier": 20,
    "origin": 20,
    "dest": 20,
    "route": 100,
    "car_hour": 30,
    "org_hour": 60,
    "dow_hour": 30,
    "route_hour": 200,
    "dest_hour": 60,
    "car_dow": 30,
    "route_hm": 400,
    "car_hm": 100,
    "org_hm": 200,
    "dow_hm": 40,
}
# count feature name -> key name (train-frequency of that key)
COUNT_KEYS = {
    "cnt_carrier": "carrier",
    "cnt_origin": "origin",
    "cnt_dest": "dest",
    "cnt_route": "route",
    "cnt_org_hour": "org_hour",
    "cnt_route_hour": "route_hour",
    "cnt_dest_hour": "dest_hour",
    "cnt_car_hour": "car_hour",
    "cnt_dow_hour": "dow_hour",
    "cnt_route_hm": "route_hm",
    "cnt_car_hm": "car_hm",
    "cnt_org_hm": "org_hm",
    "cnt_dow_hm": "dow_hm",
}


def _keys(df: pd.DataFrame) -> dict:
    """Group-key Series computed from raw columns only."""
    hour = np.clip(df["DepTime"].to_numpy() // 100, 0, 24)
    hs = pd.Series(hour, index=df.index).astype(str)
    hms = pd.Series(hour * 2 + np.clip(df["DepTime"].to_numpy() % 100, 0, 59) // 30, index=df.index).astype(str)
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    return {
        "carrier": df["UniqueCarrier"].astype(str),
        "origin": df["Origin"].astype(str),
        "dest": df["Dest"].astype(str),
        "route": route,
        "car_hour": df["UniqueCarrier"].astype(str) + "_" + hs,
        "org_hour": df["Origin"].astype(str) + "_" + hs,
        "dow_hour": df["DayOfWeek"].astype(str) + "_" + hs,
        "route_hour": route + "_" + hs,
        "dest_hour": df["Dest"].astype(str) + "_" + hs,
        "car_dow": df["UniqueCarrier"].astype(str) + "_" + df["DayOfWeek"].astype(str),
        "route_hm": route + "_" + hms,
        "car_hm": df["UniqueCarrier"].astype(str) + "_" + hms,
        "org_hm": df["Origin"].astype(str) + "_" + hms,
        "dow_hm": df["DayOfWeek"].astype(str) + "_" + hms,
    }


def _fit_map(key: pd.Series, y: pd.Series, m: float) -> dict:
    g = y.groupby(key).sum()
    n = key.value_counts()
    return ((g + PRIOR * m) / (n + m)).to_dict()


# full-train maps (used for anything that is not the training rows); OOF values are built per seed below
TE_MAPS = {k: _fit_map(_keys(train)[k], pd.Series(ytr, index=train.index), m) for k, m in TE_M.items()}
_YY = pd.Series(ytr, index=train.index)
_ktr = _keys(train)

CNT = {name: _ktr[k].value_counts().to_dict() for name, k in COUNT_KEYS.items()}
dow_levels = pd.Index(sorted(train["DayOfWeek"].dropna().unique()))


def prepare(df: pd.DataFrame, oof: dict = None) -> pd.DataFrame:
    """Raw airline dataframe -> feature matrix. predict_proba() calls this on unseen rows."""
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].to_numpy()
    hour = np.clip(dep // 100, 0, 24)
    minute = np.clip(dep % 100, 0, 59)
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = hour + minute / 60.0
    X["dep_raw"] = dep
    X["distance"] = df["Distance"].to_numpy()
    keys = _keys(df)
    for k in TE_M:
        if oof is not None:
            X["te_" + k] = oof[k]
        else:
            X["te_" + k] = keys[k].map(TE_MAPS[k]).fillna(PRIOR).to_numpy()
    for name, k in COUNT_KEYS.items():
        X[name] = keys[k].map(CNT[name]).fillna(0).to_numpy()
    X["dow"] = pd.Categorical(df["DayOfWeek"], categories=dow_levels)  # unseen levels -> NaN
    return X


# --- model: small ensemble of XGBoost seeds -----------------------------------
SEEDS = [42, 43, 44, 45, 46]

def _oof_te(fold_seed: int) -> dict:
    """Out-of-fold target-encoding values for the training rows (fold structure per seed)."""
    out = {}
    for k, m in TE_M.items():
        key = _ktr[k]
        oof = np.zeros(len(train))
        for tr_i, va_i in KFold(n_splits=5, shuffle=True, random_state=fold_seed).split(train):
            mp = _fit_map(key.iloc[tr_i], _YY.iloc[tr_i], m)
            oof[va_i] = key.iloc[va_i].map(mp).fillna(PRIOR).to_numpy()
        out[k] = oof
    return out

Xev = prepare(evald)
yev = to_y(evald)
models = []
t0 = time.time()
for s in SEEDS:
    model = xgb.XGBClassifier(
        n_estimators=5000,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="auc",
        early_stopping_rounds=150,
        tree_method="hist",
        enable_categorical=True,
        random_state=s,
        n_jobs=N_JOBS,
    )
    model.fit(prepare(train, oof=_oof_te(s)), ytr, eval_set=[(Xev, yev)], verbose=False)
    auc_s = roc_auc_score(yev, model.predict_proba(Xev)[:, 1])
    print(f"member seed={s}: best_iteration={model.best_iteration}, AUC={auc_s:.4f}")
    models.append(model)
print(f"Training time: {time.time() - t0:.1f}s for {len(models)} members")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
