"""XGBoost binary classifier for airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

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
N_MEMBERS = 10
SEEDS = (42, 7, 123, 2024, 99, 555, 314, 2718, 161, 777)

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _dep_hour(df: pd.DataFrame) -> pd.Series:
    return (_to_int(df["DepTime"]) // 100).clip(0, 27).astype(int)


# carrier x departure-hour interaction levels, fit on train only
CXH_LEVELS = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_" + _dep_hour(train).astype(str)).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dep = _to_int(df["DepTime"])
    hour = (dep // 100).astype(float)
    dep_min = hour * 60 + (dep % 100).astype(float)
    X["dep_hour"] = hour
    X["dep_min_mod"] = dep_min % 1440
    ang = 2 * np.pi * X["dep_min_mod"] / 1440.0
    X["dep_sin"], X["dep_cos"] = np.sin(ang), np.cos(ang)
    dow = _to_int(df["DayOfWeek"].str.replace("c-", "", regex=False))
    mon = _to_int(df["Month"].str.replace("c-", "", regex=False))
    dom = _to_int(df["DayofMonth"].str.replace("c-", "", regex=False))
    doy = (mon - 1) * 31 + dom
    for name, v, period in (("dow", dow, 7.0), ("doy", doy, 372.0)):
        a = 2 * np.pi * v / period
        X[f"{name}_sin"], X[f"{name}_cos"] = np.sin(a), np.cos(a)
    X["distance_log"] = np.log1p(_to_int(df["Distance"]))
    X["carrier_x_hour"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "_" + _dep_hour(df).astype(str), categories=CXH_LEVELS)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    max_depth=16,
    learning_rate=0.03,
    subsample=0.6,
    colsample_bytree=0.3,
    reg_alpha=0.25,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_train, y_train = prepare(train), to_y(train)
X_eval, y_eval = prepare(evald), to_y(evald)

# ensemble: first members are logistic learners, the rest squared-error regressors on the 0/1 label
# (L2 members are weaker alone but decorrelate the ensemble; higher lr keeps them cheap)
L2_LR = 0.05
members = []
fit_times = []
for i, seed in enumerate(SEEDS[:N_MEMBERS]):
    ts = time.time()
    if i < 2:
        m = xgb.XGBClassifier(n_estimators=3000, early_stopping_rounds=50, random_state=seed, **PARAMS)
        m.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
        members.append((m, int(m.best_iteration) + 1, True))
    else:
        p2 = dict(PARAMS)
        p2["learning_rate"] = L2_LR
        m = xgb.XGBRegressor(n_estimators=3000, early_stopping_rounds=50,
                             objective="reg:squarederror", random_state=seed, **p2)
        m.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
        members.append((m, int(m.best_iteration) + 1, False))
    fit_times.append(round(time.time() - ts, 1))
print(f"rounds: {[r for _, r, _ in members]}")
print(f"fit_times: {fit_times}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    P = []
    for m, r, is_clf in members:
        p = m.predict_proba(X, iteration_range=(0, r))[:, 1] if is_clf else m.predict(X, iteration_range=(0, r))
        P.append(np.clip(p, 0.0, 1.0))
    return np.mean(P, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(y_eval, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
