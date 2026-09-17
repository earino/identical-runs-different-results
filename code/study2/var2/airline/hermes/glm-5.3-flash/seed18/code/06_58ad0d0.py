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
cat_cols = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    # decode c-<n> strings to integer ordinals
    for c in ["Month", "DayofMonth", "DayOfWeek"]:
        X[c] = df[c].astype(str).str.replace("c-", "", regex=False).astype(float)
    # scheduled departure time: hour-of-day is the dominant delay driver; add cyclical encoding
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dt // 100).fillna(0.0)
    minute = (dt % 100).fillna(0.0)
    X["DepHour"] = hour
    ang = 2.0 * np.pi * (hour * 60.0 + minute) / 1440.0
    X["DepSin"] = np.sin(ang)
    X["DepCos"] = np.cos(ang)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").fillna(0.0)
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


y_all = to_y(train)

# --- congestion / schedule-density features (fit on train only) ----------------
_tr = train.copy()
_tr["_depmin"] = pd.to_numeric(_tr["DepTime"], errors="coerce").fillna(0.0)
_tr["_hr"] = (_tr["_depmin"] // 100).astype(int)

CNT = {}
for name, keys in [
    ("n_origin_hr", _tr["Origin"].astype(str) + "|" + _tr["_hr"].astype(str)),
    ("n_dest_hr", _tr["Dest"].astype(str) + "|" + _tr["_hr"].astype(str)),
    ("n_carrier_hr", _tr["UniqueCarrier"].astype(str) + "|" + _tr["_hr"].astype(str)),
    ("n_carrier_month", _tr["UniqueCarrier"].astype(str) + "|" + _tr["Month"].astype(str)),
    ("n_origin_month", _tr["Origin"].astype(str) + "|" + _tr["Month"].astype(str)),
    ("n_dest_month", _tr["Dest"].astype(str) + "|" + _tr["Month"].astype(str)),
    ("n_origin", _tr["Origin"].astype(str)),
    ("n_dest", _tr["Dest"].astype(str)),
    ("n_route", _tr["Origin"].astype(str) + "|" + _tr["Dest"].astype(str)),
    ("n_carrier", _tr["UniqueCarrier"].astype(str)),
]:
    CNT[name] = keys.value_counts().to_dict()

KEYS = [
    ("n_origin_hr", lambda d, x: d["Origin"].astype(str) + "|" + x["DepHour"].astype(int).astype(str)),
    ("n_dest_hr", lambda d, x: d["Dest"].astype(str) + "|" + x["DepHour"].astype(int).astype(str)),
    ("n_carrier_hr", lambda d, x: d["UniqueCarrier"].astype(str) + "|" + x["DepHour"].astype(int).astype(str)),
    ("n_carrier_month", lambda d, x: d["UniqueCarrier"].astype(str) + "|" + d["Month"].astype(str)),
    ("n_origin_month", lambda d, x: d["Origin"].astype(str) + "|" + d["Month"].astype(str)),
    ("n_dest_month", lambda d, x: d["Dest"].astype(str) + "|" + d["Month"].astype(str)),
    ("n_origin", lambda d, x: d["Origin"].astype(str)),
    ("n_dest", lambda d, x: d["Dest"].astype(str)),
    ("n_route", lambda d, x: d["Origin"].astype(str) + "|" + d["Dest"].astype(str)),
    ("n_carrier", lambda d, x: d["UniqueCarrier"].astype(str)),
]


def add_congestion(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    for name, fn in KEYS:
        k = fn(df, X).to_numpy()
        X[name] = np.log1p(pd.Series([CNT[name].get(ki, 0.0) for ki in k], index=df.index).to_numpy())
    # relative busyness: joint / marginal (counts are log1p; exp of the difference recovers the ratio)
    with np.errstate(invalid="ignore", over="ignore"):
        for j, m, name in [("n_origin_hr", "n_origin", "sh_origin_hr"),
                           ("n_dest_hr", "n_dest", "sh_dest_hr"),
                           ("n_carrier_hr", "n_carrier", "sh_carrier_hr")]:
            ratio = np.exp(X[j].to_numpy() - X[m].to_numpy())
            ratio[~np.isfinite(ratio)] = 0.0
            X[name] = ratio
    return X


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=50,
    eval_metric="auc",
)

y_all = to_y(train)
X_all = prepare(train)
X_all = add_congestion(train, X_all)

# early stopping on a time-like split (last 20% of train)
n_val = int(0.2 * len(X_all))
X_fit, y_fit = X_all.iloc[:-n_val], y_all[:-n_val]
X_val, y_val = X_all.iloc[-n_val:], y_all[-n_val:]

t0 = time.time()
es_model = xgb.XGBClassifier(**PARAMS)
es_model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
best_it = int(getattr(es_model, "best_iteration", PARAMS["n_estimators"] - 1)) + 1
print(f"Early-stop fit: {time.time() - t0:.1f}s, best_iteration={best_it}")
del es_model

# refit on ALL training rows with the chosen iteration count (+10% for the extra 25% data)
N_EST = min(int(best_it * 1.1) + 1, PARAMS["n_estimators"])
K = 5
rng = np.random.RandomState(SEED)
perm = rng.permutation(len(X_all))
models = []
for k in range(K):
    t0 = time.time()
    hold = perm[k::K]
    mask = np.ones(len(X_all), dtype=bool)
    mask[hold] = False
    m = xgb.XGBClassifier(**{k2: v for k2, v in PARAMS.items() if k2 != "early_stopping_rounds"})
    m.set_params(n_estimators=N_EST, random_state=SEED + k)
    m.fit(X_all.iloc[mask], y_all[mask])
    print(f"Bag fit {k + 1}/{K} (n={mask.sum()}): {time.time() - t0:.1f}s")
    models.append(m)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    X = add_congestion(df, X)
    p = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return p


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
