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

DATE_COLS = ["Month", "DayofMonth", "DayOfWeek"]
CAT_COLS = DATE_COLS + ["UniqueCarrier", "Origin", "Dest"]


def add_base(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X[DATE_COLS] = df[DATE_COLS].astype(str)
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    X["DepTime"] = dep
    hour = (dep // 100) % 24
    tod = hour * 60 + dep % 100
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440)
    mnum = pd.to_numeric(df["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dnum = pd.to_numeric(df["DayofMonth"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    doy = (mnum - 1) * 31 + dnum  # monotone proxy for day-of-year
    X["doy"] = doy
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    X["hour"] = hour
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["Distance"] = dist
    X["log_dist"] = np.log1p(dist)
    X["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    X["Origin"] = df["Origin"].astype(str)
    X["Dest"] = df["Dest"].astype(str)
    return X


_full = add_base(train)
cat_levels = {c: pd.Index(sorted(_full[c].unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = add_base(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
y_all = to_y(train)
X_all = prepare(train)

# pick n_estimators via a temporal split: fit months 1-10, validate months 11-12
mnum = pd.to_numeric(train["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
late = (mnum >= 11).to_numpy()
params = dict(
    n_estimators=600,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    colsample_bynode=0.3,
    max_bin=1024,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
probe = xgb.XGBClassifier(early_stopping_rounds=40, **params)
probe.fit(X_all[~late], y_all[~late], eval_set=[(X_all[late], y_all[late])], verbose=False)
BEST_ROUNDS = max(probe.best_iteration + 1, 50)
probe_auc = roc_auc_score(y_all[late], probe.predict_proba(X_all[late])[:, 1])
print(f"Temporal ES: best rounds={BEST_ROUNDS}, late-2005 AUC={probe_auc:.4f}")

N_MODELS = 20
models = []
t0 = time.time()
for k in range(N_MODELS):
    m = xgb.XGBClassifier(n_estimators=BEST_ROUNDS, random_state=SEED + 1000 * k, **{k2: v for k2, v in params.items() if k2 != "n_estimators"})
    m.fit(X_all, y_all)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({N_MODELS} models @ {BEST_ROUNDS}r)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
