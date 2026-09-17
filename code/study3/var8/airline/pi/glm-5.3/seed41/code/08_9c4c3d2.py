"""XGBoost binary classifier — airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
N_MODELS = 3

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
# Month is deliberately dropped: exact-month patterns learned on 2005 do not transfer to 2006.
DATE_COLS = ["DayofMonth", "DayOfWeek"]  # stored as c-<n> strings
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _strip_c(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("c-", "", regex=False).astype(int)


# stable congestion proxies: flight counts per (Origin, hour) and per (Origin, day-of-week), fit on train only
_ohc = train.groupby([train["Origin"], train["DepTime"] // 100]).size()
_odc = train.groupby([train["Origin"], _strip_c(train["DayOfWeek"])]).size()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in DATE_COLS:
        X[c] = _strip_c(df[c])
    dep = df["DepTime"].astype(int)
    hour = dep // 100
    tod = hour * 60 + dep % 100  # minutes since midnight
    X["DepHour"] = hour
    X["HourCat"] = pd.Categorical(hour)
    X["TodMin"] = tod
    X["TodSin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["TodCos"] = np.cos(2 * np.pi * tod / 1440.0)
    X["Distance"] = df["Distance"].astype(float)
    X["OHCount"] = pd.MultiIndex.from_arrays([df["Origin"], hour]).map(_ohc).fillna(0).astype(float)
    X["ODCount"] = pd.MultiIndex.from_arrays([df["Origin"], X["DayOfWeek"]]).map(_odc).fillna(0).astype(float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: small bagged ensemble of deep, regularized XGB trees ----------------
X_train, y_train = prepare(train), to_y(train)

t0 = time.time()
models = []
for seed in range(1, N_MODELS + 1):
    m = xgb.XGBClassifier(
        n_estimators=450,
        max_depth=24,
        learning_rate=0.02,
        subsample=0.75,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({N_MODELS} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
