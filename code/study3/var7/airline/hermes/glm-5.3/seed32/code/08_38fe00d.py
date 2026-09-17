"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     ALL feature engineering lives inside prepare() so predict_proba reproduces it on the hidden holdout.
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
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
HOUR_LEVELS = pd.Index(range(0, 28))
_tdep = pd.to_numeric(train["DepTime"], errors="coerce")
_tour = train["Origin"].astype(str)
_tdest = train["Dest"].astype(str)
_thour = (_tdep // 100).astype("Int64")
OH_LEVELS = pd.Index(sorted((_tour + "@" + _thour.astype(str)).unique()))
CH_LEVELS = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "@" + _thour.astype(str)).unique()))
cnt_origin_hour = (_tour + "@" + _thour.astype(str)).value_counts()
cnt_dest_hour = (_tdest + "@" + _thour.astype(str)).value_counts()
cnt_route = (_tour + "_" + _tdest).value_counts()


def _cnum(s: pd.Series) -> pd.Series:
    """'c-7' -> 7.0 (NaN-safe)."""
    return pd.to_numeric(s.astype("string").str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    # numeric calendar features
    month = _cnum(df["Month"])
    day = _cnum(df["DayofMonth"])
    dow = _cnum(df["DayOfWeek"])
    X["month"] = month
    X["day"] = day
    X["dow"] = dow
    # time of day
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0.0)
    hh = (dep // 100).astype(int)
    mm = (dep % 100).astype(int)
    mins = hh * 60 + mm  # minutes past midnight
    X["dep_time"] = dep
    X["mins"] = mins
    X["sin_mins"] = np.sin(2 * np.pi * mins / 1440.0)
    X["cos_mins"] = np.cos(2 * np.pi * mins / 1440.0)
    X["is_weekend"] = (dow >= 6).astype(int)
    # distance
    dist = pd.to_numeric(df["Distance"], errors="coerce").fillna(-1.0)
    X["distance"] = dist
    X["dist_log"] = np.log1p(dist.clip(lower=0))
    # categoricals
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    # hour of day as categorical + origin/dest congestion counts from the TRAIN schedule
    hour = (dep // 100).astype("Int64")
    X["hour_cat"] = pd.Categorical(hour, categories=HOUR_LEVELS)
    orig = df["Origin"].astype(str)
    dest = df["Dest"].astype(str)
    hs = hour.astype(str)
    X["oh"] = pd.Categorical(orig + "@" + hs, categories=OH_LEVELS)
    X["ch"] = pd.Categorical(df["UniqueCarrier"].astype(str) + "@" + hs, categories=CH_LEVELS)
    X["fph_origin"] = (orig + "@" + hs).map(cnt_origin_hour).fillna(0).astype(float)
    X["fpd_dest"] = (dest + "@" + hs).map(cnt_dest_hour).fillna(0).astype(float)
    X["route_cnt"] = (orig + "_" + dest).map(cnt_route).fillna(0).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=1800,
    max_depth=8,
    learning_rate=0.02,
    colsample_bytree=0.7,
    colsample_bylevel=0.7,
    reg_alpha=1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_tr = prepare(train)
y_tr = to_y(train)
models = []
for s, cs in [(1, 0.5), (2, 0.6), (3, 0.7), (4, 0.8)]:
    m = xgb.XGBClassifier(**{**PARAMS, "colsample_bytree": cs, "random_state": s})
    m.fit(X_tr, y_tr)
    models.append(m)
model = models
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
