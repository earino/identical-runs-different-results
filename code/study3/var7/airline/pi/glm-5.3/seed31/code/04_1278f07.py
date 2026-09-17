"""XGBoost ensemble for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Findings so far (diagnostics):
- Temporal drift 2005(train) -> 2006(eval/holdout) dominates: deep/many-tree models fit 2005 noise and HURT.
- Month and DayofMonth are 2005-calendar noise; keeping DayOfWeek only is the best single feature set.
- Time-of-day (Hour/Minute/DepMinutes) is the strongest drift-robust signal.
- Route (Origin_Dest) and 2005-conditioned target encodings hurt.
- Averaging many feature-diverse, hyper-diverse XGB models beats any single model by ~+0.003.
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


def _daypart(df: pd.DataFrame) -> pd.Series:
    hour = (df["DepTime"].astype("int32") // 100).clip(0, 24)
    return pd.cut(hour, bins=[-1, 5, 11, 17, 25], labels=["night", "am", "pm", "eve"]).astype(str)


# categorical levels / part levels: statistics fit on TRAIN ONLY; unseen holdout levels -> NaN
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
_part_tr = _daypart(train)
for _c in ["UniqueCarrier", "Origin", "Dest"]:
    cat_levels[_c + "Part"] = pd.Index(sorted((train[_c].astype(str) + "_" + _part_tr).unique()))
cat_levels["DOWHour"] = pd.Index(
    sorted((train["DayOfWeek"].astype(str) + "_" + (train["DepTime"].astype("int32") // 100).clip(0, 24).astype(str)).unique())
)
freq_levels = {c: train[c].astype(str).value_counts() for c in ["UniqueCarrier", "Origin", "Dest"]}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here: predict_proba() calls this on unseen rows."""
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    dep = df["DepTime"].astype("int32")
    X["DepTime"] = dep
    hour = (dep // 100).clip(0, 24).astype("int16")
    X["Hour"] = hour
    X["Minute"] = (dep % 100).astype("int16")
    X["DepMinutes"] = hour * 60 + (dep % 100).astype("int16")
    X["Distance"] = df["Distance"].astype("float32")
    part = _daypart(df)
    for c in ["UniqueCarrier", "Origin", "Dest"]:
        X[c + "Part"] = pd.Categorical(df[c].astype(str) + "_" + part, categories=cat_levels[c + "Part"])
    X["DOWHour"] = pd.Categorical(df["DayOfWeek"].astype(str) + "_" + hour.astype(str), categories=cat_levels["DOWHour"])
    for c in ["UniqueCarrier", "Origin", "Dest"]:
        X[c + "Freq"] = df[c].astype(str).map(freq_levels[c]).fillna(0).astype("float32").pipe(np.log1p)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- ensemble members: (feature subset, xgboost params) ------------------------
# BASE keeps DayOfWeek but drops Month/DayofMonth (2005-calendar noise under drift).
BASE = ["DayOfWeek", "UniqueCarrier", "Origin", "Dest", "DepTime", "Hour", "Minute", "DepMinutes", "Distance"]
FULL = ["Month", "DayofMonth"] + BASE          # all raw columns
NODATE = [c for c in BASE if c != "DayOfWeek"]
PARTS = BASE + ["UniqueCarrierPart", "OriginPart", "DestPart"]
CPART = BASE + ["UniqueCarrierPart"]
FREQ = BASE + ["UniqueCarrierFreq", "OriginFreq", "DestFreq"]
DOWH = BASE + ["DOWHour"]

D4 = dict(n_estimators=300, max_depth=4, learning_rate=0.05, tree_method="hist",
          enable_categorical=True, random_state=SEED, n_jobs=N_JOBS)


def _cfg(**kw):
    p = dict(D4)
    p.update(kw)
    return p


MEMBERS = [
    (BASE, _cfg()),
    (BASE, _cfg(colsample_bynode=0.8)),
    (BASE, _cfg(subsample=0.9, colsample_bytree=0.7, random_state=7)),
    (BASE, _cfg(min_child_weight=20)),
    (BASE, _cfg(max_depth=3, learning_rate=0.1)),
    ([c for c in BASE if c != "Distance"], _cfg()),
    (NODATE, _cfg()),
    (NODATE, _cfg(colsample_bynode=0.8)),
    (FULL, _cfg()),
    (FULL, _cfg(max_depth=5, reg_lambda=5)),
    ([c for c in FULL if c != "Distance"], _cfg()),
    (BASE, _cfg(max_depth=5, reg_lambda=5)),
    (PARTS, _cfg()),
    (CPART, _cfg()),
    (FREQ, _cfg()),
    (DOWH, _cfg()),
    (PARTS + ["DOWHour"], _cfg()),
    (BASE, _cfg(grow_policy="lossguide", max_leaves=20, max_depth=0)),
    (BASE, _cfg(reg_alpha=1)),
    (BASE, _cfg(max_bin=48)),
    (BASE, _cfg(learning_rate=0.07, n_estimators=400)),
    (BASE, _cfg(max_depth=2, learning_rate=0.15, n_estimators=500)),
    ([c for c in BASE if c != "Origin"], _cfg()),
    ([c for c in BASE if c != "Dest"], _cfg()),
    (BASE + ["UniqueCarrierFreq", "OriginFreq", "DestFreq"], _cfg(max_depth=6, reg_lambda=10, min_child_weight=30)),
]


def fit_members(X_all: pd.DataFrame, y: np.ndarray):
    models = []
    for cols, cfg in MEMBERS:
        m = xgb.XGBClassifier(**cfg)
        m.fit(X_all[cols], y)
        models.append((cols, m))
    return models


t0 = time.time()
X_all = prepare(train)
y_all = to_y(train)
members = fit_members(X_all, y_all)
print(f"Training time: {time.time() - t0:.1f}s ({len(members)} members)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X[cols])[:, 1] for cols, m in members], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
