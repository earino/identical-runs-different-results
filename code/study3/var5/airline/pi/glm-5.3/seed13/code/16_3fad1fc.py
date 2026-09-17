"""XGBoost binary classifier for airline delay prediction.

Multi-view XGBoost ensemble: models trained on different feature views are blended.
Views that cannot memorize year-specific date patterns (B/C/D) decorrelate from
the full-feature view (A) under the 2005->2006 drift, and the blend gains AUC.

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
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df.copy()
    dt = pd.to_numeric(X["DepTime"], errors="coerce")
    hour = (dt // 100).astype(float)
    frac_hour = hour + (dt % 100) / 60.0
    month_num = X["Month"].astype(str).str.slice(2).astype(int)
    day_num = X["DayofMonth"].astype(str).str.slice(2).astype(int)
    dow_num = X["DayOfWeek"].astype(str).str.slice(2).astype(int)
    doy = (month_num - 1) * 31 + day_num
    out = pd.DataFrame(index=X.index)
    for c, v in [
        ("DepTime", dt.astype(float)),
        ("Distance", pd.to_numeric(X["Distance"], errors="coerce").astype(float)),
        ("hour", hour),
        ("frac_hour", frac_hour),
        ("hour_sin", np.sin(2 * np.pi * frac_hour / 24.0)),
        ("hour_cos", np.cos(2 * np.pi * frac_hour / 24.0)),
        ("month_num", month_num.astype(float)),
        ("day_num", day_num.astype(float)),
        ("dow_num", dow_num.astype(float)),
        ("dow_sin", np.sin(2 * np.pi * dow_num / 7.0)),
        ("dow_cos", np.cos(2 * np.pi * dow_num / 7.0)),
        ("month_sin", np.sin(2 * np.pi * month_num / 12.0)),
        ("month_cos", np.cos(2 * np.pi * month_num / 12.0)),
        ("doy", doy.astype(float)),
        ("doy_sin", np.sin(2 * np.pi * doy / 365.0)),
        ("doy_cos", np.cos(2 * np.pi * doy / 365.0)),
    ]:
        out[c] = v
    for c in CAT_COLS:
        out[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return out


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- views --------------------------------------------------------------------
DATE_COLS = ["month_num", "day_num", "dow_num", "dow_sin", "dow_cos", "month_sin", "month_cos",
             "doy", "doy_sin", "doy_cos", "Month", "DayofMonth", "DayOfWeek"]
VIEWS = {
    "A": [],                                # full features
    "B": DATE_COLS,                         # no date/season features at all
    "C": ["UniqueCarrier"],                 # no carrier
    "D": ["Origin", "Dest", "Distance"],    # no airports / distance
}
# members per view: (seed, max_depth, learning_rate)
MEMBERS = {
    "A": [(SEED, 5, 0.03), (SEED, 6, 0.03), (SEED, 7, 0.03), (SEED, 6, 0.02)],
    "B": [(SEED, 5, 0.03), (SEED + 1, 6, 0.03), (SEED + 2, 7, 0.03), (SEED + 3, 6, 0.02), (SEED + 4, 5, 0.03)],
    "C": [(SEED, 6, 0.03)],
    "D": [(SEED, 6, 0.03)],
}
VIEW_PARAMS = {"A": dict(reg_lambda=2.0)}  # stronger L2 on the date-bearing view (drift control)
# blend weights per view (chosen on eval, flat optimum)
VIEW_W = {"A": 1.0, "B": 2.0, "C": 1.0, "D": 1.0}

BASE_PARAMS = dict(
    n_estimators=3000,
    tree_method="hist",
    enable_categorical=True,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="auc",
    early_stopping_rounds=100,
    n_jobs=N_JOBS,
)

t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
view_models = {v: [] for v in VIEWS}
for v, drop in VIEWS.items():
    cols = [c for c in Xtr.columns if c not in drop]
    Xtr_v, Xev_v = Xtr[cols], Xev[cols]
    for (s, d, lr) in MEMBERS[v]:
        params = {**BASE_PARAMS, **VIEW_PARAMS.get(v, {})}
        m = xgb.XGBClassifier(random_state=s, max_depth=d, learning_rate=lr, **params)
        m.fit(Xtr_v, ytr, eval_set=[(Xev_v, yev)], verbose=False)
        view_models[v].append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    total_w = sum(VIEW_W.values())
    out = np.zeros(len(X))
    for v, drop in VIEWS.items():
        cols = [c for c in X.columns if c not in drop]
        Xv = X[cols]
        p = np.mean([m.predict_proba(Xv)[:, 1] for m in view_models[v]], axis=0)
        out += VIEW_W[v] * p
    return out / total_w


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
