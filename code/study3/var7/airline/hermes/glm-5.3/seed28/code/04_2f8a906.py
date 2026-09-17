"""XGBoost binary classifier for airline delay prediction.

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

# --- feature engineering -------------------------------------------------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}


def _num(s):
    """'c-4' -> 4. Missing/unparseable -> NaN."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _hour(deptime):
    """CRS DepTime (hhmm int) -> fractional hour of day."""
    t = pd.to_numeric(deptime, errors="coerce").astype(float)
    hh = np.floor(t / 100.0)
    mm = t - hh * 100.0
    bad = (t < 0) | (t > 2400) | (mm > 59)
    return pd.Series(np.where(bad, np.nan, hh + mm / 60.0), index=deptime.index)


# carrier x hour-of-day interaction levels (train only)
HOUR_BINS = [-np.inf] + list(range(1, 24)) + [np.inf]
_hb_tr = pd.cut(_hour(train["DepTime"]), bins=HOUR_BINS, labels=False).astype(str)
CARRIER_HB_LEVELS = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "-" + _hb_tr).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = pd.to_numeric(df["DepTime"], errors="coerce").astype(float)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").astype(float)
    hr = _hour(df["DepTime"])
    mon = _num(df["Month"])
    dom = _num(df["DayofMonth"])
    dow = _num(df["DayOfWeek"])
    # time-of-day and season structure
    X["hour"] = hr
    X["sin_h"] = np.sin(2 * np.pi * hr / 24.0)
    X["cos_h"] = np.cos(2 * np.pi * hr / 24.0)
    X["minute"] = X["DepTime"] - np.floor(X["DepTime"] / 100.0) * 100.0
    X["mon_n"] = mon
    X["dom_n"] = dom
    X["dow_n"] = dow
    X["winter"] = (mon <= 2).astype(float)
    X["dec_jan"] = ((mon == 12) | (mon == 1)).astype(float)
    # numeric interactions
    X["log_dist"] = np.log1p(X["Distance"])
    X["dist_x_hr"] = X["Distance"] * hr
    # categoricals
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=CAT_LEVELS[c])
    # carrier x hour-of-day interaction (the schedule effect: when a carrier flies matters)
    hb = pd.cut(hr, bins=HOUR_BINS, labels=False).astype(str)
    X["carrier_hb"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "-" + hb, categories=CARRIER_HB_LEVELS
    )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# small bagged ensemble of XGBoost models: seeds + colsample/depth variation decorrelate members
ENSEMBLE = [
    dict(seed=42),
    dict(seed=7, colsample_bytree=0.6),
    dict(seed=2024, colsample_bytree=0.4),
    dict(seed=13, max_depth=8),
    dict(seed=99, max_depth=8, colsample_bytree=0.6),
    dict(seed=5, subsample=0.8, colsample_bytree=0.5),
]
_models = []
t0 = time.time()
Xtr_all = prepare(train)
ytr_all = to_y(train)
for spec in ENSEMBLE:
    params = dict(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        min_child_weight=1,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        random_state=spec["seed"],
        n_jobs=N_JOBS,
    )
    params.update({k: v for k, v in spec.items() if k != "seed"})
    m = xgb.XGBClassifier(**params)
    m.fit(Xtr_all, ytr_all)
    _models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(_models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in _models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
