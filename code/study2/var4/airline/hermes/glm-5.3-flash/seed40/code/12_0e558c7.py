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
from sklearn.model_selection import StratifiedKFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42
N_FOLDS = 5

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# category levels fit on TRAIN only; unseen levels in eval/holdout -> code -1 -> missing
CAT_COLS = ("UniqueCarrier", "Origin", "Dest")
CAT_LEVELS = {c: sorted(train[c].dropna().astype(str).unique().tolist()) for c in CAT_COLS}

# smoothed target encoding of Origin/Dest, fit on TRAIN only
N_train = float(len(train))
GLOB = float((train[TARGET] == POSITIVE).mean())
TE_ALPHA = 20.0
TE_MAPS = {}
for c in ("Origin", "Dest", "UniqueCarrier"):
    grp = train.groupby(train[c].astype(str))[TARGET].apply(lambda s: (s == POSITIVE).mean())
    cnt = train.groupby(train[c].astype(str)).size()
    TE_MAPS[c] = ((grp * cnt + GLOB * TE_ALPHA) / (cnt + TE_ALPHA)).astype(float)


def te_col(df: pd.DataFrame, c: str) -> pd.Series:
    return df[c].astype(str).map(TE_MAPS[c]).astype(float).fillna(GLOB)  # type: ignore[return-value]


# --- features -----------------------------------------------------------------
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here so predict_proba() applies it to unseen rows too."""
    X = pd.DataFrame(index=df.index)

    # Month / DayofMonth / DayOfWeek come as "c-<n>" strings -> numeric
    for c in ("DayofMonth", "DayOfWeek"):
        X[c] = pd.to_numeric(df[c].astype(str).str.lstrip("c"), errors="coerce")
    # Month intentionally NOT used as a feature: its delay-rate mapping shifts
    # year-over-year (train vs eval rates disagree), so it generalizes poorly.

    # DepTime: scheduled departure as hhmm integer (values like 1..959 lack a leading zero).
    s = df["DepTime"].fillna(0).astype("int64").astype(str).str.zfill(4)
    hh = pd.to_numeric(s.str[:2], errors="coerce")
    mm = pd.to_numeric(s.str[2:4], errors="coerce")
    hh = hh.where((hh >= 0) & (hh < 24), np.nan)
    mm = mm.where((mm >= 0) & (mm < 60), np.nan)
    mins = hh * 60 + mm
    X["dep_hour"] = hh
    X["dep_tod"] = np.floor(mins / 15)  # 0..95 quarter of day
    X["dep_sin"] = np.sin(2 * np.pi * mins / 1440)
    X["dep_cos"] = np.cos(2 * np.pi * mins / 1440)
    X["red_eye"] = np.where(hh.isna(), -1, ((hh >= 20) | (hh <= 5)).astype(float))
    # non-monotonic patterns: hour and weekday as categoricals
    X["dep_hour_cat"] = pd.Categorical(hh, categories=list(range(24)))
    X["dow_cat"] = pd.Categorical(X["DayOfWeek"], categories=list(range(1, 8)))

    X["Distance"] = df["Distance"].astype(float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=CAT_LEVELS[c])
    X["te_" + "Origin"] = te_col(df, "Origin")
    X["te_" + "Dest"] = te_col(df, "Dest")
    X["te_" + "UniqueCarrier"] = te_col(df, "UniqueCarrier")
    # dep_tod as a categorical too (96 quarters) — the hour-cat alone may be too coarse
    X["dep_tod_cat"] = pd.Categorical(X["dep_tod"], categories=list(range(96)))
    # day-of-month categorical: end-of-month / holiday travel days differ
    X["dom_cat"] = pd.Categorical(X["DayofMonth"], categories=list(range(1, 32)))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "max_depth": 12,
    "eta": 0.02,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 20,
    "lambda": 2.0,
    "alpha": 0.5,
    "seed": SEED,
    "nthread": N_JOBS,
}


def fit_model(X, y, params=None, num_boost_round=800):
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
    p = dict(PARAMS)
    if params:
        p.update(params)
    return xgb.train(p, dtrain, num_boost_round, verbose_eval=False)


def cv_score(X, y, params=None):
    """Time-ordered CV: for each fold, train on the past, validate on the future.
    Mirrors the 2005->2006 shift better than random K-fold."""
    aucs = []
    n = len(y)
    Xv = X.reset_index(drop=True)
    for k in range(N_FOLDS):
        cut = int(n * (0.5 + 0.1 * k))
        tr, va = np.arange(0, cut), np.arange(cut, min(cut + int(n * 0.1), n))
        if len(va) < 1000:
            continue
        m = fit_model(Xv.iloc[tr], y[tr], params=params)
        dp = xgb.DMatrix(Xv.iloc[va], enable_categorical=True)
        aucs.append(roc_auc_score(y[va], m.predict(dp)))
    return float(np.mean(aucs)), aucs


model = None
model2 = None


def fit_all():
    global model, model2
    t0 = time.time()
    X_train = prepare(train)
    y_train = to_y(train)

    model = fit_model(X_train, y_train)
    # second model on 2006-style robustness: stronger subsample/colsample, deeper leaves
    p2 = {"max_depth": 13, "subsample": 0.9, "colsample_bytree": 0.9,
          "min_child_weight": 50, "eta": 0.015, "lambda": 4.0, "alpha": 1.0}
    model2 = fit_model(X_train, y_train, params=p2)
    print(f"2-model training time: {time.time() - t0:.1f}s")


fit_all()


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    dp = xgb.DMatrix(prepare(df), enable_categorical=True)
    return 0.5 * model.predict(dp) + 0.5 * model2.predict(dp)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
