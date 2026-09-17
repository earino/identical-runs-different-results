"""XGBoost binary classifier for airline delay. Only file the agent edits.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Architecture: one shared engineered feature frame per (drop_month variant); an ensemble of
XGBoost members that differ by which scheduled-departure-time interaction categorical they
see (Carrier x half-hour, Carrier x Origin x 10-min, Origin x hour, route x quarter-hour, ...)
plus depth/rounds/colsample diversity, mild recency weighting (drift adaptation), and mean
probability blending.
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

# --- stats fitted on training data only ---------------------------------------
CATS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
TR_LEVELS = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CATS}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.replace("c-", "", regex=False), errors="coerce").astype(float)


def _timekey(df: pd.DataFrame, grain: str) -> pd.Series:
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hh = (dt // 100) % 24
    mm = dt % 100
    if grain == "hour":
        return hh.astype(int).astype(str)
    if grain == "half":
        return (hh * 2 + (mm // 30)).astype(int).astype(str)
    if grain == "quarter":
        return (hh * 4 + (mm // 15)).astype(int).astype(str)
    if grain == "min10":
        return (hh * 6 + (mm // 10)).astype(int).astype(str)
    raise ValueError(grain)


_TAG_COL = {"Origin": "Origin", "Dest": "Dest", "UniqueCarrier": "UniqueCarrier"}

# interaction columns in the shared frame: (tag tuple, grain)
_INT_SPECS = [
    (("UniqueCarrier",), "half"),
    (("UniqueCarrier",), "quarter"),
    (("UniqueCarrier", "Origin"), "min10"),
    (("UniqueCarrier", "Origin"), "quarter"),
    (("Origin",), "hour"),
    (("Origin", "Dest"), "quarter"),
]
_INT_LEVELS = {}
for tags, grain in _INT_SPECS:
    parts = [train[_TAG_COL[c]].astype(str) for c in tags] + [_timekey(train, grain)]
    name = "INT_" + "_".join(tags) + f"_{grain}"
    _INT_LEVELS[name] = pd.Index(pd.concat(parts, axis=1).agg("_".join, axis=1).unique())
INT_KEYS = list(_INT_LEVELS)

# ensemble members: (interaction column or None, params, use recency weights)
_T600 = dict(n_estimators=600, learning_rate=0.02, subsample=0.8, max_depth=7, random_state=8)
_SMALL = dict(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42)
_MEMBER_SPECS = [
    ("INT_UniqueCarrier_half", {**_T600, "colsample_bynode": 0.8}, True),
    ("INT_UniqueCarrier_Origin_min10", {**_SMALL, "max_depth": 7}, False),
    ("INT_UniqueCarrier_quarter", dict(_T600), True),
    (None, dict(_SMALL), True),
    ("INT_UniqueCarrier_half", dict(n_estimators=1000, learning_rate=0.012, subsample=0.8, max_depth=7, random_state=8), True),
    ("INT_Origin_Dest_quarter", dict(_SMALL), False),
    ("INT_UniqueCarrier_half", dict(_T600), True),
    ("INT_UniqueCarrier_Origin_quarter", {**_SMALL, "max_depth": 7}, False),
]


# --- feature engineering (everything inside prepare) ---------------------------
def prepare(df: pd.DataFrame, drop_month: bool = True) -> pd.DataFrame:
    df = df.drop(columns=ID_COLS + [TARGET], errors="ignore")
    X = pd.DataFrame(index=df.index)
    if not drop_month:
        X["Month"] = _num(df["Month"])
    X["DayOfWeek"] = _num(df["DayOfWeek"])
    X["DayofMonth"] = _num(df["DayofMonth"])
    for c in ("UniqueCarrier", "Origin", "Dest"):
        X[c] = pd.Categorical(df[c].astype(str), categories=TR_LEVELS[c])
    dt = pd.to_numeric(df["DepTime"], errors="coerce").astype(float)
    X["DepTime"] = dt
    X["DepHour"] = (dt // 100) % 24
    X["DepAbs"] = (dt // 100) * 60 + dt % 100
    X["LateNight"] = ((dt >= 1900) | (dt < 600)).astype(float)
    ang = 2 * np.pi * X["DepAbs"] / 1440.0
    X["DepSin"], X["DepCos"] = np.sin(ang), np.cos(ang)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").astype(float)
    for tags, grain in _INT_SPECS:
        name = "INT_" + "_".join(tags) + f"_{grain}"
        parts = [df[_TAG_COL[c]].astype(str) for c in tags] + [_timekey(df, grain)]
        X[name] = pd.Categorical(pd.concat(parts, axis=1).agg("_".join, axis=1),
                                 categories=_INT_LEVELS[name])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


_MONTH_WEIGHT = np.exp(0.5 * (_num(train["Month"]).to_numpy() - 1) / 11)


def fit_ensemble():
    Xfull = prepare(train, drop_month=True)
    base_cols = [c for c in Xfull.columns if c not in INT_KEYS]
    models = []
    for icol, kw, use_w in _MEMBER_SPECS:
        cols = base_cols + ([icol] if icol else [])
        p = dict(kw)
        m = xgb.XGBClassifier(**p)
        m.fit(Xfull[cols], to_y(train), sample_weight=_MONTH_WEIGHT if use_w else None)
        models.append((m, icol))
    return models


t0 = time.time()
MODELS = fit_ensemble()
print(f"Training time ({len(MODELS)} members): {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xfull = prepare(df, drop_month=True)
    base_cols = [c for c in Xfull.columns if c not in INT_KEYS]
    ps = []
    for m, icol in MODELS:
        cols = base_cols + ([icol] if icol else [])
        ps.append(m.predict_proba(Xfull[cols])[:, 1])
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
