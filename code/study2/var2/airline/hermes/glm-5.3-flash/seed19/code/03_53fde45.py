"""XGBoost binary classifier for airline delay. Only file the agent edits.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Architecture: an ensemble of XGBoost models that differ by which scheduled-departure-time
interaction categorical they see (Origin x hour / Carrier x half-hour / route x quarter-hour, ...)
plus small hyperparameter diversity. Averaging their probabilities is what drives AUC.
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
    raise ValueError(grain)


# interaction columns used by the ensemble members: grain -> tags
_MEMBER_SPECS = [
    ("hour", ("Org",), {}),
    ("half", ("Car",), dict(n_estimators=200)),
    ("half", ("Car", "Org"), dict(max_depth=7)),
    ("quarter", ("Org", "Dest"), {}),
    ("quarter", ("Car",), dict(max_depth=7)),
    ("quarter", ("Car", "Org"), dict(max_depth=7)),
    ("half", ("Car",), dict(n_estimators=300, learning_rate=0.033)),
    ("half", ("Org", "Dest"), dict(max_depth=8)),
]
_TAG_COL = {"Org": "Origin", "Dest": "Dest", "Car": "UniqueCarrier"}
_COMBOS = sorted({(g, tags) for g, tags, _ in _MEMBER_SPECS})
_INT_LEVELS = {}  # (grain, tags) -> categories from train only


def _int_name(grain: str, tags) -> str:
    return f"I_{grain}_{'_'.join(tags)}"


for g, tags in _COMBOS:
    parts = [train[_TAG_COL[t]].astype(str) for t in tags] + [_timekey(train, g)]
    key = _int_name(g, tags)
    _INT_LEVELS[key] = pd.Index(pd.concat(parts, axis=1).agg("_".join, axis=1).unique())


# --- feature engineering (everything inside prepare) ---------------------------
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=ID_COLS + [TARGET], errors="ignore")
    X = pd.DataFrame(index=df.index)
    # calendar as numeric (beats categorical on time-separated eval)
    X["Month"] = _num(df["Month"])
    X["DayOfWeek"] = _num(df["DayOfWeek"])
    X["DayofMonth"] = _num(df["DayofMonth"])
    # IDs as categoricals with train-fitted levels (unseen -> NaN)
    for c in ("UniqueCarrier", "Origin", "Dest"):
        X[c] = pd.Categorical(df[c].astype(str), categories=TR_LEVELS[c])
    # time-of-day features
    dt = pd.to_numeric(df["DepTime"], errors="coerce").astype(float)
    X["DepTime"] = dt
    X["DepHour"] = (dt // 100) % 24
    X["DepAbs"] = (dt // 100) * 60 + dt % 100
    X["LateNight"] = ((dt >= 1900) | (dt < 600)).astype(float)
    ang = 2 * np.pi * X["DepAbs"] / 1440.0
    X["DepSin"], X["DepCos"] = np.sin(ang), np.cos(ang)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").astype(float)
    # interaction categoricals (levels fitted on train only)
    for g, tags in _COMBOS:
        key = _int_name(g, tags)
        parts = [df[_TAG_COL[t]].astype(str) for t in tags] + [_timekey(df, g)]
        col = pd.concat(parts, axis=1).agg("_".join, axis=1)
        X[key] = pd.Categorical(col, categories=_INT_LEVELS[key])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: ensemble over interaction variants ---------------------------------
BASE_PARAMS = dict(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
KEYS = [_int_name(g, tags) for g, tags in _COMBOS]


def fit_ensemble():
    Xfull = prepare(train)
    base_cols = [c for c in Xfull.columns if c not in KEYS]
    models = []
    for g, tags, kw in _MEMBER_SPECS:
        cols = base_cols + [_int_name(g, tags)]
        p = dict(BASE_PARAMS)
        p.update(kw)
        m = xgb.XGBClassifier(**p)
        m.fit(Xfull[cols], to_y(train))
        models.append((m, cols))
    return models


t0 = time.time()
MODELS = fit_ensemble()
print(f"Training time ({len(MODELS)} members): {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xfull = prepare(df)
    ps = [m.predict_proba(Xfull[cols])[:, 1] for m, cols in MODELS]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
