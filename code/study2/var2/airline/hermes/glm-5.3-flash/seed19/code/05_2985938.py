"""XGBoost binary classifier for airline delay. Only file the agent edits.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Architecture: an ensemble of XGBoost models differing by which scheduled-departure-time
interaction categorical they see (Carrier x half-hour, Origin x hour, Carrier x Origin x
10-minute slots, ...) with Month dropped from most members (calendar drift), row subsampling
and mild recency weighting for drift adaptation. Mean of member probabilities.
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


# ensemble members: (grain, tags, params, drop_month, recency_power or None)
_MEMBER_SPECS = [
    ("half", ("Car",), dict(n_estimators=600, learning_rate=0.02, subsample=0.8, random_state=8), True, 0.5),
    ("quarter", ("Org", "Dest"), {}, False, None),
    ("half", ("Car",), dict(n_estimators=200, max_depth=7, subsample=0.8, random_state=8), True, 0.5),
    ("min10", ("Car", "Org"), dict(max_depth=7), True, None),
    ("half", ("Car",), dict(n_estimators=200, max_depth=7, subsample=0.8, random_state=8), True, None),
    ("hour", ("Org",), {}, True, None),
]
_TAG_COL = {"Org": "Origin", "Dest": "Dest", "Car": "UniqueCarrier"}
_COMBOS = sorted({(g, tags) for g, tags, *_ in _MEMBER_SPECS})
_INT_LEVELS = {}


def _int_name(grain: str, tags) -> str:
    return f"I_{grain}_{'_'.join(tags)}"


for g, tags in _COMBOS:
    parts = [train[_TAG_COL[t]].astype(str) for t in tags] + [_timekey(train, g)]
    _INT_LEVELS[_int_name(g, tags)] = pd.Index(pd.concat(parts, axis=1).agg("_".join, axis=1).unique())

_MONTH_WEIGHT = np.exp(0.5 * (_num(train["Month"]).to_numpy() - 1) / 11)


# --- feature engineering (everything inside prepare) ---------------------------
def prepare(df: pd.DataFrame, drop_month: bool = False) -> pd.DataFrame:
    df = df.drop(columns=ID_COLS + [TARGET], errors="ignore")
    X = pd.DataFrame(index=df.index)
    # calendar as numeric; Month optionally dropped (drift-sensitive feature)
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
    # interaction categoricals (levels fitted on train only)
    for g, tags in _COMBOS:
        key = _int_name(g, tags)
        parts = [df[_TAG_COL[t]].astype(str) for t in tags] + [_timekey(df, g)]
        X[key] = pd.Categorical(pd.concat(parts, axis=1).agg("_".join, axis=1),
                                categories=_INT_LEVELS[key])
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
    models = []
    for g, tags, kw, dm, wpow in _MEMBER_SPECS:
        Xfull = prepare(train, drop_month=dm)
        cols = [c for c in Xfull.columns if c not in KEYS or (g, tags) == _parse_key(c)]
        p = dict(BASE_PARAMS)
        p.update(kw)
        m = xgb.XGBClassifier(**p)
        m.fit(Xfull[cols], to_y(train), sample_weight=None if wpow is None else _MONTH_WEIGHT)
        models.append((m, cols, dm))
    return models


def _parse_key(c: str):
    body = c[2:]  # strip "I_"
    grain, tags = body.split("_", 1)
    return grain, tuple(tags.split("_")) if tags else ()


t0 = time.time()
MODELS = fit_ensemble()
print(f"Training time ({len(MODELS)} members): {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    ps = []
    for m, cols, dm in MODELS:
        Xfull = prepare(df, drop_month=dm)
        ps.append(m.predict_proba(Xfull[cols])[:, 1])
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
