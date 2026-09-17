"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
from sklearn.model_selection import KFold, train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]


def _cnum(s: pd.Series) -> pd.Series:
    """'c-4' -> 4 (falls back to -1 for anything unexpected)."""
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce").fillna(-1).astype(int)


def _dep(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["DepTime"], errors="coerce").fillna(-1).astype(int)


def _hour(df: pd.DataFrame) -> pd.Series:
    return _dep(df) // 100


def _bucket(df: pd.DataFrame, slots: int) -> pd.Series:
    """Time-of-day bucket: (hhmm -> hour*slots + minute// (60/slots))."""
    d = _dep(df)
    return (d // 100) * slots + (d % 100) // (60 // slots)


def route_of(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# Smoothed target encodings, fit on training data only.
# For the training rows themselves we store 5-fold out-of-fold values (no row sees its own label).
# The dominant signal: departure-delay rate per (route|origin) x time-of-day bucket.
PRIOR = float((train[TARGET] == POSITIVE).mean())
TE_SPECS = {
    "te_rmin15": (lambda d: route_of(d) + "_" + _bucket(d, 4).astype(str), 100.0),   # route x 15 min
    "te_omin15": (lambda d: d["Origin"].astype(str) + "_" + _bucket(d, 4).astype(str), 100.0),
    "te_dmin30": (lambda d: d["Dest"].astype(str) + "_" + _bucket(d, 2).astype(str), 75.0),
    "te_cmin30": (lambda d: d["UniqueCarrier"].astype(str) + "_" + _bucket(d, 2).astype(str), 50.0),
    "te_rhour": (lambda d: route_of(d) + "_" + _hour(d).astype(str), 50.0),         # coarser companions
    "te_ohour": (lambda d: d["Origin"].astype(str) + "_" + _hour(d).astype(str), 50.0),
}
_y_all = (train[TARGET] == POSITIVE).to_numpy()


def _te_map(values: pd.Series, y: np.ndarray, m: float) -> pd.Series:
    g = pd.DataFrame({"v": values.to_numpy(), "y": y}).groupby("v")["y"].agg(["sum", "count"])
    return ((g["sum"] + PRIOR * m) / (g["count"] + m)).astype(float)


TE_MAPS = {name: _te_map(f(train), _y_all, m) for name, (f, m) in TE_SPECS.items()}
OOF = {name: np.zeros(len(train)) for name in TE_SPECS}
for tr_i, va_i in KFold(n_splits=5, shuffle=True, random_state=SEED).split(train):
    for name, (f, m) in TE_SPECS.items():
        mp = _te_map(f(train.iloc[tr_i]), _y_all[tr_i], m)
        OOF[name][va_i] = f(train.iloc[va_i]).map(mp).fillna(PRIOR).to_numpy()

# flight-volume counts (congestion proxies), counted on the training data only
CNT_SPECS = {
    "cnt_origin": lambda d: d["Origin"].astype(str),
    "cnt_dest": lambda d: d["Dest"].astype(str),
    "cnt_route": route_of,
    "cnt_carrier": lambda d: d["UniqueCarrier"].astype(str),
    "cnt_ohour": lambda d: d["Origin"].astype(str) + "_" + _hour(d).astype(str),
    "cnt_rhour": lambda d: route_of(d) + "_" + _hour(d).astype(str),
}
CNT = {name: f(train).value_counts() for name, f in CNT_SPECS.items()}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["month"] = _cnum(df["Month"])
    X["day"] = _cnum(df["DayofMonth"])
    X["dow"] = _cnum(df["DayOfWeek"])
    dt = _dep(df)
    X["dep_hour"] = dt // 100
    X["dep_min"] = dt % 100
    X["dep_minutes"] = X["dep_hour"] * 60 + X["dep_min"]
    X["dep_sin"] = np.sin(2 * np.pi * X["dep_minutes"] / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * X["dep_minutes"] / 1440.0)
    X["distance"] = pd.to_numeric(df["Distance"], errors="coerce").fillna(-1)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    for name, (f, _) in TE_SPECS.items():
        X[name] = f(df).map(TE_MAPS[name]).fillna(PRIOR).to_numpy()
    for name, f in CNT_SPECS.items():
        X[name] = f(df).map(CNT[name]).fillna(0).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    learning_rate=0.05,
    max_depth=20,
    min_child_weight=1,
    subsample=1.0,
    colsample_bytree=0.3,
    reg_lambda=5.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
SEEDS = [1, 2, 3, 5, 7]

t0 = time.time()
X = prepare(train)
y = to_y(train)
for name in TE_SPECS:  # training rows use out-of-fold target encodings
    X[name] = OOF[name]
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
es = xgb.XGBClassifier(n_estimators=2500, early_stopping_rounds=50, random_state=SEED, **PARAMS)
es.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
best_iter = es.best_iteration + 1
print(f"Training time (ES split): {time.time() - t0:.1f}s, best_iter={best_iter}")

t0 = time.time()
models = []
for s in SEEDS:
    m = xgb.XGBClassifier(n_estimators=best_iter, random_state=s, **PARAMS)
    m.fit(X, y, verbose=False)
    models.append(m)
print(f"Training time ({len(SEEDS)} full models): {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
