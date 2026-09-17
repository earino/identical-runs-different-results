"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
# key insight: interaction of fine time-of-day bins with carrier is the strongest signal;
# ensemble members alternate bin granularity (15/30 vs 20/60-minute bins)
BASE_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
NUM_COLS = ["DepTime", "hour", "minute_of_day", "sin_tod", "cos_tod", "mod_shift", "sin_shift",
            "cos_shift", "Distance"]
BINSETS = [(15, 30), (20, 60)]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in BASE_COLS}
_dep = train["DepTime"].astype(int)
_carr = train["UniqueCarrier"].astype(str)


def _binkey(dep: pd.Series, minutes: int) -> pd.Series:
    return dep // 100 * (60 // minutes) + (dep % 100) // minutes


for _m in (15, 20, 30, 60):
    cat_levels[f"b{_m}_carr"] = pd.Index(sorted((_binkey(_dep, _m).astype(str) + "_" + _carr).unique()))


def _shift_hhmm(dep: pd.Series, delta: int) -> pd.Series:
    m = ((dep // 100 * 60 + dep % 100) + delta) % 1440
    return m // 60 * 100 + m % 60


def prepare(df: pd.DataFrame, bins=BINSETS[0]) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in BASE_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    dep = df["DepTime"].astype(int)
    carr = df["UniqueCarrier"].astype(str)
    for m in bins:
        X[f"b{m}_carr"] = pd.Categorical(_binkey(dep, m).astype(str) + "_" + carr,
                                         categories=cat_levels[f"b{m}_carr"])
    mod = dep // 100 * 60 + dep % 100
    X["DepTime"] = dep
    X["hour"] = dep // 100 % 24
    X["minute_of_day"] = mod
    X["sin_tod"] = np.sin(2 * np.pi * mod / 1440.0)
    X["cos_tod"] = np.cos(2 * np.pi * mod / 1440.0)
    # operational day starts ~4:30am: red-eye flights (0-4am) belong to the previous operating day
    msh = (mod - 270) % 1440
    X["mod_shift"] = msh
    X["sin_shift"] = np.sin(2 * np.pi * msh / 1440.0)
    X["cos_shift"] = np.cos(2 * np.pi * msh / 1440.0)
    d = (df["DayOfWeek"].str.slice(2).astype(int) - 1 - (dep // 100 % 24 < 4).astype(int)) % 7
    X["dow_shift"] = pd.Categorical("c-" + (d + 1).astype(str), categories=cat_levels["DayOfWeek"])
    X["Distance"] = df["Distance"].astype(float)
    return X[NUM_COLS + [c for c in X.columns if c not in NUM_COLS]]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: granularity-diverse, jitter-augmented ensemble (2 configs x 2 seeds) ----
BASE = dict(min_child_weight=20, reg_lambda=10.0, reg_alpha=0.5, subsample=0.8, colsample_bytree=0.8,
            colsample_bynode=0.5, tree_method="hist", enable_categorical=True)
CONFIGS = [
    dict(n_estimators=250, max_depth=10, learning_rate=0.05),
    dict(n_estimators=400, max_depth=10, learning_rate=0.04),
]
SEEDS = (42, 1, 7)


def _augment(bins):
    """train rows plus +/-8-minute jittered copies (weight 0.5) to smooth time-bin boundaries."""
    parts = [prepare(train, bins)]
    ys = [to_y(train)]
    ws = [np.ones(len(train))]
    for d in (8, -8):
        trs = train.copy()
        trs["DepTime"] = _shift_hhmm(_dep, d)
        parts.append(prepare(trs, bins))
        ys.append(to_y(train))
        ws.append(np.full(len(train), 0.5))
    return pd.concat(parts, ignore_index=True), np.concatenate(ys), np.concatenate(ws)


y_train = to_y(train)
models = []
model_bins = []
t0 = time.time()
for i, seed in enumerate((42, 1)):
    for j, kw in enumerate(CONFIGS):
        bins = BINSETS[(i + j) % 2]
        X_tr, y_tr, w_tr = _augment(bins)
        m = xgb.XGBClassifier(random_state=seed, n_jobs=N_JOBS, **BASE, **kw)
        m.fit(X_tr, y_tr, sample_weight=w_tr)
        models.append(m)
        model_bins.append(bins)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    ps = [m.predict_proba(prepare(df, bins))[:, 1] for m, bins in zip(models, model_bins)]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
