"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Feature set B (time-of-day) + smoothed target encodings fit on train only (OOF for train rows).
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
y_eval = (evald[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(y_train.mean())

BASE_OBJ = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
base_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in BASE_OBJ}


def _hour(df):
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    return (dt // 100).clip(0, 24), (dt % 100).clip(0, 59), dt


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


H_train, M_train, DT_train = _hour(train)
H_eval, M_eval, DT_eval = _hour(evald)


def base_X(df, hour, minute, dt):
    X = pd.DataFrame(index=df.index)
    for c in BASE_OBJ:
        X[c] = pd.Categorical(df[c], categories=base_levels[c])
    X["DepTime"] = dt
    X["hour"] = hour
    X["minute"] = minute
    tod = hour * 60 + minute
    X["sin_tod"] = np.sin(2 * np.pi * tod / 1440.0)
    X["cos_tod"] = np.cos(2 * np.pi * tod / 1440.0)
    X["Distance"] = df["Distance"].astype(float)
    return X


X_train = base_X(train, H_train, M_train, DT_train)
X_eval = base_X(evald, H_eval, M_eval, DT_eval)

# --- target encodings: group keys built identically for train/eval -------------
TE_SPECS = {
    "te_carrier": ("UniqueCarrier",),
    "te_origin": ("Origin",),
    "te_dest": ("Dest",),
    "te_month": ("Month",),
    "te_hour": ("hour",),
    "te_carrier_hour": ("UniqueCarrier", "hour"),
    "te_origin_hour": ("Origin", "hour"),
    "te_dest_hour": ("Dest", "hour"),
}
TE_M = {k: 150.0 for k in TE_SPECS}
TE_M["te_hour"] = 300.0
TE_M["te_month"] = 300.0

grp_train = {name: X_train[list(cols)].astype(str).agg("|".join, axis=1) for name, cols in TE_SPECS.items()}
grp_eval = {name: X_eval[list(cols)].astype(str).agg("|".join, axis=1) for name, cols in TE_SPECS.items()}


def te_map(grp: pd.Series, y: np.ndarray, m: float) -> pd.Series:
    df = pd.DataFrame({"g": grp.values, "y": y})
    agg = df.groupby("g")["y"].agg(["sum", "count"])
    enc = (agg["sum"] + m * PRIOR) / (agg["count"] + m)
    return enc


te_maps = {name: te_map(grp_train[name], y_train, TE_M[name]) for name in TE_SPECS}

# out-of-fold TE for the training matrix (no leakage into training rows)
oof = {name: np.zeros(len(train)) for name in TE_SPECS}
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for tr_idx, va_idx in kf.split(train):
    yf = y_train[tr_idx]
    for name in TE_SPECS:
        enc = te_map(grp_train[name].iloc[tr_idx], yf, TE_M[name])
        oof[name][va_idx] = enc.reindex(grp_train[name].iloc[va_idx].values).fillna(PRIOR).to_numpy()

for name in TE_SPECS:
    X_train[name] = oof[name]
    X_eval[name] = te_maps[name].reindex(grp_eval[name].values).fillna(PRIOR).to_numpy()

# --- prepare(): the ONLY path predict_proba uses -------------------------------
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    hour, minute, dt = _hour(df)
    X = base_X(df, hour, minute, dt)
    for name, cols in TE_SPECS.items():
        grp = X[list(cols)].astype(str).agg("|".join, axis=1)
        X[name] = te_maps[name].reindex(grp.values).fillna(PRIOR).to_numpy()
    return X


PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model = xgb.XGBClassifier(**PARAMS)
model.fit(X_train, y_train, verbose=False)
for k in (50, 100, 200, 300):
    p = model.predict_proba(X_eval, iteration_range=(0, k))[:, 1]
    print(f"DIAG n={k} eval_auc={roc_auc_score(y_eval, p):.4f}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


eval_auc = roc_auc_score(y_eval, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
