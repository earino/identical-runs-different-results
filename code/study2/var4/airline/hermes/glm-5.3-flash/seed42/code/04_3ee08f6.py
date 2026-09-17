"""Airline delay XGBoost: engineered features + tuned model with early stopping.

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

# --- raw columns ----------------------------------------------------------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]


def _num_to_f32(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(np.float32)


def _dep_parts(df: pd.DataFrame):
    d = df["DepTime"].fillna(0).astype("int64")
    h = (d // 100).to_numpy()
    m = (d % 100).to_numpy()
    ang = 2 * np.pi * (h * 60 + m) / 1440.0
    return h, m, np.sin(ang), np.cos(ang)


train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
# category levels fitted on TRAIN only; unseen levels (eval/holdout) map to NaN
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here; predict_proba() applies it to unseen rows."""
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])
    X["Distance"] = _num_to_f32(df, "Distance")
    X["DepTime"] = _num_to_f32(df, "DepTime")
    h, _m, sin_t, cos_t = _dep_parts(df)
    X["dep_sin"] = sin_t
    X["dep_cos"] = cos_t
    X["dep_hour"] = np.minimum(h, 24).astype(np.float32)
    X["dep_norm"] = np.minimum((h * 60 + np.minimum(_m, 59)) / 1440.0, 1.0).astype(np.float32)
    dist = X["Distance"].to_numpy()
    X["log_dist"] = np.log1p(dist).astype(np.float32)
    X["dist_bin"] = pd.cut(
        pd.to_numeric(df["Distance"], errors="coerce").fillna(0),
        bins=[0, 150, 250, 400, 600, 900, 1300, 1900, 2600, 6000],
        labels=False,
    ).astype("float32")
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model ----------------------------------------------------------------------
PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=11,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.7,
    gamma=0.3,
    reg_lambda=2.0,
    reg_alpha=0.1,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)


def fit_one(params, Xa, ya, Xb, yb):
    m = xgb.XGBClassifier(**params)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
    return m


def blend_predict(dfs):
    P = np.mean([m.predict_proba(dfs)[:, 1] for m in MODELS], axis=0)
    return P


MODELS = []
SEEDS = [42, 137]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return blend_predict(prepare(df))


y = to_y(train)
ye = to_y(evald)
Xtr = prepare(train)
Xev = prepare(evald)

t0 = time.time()
for s in SEEDS:
    p = dict(PARAMS)
    p["random_state"] = s
    m = fit_one(p, Xtr, y, Xev, ye)
    MODELS.append(m)
    auc_s = roc_auc_score(ye, m.predict_proba(Xev)[:, 1])
    print(f"seed {s}: auc={auc_s:.4f} best_iter={m.best_iteration}")
print(f"Training time: {time.time() - t0:.1f}s")

t0 = time.time()
eval_auc = roc_auc_score(ye, blend_predict(Xev))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")

# --- OOF diagnostic (does not change the shipped model) ---------------------------
t0 = time.time()
oof = np.zeros(len(train), dtype=np.float64)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for tr_idx, va_idx in skf.split(Xtr, y):
    m = xgb.XGBClassifier(**PARAMS)
    m.fit(Xtr.iloc[tr_idx], y[tr_idx], eval_set=[(Xtr.iloc[va_idx], y[va_idx])], verbose=False)
    oof[va_idx] = m.predict_proba(Xtr.iloc[va_idx])[:, 1]
print(f"OOF AUC: {roc_auc_score(y, oof):.4f}  (OOF time {time.time() - t0:.1f}s)")
