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
cat_cols = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# --- smoothed target encodings, fit on TRAIN ONLY (applied to any df inside prepare) ----
_y = (train[TARGET] == POSITIVE).astype(float)
PRIOR = float(_y.mean())
_dt_tr = train["DepTime"].astype(float)
_hb_tr = (_dt_tr // 100) // 3  # 3-hour buckets


def _te_map(keys, w: float) -> dict:
    g = pd.DataFrame({"k": keys, "y": _y}).groupby("k")["y"].agg(["sum", "count"])
    return ((g["sum"] + PRIOR * w) / (g["count"] + w)).to_dict()


TE = {
    "Hour": (_te_map(_hb_tr, 200), lambda df: (df["DepTime"].astype(float) // 100) // 3),
    "OriginHour": (_te_map(_hb_tr.astype(str) + "-" + train["Origin"], 50),
                   lambda df: ((df["DepTime"].astype(float) // 100) // 3).astype(str) + "-" + df["Origin"]),
    "DestHour": (_te_map(_hb_tr.astype(str) + "-" + train["Dest"], 50),
                 lambda df: ((df["DepTime"].astype(float) // 100) // 3).astype(str) + "-" + df["Dest"]),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame()
    # --- calendar features: c-<n> strings -> ints + cyclic sin/cos -------------------------
    for c, period in (("Month", 12), ("DayofMonth", 31), ("DayOfWeek", 7)):
        n = df[c].str.extract(r"c-(\d+)", expand=False).astype(float)
        X[c + "_n"] = n
        X[c + "_sin"] = np.sin(2 * np.pi * n / period)
        X[c + "_cos"] = np.cos(2 * np.pi * n / period)
    # --- departure time: hhmm int -> hour / minute / minutes-since-midnight + cyclic -------
    dt = df["DepTime"].astype(float)
    hh = (dt // 100).clip(0, 24)
    mm = dt - (dt // 100) * 100
    mins = (hh * 60 + mm).clip(0, 24 * 60)
    X["DepHour"] = hh
    X["DepMinute"] = mm
    X["DepMinutes"] = mins
    X["DepMinutes_sin"] = np.sin(2 * np.pi * mins / (24 * 60))
    X["DepMinutes_cos"] = np.cos(2 * np.pi * mins / (24 * 60))
    X["Distance"] = df["Distance"]
    # --- smoothed target encodings ----------------------------------------------------------
    for name, (m, keyfn) in TE.items():
        X["te_" + name] = keyfn(df).map(m).fillna(PRIOR)
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=2000,
    max_depth=12,
    colsample_bytree=0.4,
    learning_rate=0.07,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=40,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_all, y_all = prepare(train), to_y(train)
n_folds = 10
SEEDS = (SEED, SEED + 1)
folds = np.array_split(np.random.RandomState(SEED).permutation(np.arange(len(X_all))), n_folds)
models = []
for seed in SEEDS:
    for va_idx in folds:
        mask = np.ones(len(X_all), dtype=bool)
        mask[va_idx] = False
        m = xgb.XGBClassifier(**{**PARAMS, "random_state": seed})
        m.fit(X_all[mask], y_all[mask], eval_set=[(X_all[~mask], y_all[~mask])], verbose=False)
        models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  iters={models[0].best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
