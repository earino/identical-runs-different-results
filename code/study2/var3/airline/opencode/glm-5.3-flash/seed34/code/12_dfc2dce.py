"""XGBoost binary classifier on the airline dataset. THIS IS THE ONLY FILE THE AGENT EDITS.

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
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42
ALPHA = 20  # target-encoding smoothing

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
y = (train[TARGET] == POSITIVE).astype(int)

# --- target encodings (fit on TRAIN only) --------------------------------------
TE_COLS = ["UniqueCarrier", "Origin", "Dest", "route", "origin_hour", "dest_hour", "carrier_hour", "origin_dow"]
GLOBAL_MEAN = float(y.mean())


def te_key(df: pd.DataFrame, col: str) -> pd.Series:
    if col == "route":
        return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    if col in ("origin_hour", "dest_hour", "carrier_hour"):
        base = {"origin_hour": "Origin", "dest_hour": "Dest", "carrier_hour": "UniqueCarrier"}[col]
        # 15-minute blocks: finer time profile
        b = (pd.to_numeric(df["DepTime"], errors="coerce") // 15).astype("Int64")
        return df[base].astype(str) + "_" + b.astype(str)
    if col == "origin_dow":
        return df["Origin"].astype(str) + "_" + df["DayOfWeek"].astype(str)
    if col == "origin_block":
        b = (pd.to_numeric(df["DepTime"], errors="coerce") // 15).astype("Int64")
        return df["Origin"].astype(str) + "_" + b.astype(str)
    return df[col]


def smoothed_map(keys: pd.Series, yy: pd.Series) -> dict:
    df = pd.DataFrame({"k": keys, "y": yy})
    g = df.groupby("k", observed=True)["y"].agg(["sum", "size"])
    return ((g["sum"] + ALPHA * GLOBAL_MEAN) / (g["size"] + ALPHA)).to_dict()


te_maps = {c: smoothed_map(te_key(train, c), y) for c in TE_COLS}

COUNT_COLS = ["Origin", "Dest", "route", "origin_block", "UniqueCarrier"]
count_maps = {c: te_key(train, c).value_counts().to_dict() for c in COUNT_COLS}

FEATS = [
    "Month", "DayofMonth", "DayOfWeek", "DepTime", "hour", "minute",
    "tod_sin", "tod_cos", "Distance", "logDist",
] + [c + "_te" for c in TE_COLS] + [c + "_n" for c in COUNT_COLS]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["Month"] = pd.to_numeric(df["Month"].astype(str).str.slice(2), errors="coerce")
    X["DayofMonth"] = pd.to_numeric(df["DayofMonth"].astype(str).str.slice(2), errors="coerce")
    X["DayOfWeek"] = pd.to_numeric(df["DayOfWeek"].astype(str).str.slice(2), errors="coerce")
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    X["DepTime"] = dt
    X["hour"] = dt // 100
    X["minute"] = dt % 100
    tod = (X["hour"] * 60 + X["minute"]) / 1440.0
    X["tod_sin"] = np.sin(2 * np.pi * tod)
    X["tod_cos"] = np.cos(2 * np.pi * tod)
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["Distance"] = dist
    X["logDist"] = np.log1p(dist)
    for c in TE_COLS:
        X[c + "_te"] = te_key(df, c).map(te_maps[c]).fillna(GLOBAL_MEAN)
    for c in COUNT_COLS:
        X[c + "_n"] = np.log1p(te_key(df, c).map(count_maps[c]).fillna(0))
    return X[FEATS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def make_model(seed: int, depth: int = 6, sub: float = 0.8) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=400,
        max_depth=depth,
        learning_rate=0.05,
        subsample=sub,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=seed,
        n_jobs=N_JOBS,
    )


MEMBERS = ((42, 6, 0.8), (43, 6, 0.8), (44, 6, 0.9), (45, 8, 0.7), (46, 8, 0.8))

t0 = time.time()
X_train = prepare(train)
# out-of-fold target encoding for the training matrix (no self-leakage)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof = pd.DataFrame(index=train.index, columns=[c + "_te" for c in TE_COLS], dtype=float)
for tr_idx, va_idx in kf.split(train):
    tr_part = train.iloc[tr_idx]
    for c in TE_COLS:
        m = smoothed_map(te_key(tr_part, c), y.iloc[tr_idx])
        oof.iloc[va_idx, oof.columns.get_loc(c + "_te")] = (
            te_key(train, c).iloc[va_idx].map(m).fillna(GLOBAL_MEAN).to_numpy()
        )
for c in TE_COLS:
    X_train[c + "_te"] = oof[c + "_te"].to_numpy()
models = []
for s, d, sub in MEMBERS:
    m = make_model(s, d, sub)
    m.fit(X_train, y.to_numpy())
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
