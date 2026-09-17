"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

RAW = [c for c in train.columns if c not in ID_COLS + [TARGET]]

# --- categorical vocabularies, fitted on TRAIN ONLY ---------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}

CN_COLS = ["Month", "DayofMonth", "DayOfWeek"]


def _cn(s: pd.Series) -> pd.Series:
    """'c-7' -> 7.0 (missing/unparseable -> NaN)."""
    return pd.to_numeric(s.astype(str).str.slice(2), errors="coerce")


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- target encodings, fitted on TRAIN ONLY -----------------------------------
TE_KEYS = {"carrier": "UniqueCarrier", "origin": "Origin", "dest": "Dest"}
SMOOTH = 50.0

train_y = to_y(train)
PRIOR = float(train_y.mean())


def _key_frame(df: pd.DataFrame) -> pd.DataFrame:
    k = pd.DataFrame(index=df.index)
    for name, col in TE_KEYS.items():
        k[name] = df[col].astype(str)
    k["route"] = _route(df)
    return k


def _fit_te_maps(df: pd.DataFrame, y: np.ndarray | None = None):
    """Smoothed P(delay | key) from the given labelled frame."""
    if y is None:
        y = to_y(df)
    prior = float(y.mean())
    k = _key_frame(df)
    maps = {}
    for name in k.columns:
        g = pd.DataFrame({"k": k[name].to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
        maps[name] = ((g["sum"] + prior * SMOOTH) / (g["count"] + SMOOTH)).to_dict()
    return maps, prior


TE_MAPS, _ = _fit_te_maps(train)


def _apply_te(k: pd.DataFrame, maps, prior: float) -> pd.DataFrame:
    out = pd.DataFrame(index=k.index)
    for name in k.columns:
        out["te_" + name] = k[name].map(maps[name]).fillna(prior).astype(float)
    return out


def _oof_te(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Out-of-fold target encodings for the TRAINING frame (avoids self-leakage)."""
    y = to_y(df)
    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, n_splits, size=len(df))
    k = _key_frame(df)
    out = pd.DataFrame(index=df.index, columns=["te_" + c for c in k.columns], dtype=float)
    for f in range(n_splits):
        tr, va = fold != f, fold == f
        maps, prior = _fit_te_maps(df.loc[tr], y[tr])
        out.loc[va, :] = _apply_te(k.loc[va], maps, prior).to_numpy()
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in CN_COLS:
        X[c] = _cn(df[c])
    t = pd.to_numeric(df["DepTime"], errors="coerce")
    tod = (t // 100) * 60 + (t % 100)  # minutes since midnight (can exceed 1440)
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])  # unseen -> NaN
    X = pd.concat([X, _apply_te(_key_frame(df), TE_MAPS, PRIOR)], axis=1)
    return X


# --- model --------------------------------------------------------------------
X = prepare(train)
y = to_y(train)
# training path uses out-of-fold encodings so the model cannot read its own labels
X = pd.concat([X.drop(columns=[c for c in X.columns if c.startswith("te_")]), _oof_te(train)], axis=1)

rng = np.random.default_rng(SEED)
perm = rng.permutation(len(X))
n_val = int(0.1 * len(X))
val_idx, fit_idx = perm[:n_val], perm[n_val:]
Xfit, yfit = X.iloc[fit_idx], y[fit_idx]
Xval, yval = X.iloc[val_idx], y[val_idx]

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    reg_lambda=5.0,
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=8,
    n_jobs=N_JOBS,
    random_state=SEED,
    eval_metric="auc",
    early_stopping_rounds=30,
)

t0 = time.time()
model.fit(Xfit, yfit, eval_set=[(Xval, yval)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
