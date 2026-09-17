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


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# baseline: treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# --- extra feature engineering (all fitted on train only) ----------------------
_dep_min = (train["DepTime"] // 100) * 60 + (train["DepTime"] % 100)
DEP15_LEVELS = pd.Index(sorted((_dep_min // 15).unique()))

# smoothed target encoding: stats from train only; OOF values used for train rows
Y_TR = to_y(train)
GM = float(Y_TR.mean())
TE_ALPHA = 20.0


def _te_key(df: pd.DataFrame) -> dict:
    dep = df["DepTime"].to_numpy()
    mins = (dep // 100) * 60 + (dep % 100)
    d15 = (mins // 15).astype(int).astype(str)
    d5 = (mins // 5).astype(int).astype(str)
    hh = (dep // 100).astype(int).astype(str)
    dow = df["DayOfWeek"].astype(str).to_numpy()
    return {
        "te_carrier": df["UniqueCarrier"].astype(str).to_numpy(),
        "te_origin": df["Origin"].astype(str).to_numpy(),
        "te_dest": df["Dest"].astype(str).to_numpy(),
        "te_dep15": d15,
        "te_route": (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).to_numpy(),
        "te_origin_dep": (df["Origin"].astype(str) + "_T" + hh).to_numpy(),
        "te_dest_dep": (df["Dest"].astype(str) + "_T" + hh).to_numpy(),
        "te_carrier_dep": (df["UniqueCarrier"].astype(str) + "_T" + hh).to_numpy(),
        "te_dep5": d5,
        "te_dow_dep": (dow + "_T" + hh),
    }


TE_TRAIN_KEYS = _te_key(train)


def _fit_te(keys: dict, mask: np.ndarray) -> dict:
    maps = {}
    for name, k in keys.items():
        df_ = pd.DataFrame({"k": k[mask], "y": Y_TR[mask]})
        agg = df_.groupby("k")["y"].agg(["mean", "count"])
        maps[name] = (agg["mean"] * agg["count"] + GM * TE_ALPHA) / (agg["count"] + TE_ALPHA)
    return maps


N_SPLIT = 5
TE_COLS = list(TE_TRAIN_KEYS)


def _build_oof(fold_seed: int) -> dict:
    fold = np.random.RandomState(fold_seed).rand(len(train)).argsort() % N_SPLIT
    oof = {c: np.zeros(len(train)) for c in TE_COLS}
    for f in range(N_SPLIT):
        tr_m = fold != f
        maps_f = _fit_te(TE_TRAIN_KEYS, tr_m)
        va_m = ~tr_m
        for c in TE_COLS:
            oof[c][va_m] = pd.Series(TE_TRAIN_KEYS[c][va_m]).map(maps_f[c]).fillna(GM).to_numpy()
    return oof


TE_OOF = _build_oof(SEED)  # default OOF block used by prepare(train, oof=True)
TE_MAPS = _fit_te(TE_TRAIN_KEYS, np.ones(len(train), dtype=bool))


def prepare(df: pd.DataFrame, oof: bool = False) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dep = df["DepTime"].to_numpy()
    mins = (dep // 100) * 60 + (dep % 100)
    X["dep_minutes"] = mins
    X["dep_bin15"] = pd.Categorical(mins // 15, categories=DEP15_LEVELS)
    keys = _te_key(df)
    is_train_df = oof and len(df) == len(train)
    for c in keys:
        if is_train_df:
            X[c] = TE_OOF[c]
        else:
            X[c] = pd.Series(keys[c]).map(TE_MAPS[c]).fillna(GM).to_numpy()
    return X


# --- model --------------------------------------------------------------------
t0 = time.time()
ytr = to_y(train)
Xtr = prepare(train, oof=True)  # full-map version; TE columns get replaced per member
models = []
for seed in range(1, 9):
    Xtr_m = Xtr.copy()
    oof_m = _build_oof(1000 + seed)  # member-specific OOF folds decorrelate TE noise
    for c in TE_COLS:
        Xtr_m[c] = oof_m[c]
    m = xgb.XGBClassifier(
        n_estimators=450,
        max_depth=10,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.65,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr_m, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
