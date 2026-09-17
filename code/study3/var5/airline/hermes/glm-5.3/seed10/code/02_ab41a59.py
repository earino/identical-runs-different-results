"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
cat_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# --- target encoding machinery (fit on TRAIN ONLY) -----------------------------
TE_COLS = ["UniqueCarrier", "Origin", "Dest"]
TE_SMOOTH = 20.0
te_maps = {}  # col -> (dict level -> smoothed mean, global prior)


def fit_target_encoders(y):
    global te_maps
    prior = float(y.mean())
    te_maps = {}
    for c in TE_COLS:
        grp = pd.DataFrame({"k": train[c].to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
        enc = (grp["sum"] + TE_SMOOTH * prior) / (grp["count"] + TE_SMOOTH)
        te_maps[c] = (enc.to_dict(), prior)


def apply_target_enc(df: pd.DataFrame, cols_prefix: str = "") -> pd.DataFrame:
    """Apply fitted target encodings to an arbitrary raw dataframe (train at fit time, holdout at predict time)."""
    out = {}
    for c in TE_COLS:
        mapping, prior = te_maps[c]
        keys = df[c].to_numpy()
        out[cols_prefix + c + "_te"] = np.array([mapping.get(k, np.nan) for k in keys], dtype=float)
        # unseen -> NaN so the trees can learn a default direction
        _ = prior
    return pd.DataFrame(out, index=df.index)


def prepare(df: pd.DataFrame, te_block: pd.DataFrame = None) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    t = X["DepTime"].to_numpy()
    hour = (t // 100) % 24
    minute = t % 100
    mins = hour * 60 + minute  # minutes since midnight
    X["hour"] = hour
    X["mins"] = mins
    day = 2 * np.pi * (mins / 1440.0)
    X["sin_day"] = np.sin(day)
    X["cos_day"] = np.cos(day)
    if te_block is not None:
        for c in TE_COLS:
            X[c + "_te"] = te_block[c + "_te"].to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    early_stopping_rounds=30,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
y_train = to_y(train)

# OOF target encoding: 5-fold, encoders refit per fold so train rows get fold-local values
from sklearn.model_selection import KFold

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=SEED)
te_oof = pd.DataFrame(index=train.index, columns=[c + "_te" for c in TE_COLS], dtype=float)
for tr_idx, va_idx in kf.split(train):
    y_tr = y_train[tr_idx]
    te_maps_local = {}
    prior = float(y_tr.mean())
    for c in TE_COLS:
        grp = pd.DataFrame({"k": train[c].to_numpy()[tr_idx], "y": y_tr}).groupby("k")["y"].agg(["sum", "count"])
        enc = (grp["sum"] + TE_SMOOTH * prior) / (grp["count"] + TE_SMOOTH)
        te_maps_local[c] = enc.to_dict()
    for c in TE_COLS:
        keys = train[c].to_numpy()[va_idx]
        te_oof.iloc[va_idx, te_oof.columns.get_loc(c + "_te")] = np.array(
            [te_maps_local[c].get(k, np.nan) for k in keys], dtype=float
        )

# full-data encoders for eval/holdout
fit_target_encoders(y_train)

Xtr = prepare(train, te_block=te_oof)
Xev = prepare(evald, te_block=apply_target_enc(evald))
model.fit(Xtr, y_train, eval_set=[(Xev, to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, te_block=apply_target_enc(df)))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
