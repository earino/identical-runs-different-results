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
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# baseline: treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- target encoding (fit on train only; OOF for train rows, full-train maps for new rows) ---
TE_SMOOTH = 30.0
_folds = np.random.RandomState(SEED).randint(0, 5, len(train))
_y = to_y(train)
_prior = float(_y.mean())
_HOUR_TRAIN = (train["DepTime"] // 100).astype(int).astype(str)
_DOW_TRAIN = train["DayOfWeek"].astype(str)
_MONTH_TRAIN = train["Month"].astype(str)
TE_KEYS = {
    "te_carrier": train["UniqueCarrier"],
    "te_origin": train["Origin"],
    "te_dest": train["Dest"],
    "te_route": train["Origin"] + "_" + train["Dest"],
    "te_o_hour": train["Origin"] + "_" + _HOUR_TRAIN,
    "te_d_hour": train["Dest"] + "_" + _HOUR_TRAIN,
    "te_c_hour": train["UniqueCarrier"] + "_" + _HOUR_TRAIN,
    "te_r_hour": train["Origin"] + "_" + train["Dest"] + "_" + _HOUR_TRAIN,
    "te_o_dow": train["Origin"] + "_" + _DOW_TRAIN,
    "te_d_dow": train["Dest"] + "_" + _DOW_TRAIN,
    "te_o_month": train["Origin"] + "_" + _MONTH_TRAIN,
    "te_d_month": train["Dest"] + "_" + _MONTH_TRAIN,
}
COUNT_KEYS = {
    "n_origin": train["Origin"],
    "n_dest": train["Dest"],
    "n_route": train["Origin"] + "_" + train["Dest"],
    "n_o_hour": train["Origin"] + "_" + _HOUR_TRAIN,
}


def _smoothed_mean(keys: pd.Series, y: np.ndarray, prior: float) -> pd.Series:
    grp = pd.Series(y).groupby(keys.values).agg(["mean", "count"])
    return (grp["count"] * grp["mean"] + TE_SMOOTH * prior) / (grp["count"] + TE_SMOOTH)


te_oof = {}   # OOF-encoded columns for train rows (avoids leakage into training)
te_maps = {}  # category value -> smoothed mean (fit on all of train), for unseen rows
for name, keys in TE_KEYS.items():
    cols = pd.Series(0.0, index=train.index)
    for f in range(5):
        tr_m, va_m = _folds != f, _folds == f
        m = _smoothed_mean(keys[tr_m], _y[tr_m], _prior)
        cols[va_m] = keys[va_m].map(m).astype(float)
    te_oof[name] = cols
    te_maps[name] = _smoothed_mean(keys, _y, _prior)

count_maps = {name: keys.value_counts(normalize=True) for name, keys in COUNT_KEYS.items()}


def _te_keys(df: pd.DataFrame) -> dict:
    hour = (df["DepTime"] // 100).astype(int).astype(str)
    return {
        "te_carrier": df["UniqueCarrier"],
        "te_origin": df["Origin"],
        "te_dest": df["Dest"],
        "te_route": df["Origin"] + "_" + df["Dest"],
        "te_o_hour": df["Origin"] + "_" + hour,
        "te_d_hour": df["Dest"] + "_" + hour,
        "te_c_hour": df["UniqueCarrier"] + "_" + hour,
        "te_r_hour": df["Origin"] + "_" + df["Dest"] + "_" + hour,
        "te_o_dow": df["Origin"] + "_" + df["DayOfWeek"].astype(str),
        "te_d_dow": df["Dest"] + "_" + df["DayOfWeek"].astype(str),
        "te_o_month": df["Origin"] + "_" + df["Month"].astype(str),
        "te_d_month": df["Dest"] + "_" + df["Month"].astype(str),
    }


def _count_keys(df: pd.DataFrame) -> dict:
    hour = (df["DepTime"] // 100).astype(int).astype(str)
    return {
        "n_origin": df["Origin"],
        "n_dest": df["Dest"],
        "n_route": df["Origin"] + "_" + df["Dest"],
        "n_o_hour": df["Origin"] + "_" + hour,
    }


def _te_values(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if df is train:
        for name in TE_KEYS:
            out[name] = te_oof[name].to_numpy()
        for name, keys in COUNT_KEYS.items():
            out[name] = keys.map(count_maps[name]).fillna(0.0).to_numpy()
    else:
        keys = _te_keys(df)
        for name in TE_KEYS:
            out[name] = keys[name].map(te_maps[name]).fillna(_prior).astype(float).to_numpy()
        ckeys = _count_keys(df)
        for name in COUNT_KEYS:
            out[name] = ckeys[name].map(count_maps[name]).fillna(0.0).astype(float).to_numpy()
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    te = _te_values(df)
    for name in TE_KEYS:
        X[name] = te[name].to_numpy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2500,
    max_depth=3,
    learning_rate=0.02,
    min_child_weight=20,
    reg_lambda=10.0,
    subsample=0.7,
    colsample_bytree=0.7,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s, best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
