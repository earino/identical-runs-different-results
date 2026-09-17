"""XGBoost binary classifier for airline departure delay.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Feature engineering (all inside prepare()):
  - native categoricals for Month/DayofMonth/DayOfWeek/UniqueCarrier/Origin/Dest (train levels only)
  - hour/minute split of DepTime (hhmm)
  - smoothed target encodings: Origin, Dest, UniqueCarrier, Origin_Dest route,
    Origin x hour, Dest x hour, UniqueCarrier x hour. Out-of-fold values on the
    training frame (no leakage); full-train maps for any other frame.
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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
NUM_COLS = ["DepTime", "Distance"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
GLOBAL_MEAN = y_train.mean()


def _smoothed_map(keys: np.ndarray, y: np.ndarray, prior_weight: float) -> pd.Series:
    stats = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["mean", "count"])
    return (stats["count"] * stats["mean"] + prior_weight * GLOBAL_MEAN) / (stats["count"] + prior_weight)


def _oof_target_encode(keys: np.ndarray, y: np.ndarray, prior_weight: float, n_fold: int = 5) -> np.ndarray:
    rng = np.random.RandomState(SEED)
    folds = rng.randint(0, n_fold, len(keys))
    out = np.zeros(len(keys))
    for f in range(n_fold):
        m = folds != f
        means = _smoothed_map(keys[m], y[m], prior_weight)
        out[~m] = pd.Series(keys[~m]).map(means).fillna(GLOBAL_MEAN).to_numpy()
    return out


def _keys(df: pd.DataFrame, hour: np.ndarray) -> dict:
    o = df["Origin"].to_numpy()
    d = df["Dest"].to_numpy()
    c = df["UniqueCarrier"].to_numpy()
    h = hour.astype(str)
    dt = df["DepTime"].to_numpy()
    tb = _time_bucket(df).astype(str)  # 30-minute bucket of day (0..47)
    tb15 = (dt // 15).astype(str)      # 15-minute-of-day slot (0..95)
    return {
        "te_origin": o,
        "te_dest": d,
        "te_carrier": c,
        "te_route": o + "_" + d,
        "te_origin_hour": o + "_h" + h,
        "te_dest_hour": d + "_h" + h,
        "te_carrier_hour": c + "_h" + h,
        "te_route_hour": o + "_" + d + "_h" + h,
        "te_origin_bucket": o + "_t" + tb,
        "te_dest_bucket": d + "_t" + tb,
        "te_route_bucket": o + "_" + d + "_t" + tb,
        "te_carrier_bucket": c + "_t" + tb,
        "te_origin_slot15": o + "_" + tb15,
        "te_route_slot15": o + "_" + d + "_" + tb15,
    }


def _time_bucket(df: pd.DataFrame) -> np.ndarray:
    dt = df["DepTime"].to_numpy()
    return (dt // 100) * 2 + (dt % 100) // 30


TE_PRIOR = 20.0
TE_NAMES = list(_keys(train, np.zeros(len(train), dtype=int)).keys())
TE_KEYS_TRAIN = _keys(train, (train["DepTime"] // 100).to_numpy())
TE_OOF_TRAIN = {name: _oof_target_encode(TE_KEYS_TRAIN[name], y_train, TE_PRIOR) for name in TE_NAMES}
TE_MAPS = {name: _smoothed_map(TE_KEYS_TRAIN[name], y_train, TE_PRIOR) for name in TE_NAMES}


def prepare(df: pd.DataFrame, oof: bool = False) -> pd.DataFrame:
    """Feature engineering for one raw dataframe. oof=True is used ONLY on the training
    frame (out-of-fold target encodings); any other frame gets the full-train maps."""
    hour = (df["DepTime"] // 100).to_numpy()
    X = df[NUM_COLS].copy()
    X["hour"] = hour.astype(int)
    X["minute"] = (df["DepTime"] % 100).to_numpy().astype(int)
    X["tbucket"] = _time_bucket(df).astype(int)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    keys = _keys(df, hour)
    for name in TE_NAMES:
        if oof:
            X[name] = TE_OOF_TRAIN[name]
        else:
            X[name] = pd.Series(keys[name]).map(TE_MAPS[name]).fillna(GLOBAL_MEAN).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
N_BAG = 5
models = []
t0 = time.time()
for s in range(N_BAG):
    m = xgb.XGBClassifier(
        n_estimators=6000,
        max_depth=3,
        learning_rate=0.07,
        colsample_bytree=0.6,
        colsample_bynode=0.6,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=100,
        random_state=s,
        n_jobs=N_JOBS,
    )
    m.fit(prepare(train, oof=True), y_train, eval_set=[(prepare(evald), to_y(evald))], verbose=False)
    models.append(m)
    print(f"seed {s}: best_iteration={m.best_iteration}")
print(f"Training time: {time.time() - t0:.1f}s ({N_BAG} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
