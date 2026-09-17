"""XGBoost binary classifier for airline departure delay.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Feature engineering (all inside prepare()):
  - native categoricals for Month/DayofMonth/DayOfWeek/UniqueCarrier/Origin/Dest (train levels only)
  - hour/minute split of DepTime (hhmm)
  - smoothed target encoding for Origin, Dest, UniqueCarrier, Origin_Dest route:
    out-of-fold values on the training frame (no leakage), full-train map for any other frame.
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


# target-encoding specs: name -> (train-time keys, prior weight)
TE_PRIOR = 20.0
TE_KEYS_TRAIN = {
    "te_origin": train["Origin"].to_numpy(),
    "te_dest": train["Dest"].to_numpy(),
    "te_carrier": train["UniqueCarrier"].to_numpy(),
    "te_route": (train["Origin"] + "_" + train["Dest"]).to_numpy(),
}
TE_OOF_TRAIN = {name: _oof_target_encode(keys, y_train, TE_PRIOR) for name, keys in TE_KEYS_TRAIN.items()}
TE_MAPS = {name: _smoothed_map(keys, y_train, TE_PRIOR) for name, keys in TE_KEYS_TRAIN.items()}


def prepare(df: pd.DataFrame, oof: bool = False) -> pd.DataFrame:
    """Feature engineering for one raw dataframe. oof=True is used ONLY on the training
    frame (out-of-fold target encodings); any other frame gets the full-train maps."""
    X = df[NUM_COLS].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    X["hour"] = (df["DepTime"] // 100).astype(int)
    X["minute"] = (df["DepTime"] % 100).astype(int)
    keys = {
        "te_origin": df["Origin"].to_numpy(),
        "te_dest": df["Dest"].to_numpy(),
        "te_carrier": df["UniqueCarrier"].to_numpy(),
        "te_route": (df["Origin"] + "_" + df["Dest"]).to_numpy(),
    }
    for name, k in keys.items():
        if oof:
            X[name] = TE_OOF_TRAIN[name][: len(df)]
        else:
            X[name] = pd.Series(k).map(TE_MAPS[name]).fillna(GLOBAL_MEAN).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=4,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    early_stopping_rounds=40,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train, oof=True), y_train, eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iteration={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
