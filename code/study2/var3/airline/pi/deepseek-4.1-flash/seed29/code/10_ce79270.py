"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Design: target encodings are the strongest features here, so they are computed out-of-fold (5 folds)
and a separate XGBoost model is trained per fold. At prediction time the fold models are averaged.
All encoders are fitted on training data only; predict_proba() rebuilds features from a raw dataframe.
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
N_FOLDS = 10
ALPHA = 100.0

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature schema -----------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

TE_COLS = ["UniqueCarrier", "Origin", "Dest", "origin_hr", "dest_hr", "carrier_hr"]


def _hour_bucket(df: pd.DataFrame) -> pd.Series:
    return (pd.to_numeric(df["DepTime"], errors="coerce") // 100).clip(0, 23) // 4


def _te_keys(df: pd.DataFrame) -> dict:
    hr = _hour_bucket(df)
    return {
        "UniqueCarrier": df["UniqueCarrier"],
        "Origin": df["Origin"],
        "Dest": df["Dest"],
        "origin_hr": df["Origin"].astype(str) + "_" + hr.astype(str),
        "dest_hr": df["Dest"].astype(str) + "_" + hr.astype(str),
        "carrier_hr": df["UniqueCarrier"].astype(str) + "_" + hr.astype(str),
    }


def fit_encoders(df: pd.DataFrame, y: np.ndarray):
    """Fit smoothed target encodings on the given (training-only) rows."""
    prior = float(y.mean())
    ks = _te_keys(df)
    maps = {}
    for c in TE_COLS:
        g = pd.Series(y, index=df.index).groupby(ks[c].values).agg(["mean", "count"])
        maps[c] = ((g["mean"] * g["count"] + prior * ALPHA) / (g["count"] + ALPHA)).astype(float)
    return maps, prior


def build_X(df: pd.DataFrame, maps: dict, prior: float) -> pd.DataFrame:
    """All feature engineering lives here so predict_proba() reproduces it on raw unseen rows."""
    X = df[feature_cols].copy()
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 23)
    minute = (dep % 100).clip(0, 59)
    tod = hour * 60 + minute
    X["hour"] = hour
    X["minute"] = minute
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    ks = _te_keys(df)
    for c in TE_COLS:
        X[c + "_te"] = ks[c].map(maps[c]).fillna(prior).astype(float)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def make_model(seed):
    return xgb.XGBClassifier(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=20,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


t0 = time.time()
folds = list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(train))
encoders = []
models = []
for i, (tr_idx, _) in enumerate(folds):
    dtr = train.iloc[tr_idx]
    maps, prior = fit_encoders(dtr, to_y(dtr))
    encoders.append((maps, prior))
    m = make_model(SEED + i)
    m.fit(build_X(dtr, maps, prior), to_y(dtr))
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    preds = [m.predict_proba(build_X(df, maps, prior))[:, 1] for m, (maps, prior) in zip(models, encoders)]
    return np.mean(preds, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
