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
raw_feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [
    c
    for c in raw_feature_cols
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])
]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
num_cols = [c for c in raw_feature_cols if c not in obj_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# Smoothed target encoding (fit on train only): simple TE for cat cols + one route TE.
TE_SMOOTH = 20.0
te_maps: dict[str, dict] = {}
freq_maps: dict[str, dict] = {}

# interaction definitions: name -> (list of base cols, smoothing)
INTERACTIONS: dict[str, tuple[list, float]] = {
    "Origin_Dest": (["Origin", "Dest"], 30.0),
}


def _hour(df: pd.DataFrame) -> np.ndarray:
    dep = df["DepTime"].astype(float).to_numpy()
    return np.clip(np.floor(dep / 100) % 24, 0, 23)


def _interaction_key(df: pd.DataFrame, cols: list) -> pd.Series:
    parts = []
    for c in cols:
        if c == "DepTime_hour":
            parts.append(pd.Series(_hour(df), index=df.index).astype(int).astype(str).reset_index(drop=True))
        else:
            parts.append(df[c].astype(str).reset_index(drop=True))
    out = parts[0]
    for p in parts[1:]:
        out = out + "_" + p
    return pd.Series(out.to_numpy(), index=df.index)


def _fit_target_encodings(df: pd.DataFrame, y: np.ndarray) -> None:
    global_mean = float(y.mean())
    # simple column TEs
    for c in cat_cols:
        stats = pd.DataFrame({"cat": df[c], "y": y}).groupby("cat")["y"].agg(["sum", "count"])
        enc = (stats["sum"] + TE_SMOOTH * global_mean) / (stats["count"] + TE_SMOOTH)
        te_maps[c] = {**enc.to_dict(), "__global__": global_mean}
    # interaction TEs
    for name, (cols, smooth) in INTERACTIONS.items():
        key = _interaction_key(df, cols)
        stats = pd.DataFrame({"cat": key, "y": y}).groupby("cat")["y"].agg(["sum", "count"])
        enc = (stats["sum"] + smooth * global_mean) / (stats["count"] + smooth)
        te_maps[name] = {**enc.to_dict(), "__global__": global_mean}
        freq_maps[name] = key.value_counts().to_dict()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here so predict_proba() reproduces it on unseen rows."""
    X = pd.DataFrame(index=df.index)
    for c in num_cols:
        X[c] = df[c].astype(float)
    dep = df["DepTime"].astype(float)
    hour = np.clip(np.floor(dep / 100) % 24, 0, 23)
    minute = dep - 100 * np.floor(dep / 100)
    X["hour"] = hour
    X["minute"] = minute
    X["minute_of_day"] = hour * 60 + minute
    X["sin_h"] = np.sin(2 * np.pi * hour / 24)
    X["cos_h"] = np.cos(2 * np.pi * hour / 24)
    for c in cat_cols:
        X["cat_" + c] = pd.Categorical(df[c], categories=cat_levels[c])
        m = te_maps[c]
        X["te_" + c] = df[c].map(m).fillna(m["__global__"]).astype(float)
    for name, (cols, _) in INTERACTIONS.items():
        key = _interaction_key(df, cols)
        m = te_maps[name]
        X["te_" + name] = key.map(m).fillna(m["__global__"]).astype(float)
        X["logfreq_" + name] = np.log1p(key.map(freq_maps[name]).fillna(0.0)).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


_fit_target_encodings(train, to_y(train))

# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    subsample=0.8,
    colsample_bytree=1.0,
    min_child_weight=2,
    reg_lambda=1,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=40,
    eval_metric="auc",
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
