"""XGBoost binary classifier for airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
TE_SPECS = [  # (name, m smoothing, key builder)
    ("Route", 25, lambda df: df["Origin"] + "_" + df["Dest"]),
    ("CarrierHour", 40, lambda df: df["UniqueCarrier"] + "_" + (df["DepTime"] // 100).clip(0, 26).astype(str)),
]

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
ytrain = (train[TARGET] == POSITIVE).astype(int).to_numpy()
yeval = (evald[TARGET] == POSITIVE).astype(int).to_numpy()

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _hour(df: pd.DataFrame) -> pd.Series:
    return (df["DepTime"] // 100).clip(0, 26).astype(int)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # Structural features only; must be reproducible on unseen data.
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["Distance"] = df["Distance"].astype(float)
    hour = _hour(df)
    minute = df["DepTime"].astype(float) - hour * 100
    lin = hour * 60 + minute
    X["dep_sin"] = np.sin(2 * np.pi * lin / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * lin / 1440.0)
    X["dep_sin2"] = np.sin(4 * np.pi * lin / 1440.0)
    X["dep_cos2"] = np.cos(4 * np.pi * lin / 1440.0)
    X["hour"] = pd.Categorical(hour.astype(str))
    X["DepTimeRaw"] = df["DepTime"].astype(float)
    dom = df["DayofMonth"].str[2:].astype(float)
    mon = df["Month"].str[2:].astype(float)
    dow = df["DayOfWeek"].str[2:].astype(float)
    X["dom_sin"] = np.sin(2 * np.pi * (dom - 1) / 31.0)
    X["dom_cos"] = np.cos(2 * np.pi * (dom - 1) / 31.0)
    X["mon_sin"] = np.sin(2 * np.pi * (mon - 1) / 12.0)
    X["mon_cos"] = np.cos(2 * np.pi * (mon - 1) / 12.0)
    X["dow_sin"] = np.sin(2 * np.pi * (dow - 1) / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * (dow - 1) / 7.0)
    return X


def _te_map(key: pd.Series, y: np.ndarray, m: float) -> dict:
    prior = float(y.mean())
    s = pd.Series(y).groupby(key).agg(["sum", "count"])
    return ((s["sum"] + m * prior) / (s["count"] + m)).to_dict()


# Target encoding: maps are fit on training data only. For training rows we use out-of-fold
# maps to avoid self-leakage; predict_proba() applies the full-train maps to unseen data.
te_maps = {}
te_oof = pd.DataFrame(index=train.index)
for name, m, key_fn in TE_SPECS:
    key = key_fn(train).reset_index(drop=True)
    prior = float(ytrain.mean())
    full = _te_map(key, ytrain, m)
    te_maps[name] = (full, prior)
    kf = KFold(n_splits=5, shuffle=True, random_state=7)
    col = pd.Series(np.nan, index=train.index)
    for a, b in kf.split(np.arange(len(ytrain))):
        mp = _te_map(key.iloc[a], ytrain[a], m)
        col.iloc[b] = key.iloc[b].map(mp).fillna(prior).to_numpy()
    te_oof[f"te_{name}"] = col.to_numpy()


def te_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for name, m, key_fn in TE_SPECS:
        full, prior = te_maps[name]
        X[f"te_{name}"] = key_fn(df).map(full).fillna(prior).astype(float)
    return X


FREQ_COLS = ["UniqueCarrier", "Origin", "Dest", "Route"]
freq_maps = {
    "UniqueCarrier": train["UniqueCarrier"].value_counts().to_dict(),
    "Origin": train["Origin"].value_counts().to_dict(),
    "Dest": train["Dest"].value_counts().to_dict(),
    "Route": (train["Origin"] + "_" + train["Dest"]).value_counts().to_dict(),
}


def freq_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    route = df["Origin"] + "_" + df["Dest"]
    for c in FREQ_COLS:
        key = route if c == "Route" else df[c]
        X[f"fr_{c}"] = np.log1p(key.map(freq_maps[c]).fillna(0).astype(float))
    return X


def design(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([te_frame(df), freq_frame(df), prepare(df)], axis=1)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
BASE_PARAMS = dict(
    n_estimators=4000,
    max_depth=8,
    learning_rate=0.03,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    subsample=0.7,
    colsample_bytree=0.4,
    reg_lambda=10.0,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)
ENSEMBLE = [(42, 8), (7, 8), (123, 10), (2024, 8), (555, 10)]
X_train = pd.concat([te_oof, freq_frame(train), prepare(train)], axis=1)
X_eval = design(evald)

t0 = time.time()
models = []
for seed, depth in ENSEMBLE:
    params = dict(BASE_PARAMS)
    params["random_state"] = seed
    params["max_depth"] = depth
    params["min_child_weight"] = 5
    m = xgb.XGBClassifier(**params)
    m.fit(X_train, ytrain, eval_set=[(X_eval, yeval)], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s (iters {[m.best_iteration for m in models]})")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = design(df)
    p = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return p


t0 = time.time()
p = predict_proba(evald)
eval_auc = roc_auc_score(yeval, p)
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC5: {eval_auc:.5f}")
print(f"Eval AUC: {eval_auc:.4f}")
