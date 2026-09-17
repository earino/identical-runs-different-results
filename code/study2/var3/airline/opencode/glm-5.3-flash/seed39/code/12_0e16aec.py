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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

CAT_COLS = ["Month", "UniqueCarrier", "Origin", "Dest"]
CATS = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    dt = df["DepTime"].astype(np.int64)
    X["dep_min"] = (dt // 100) * 60 + dt % 100
    X["dep_hour"] = dt // 100
    m = X["dep_min"] % 1440
    X["dep_sin"] = np.sin(2 * np.pi * m / 1440)
    X["dep_cos"] = np.cos(2 * np.pi * m / 1440)
    X["month"] = df["Month"].astype(str).str[2:].astype(np.int64)
    X["dom"] = df["DayofMonth"].astype(str).str[2:].astype(np.int64)
    X["dow"] = df["DayOfWeek"].astype(str).str[2:].astype(np.int64)
    X["distance"] = df["Distance"]
    X["log_dist"] = np.log1p(df["Distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=CATS[c])
    b60 = X["dep_min"] // 60
    for name, key in {
        "CarrierBin": df["UniqueCarrier"].astype(str) + "_" + b60.astype(str),
        "OriginBin": df["Origin"].astype(str) + "_" + b60.astype(str),
        "DestBin": df["Dest"].astype(str) + "_" + b60.astype(str),
    }.items():
        X[f"te_{name}"] = key.map(TE_MAPS[name]).fillna(GM).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- interaction target encodings (fit on training data only) -------------------
_y = to_y(train)
GM = float(_y.mean())
ALPHA = 20
_bin60 = ((train["DepTime"] // 100) * 60 + train["DepTime"] % 100) // 60
TE_KEYS = {
    "CarrierBin": train["UniqueCarrier"].astype(str) + "_" + _bin60.astype(str),
    "OriginBin": train["Origin"].astype(str) + "_" + _bin60.astype(str),
    "DestBin": train["Dest"].astype(str) + "_" + _bin60.astype(str),
}


def _te_map(keys: pd.Series, yy: np.ndarray) -> pd.Series:
    g = pd.DataFrame({"k": keys.to_numpy(), "y": yy}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + ALPHA * GM) / (g["count"] + ALPHA)


TE_MAPS = {name: _te_map(s, _y) for name, s in TE_KEYS.items()}
OOF_TE = {name: np.empty(len(train)) for name in TE_KEYS}
_kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for _trn, _val in _kf.split(train):
    for name, s in TE_KEYS.items():
        m = _te_map(s.iloc[_trn], _y[_trn])
        OOF_TE[name][_val] = s.iloc[_val].map(m).fillna(GM).to_numpy()


# --- model --------------------------------------------------------------------
CONFIGS = [
    (d, s)
    for d in (3, 4, 5, 6, 7, 8, 9)
    for s in (0.6, 0.75, 0.9, 1.0)
]


def make_model(seed: int, depth: int = 6, sub: float = 0.8) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=depth,
        subsample=sub,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        max_bin=512,
        random_state=seed,
        n_jobs=N_JOBS,
    )


t0 = time.time()
y_train = to_y(train)
X_train = prepare(train)
for name, vals in OOF_TE.items():
    X_train[f"te_{name}"] = vals  # out-of-fold encodings for training rows
models = []
for i, (depth, sub) in enumerate(CONFIGS):
    m = make_model(SEED + i, depth, sub)
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
