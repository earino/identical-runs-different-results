"""XGBoost binary classifier for airline delays. Only file the agent edits.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. Module-level `predict_proba(df)` maps a raw DataFrame to 1-D P(positive). All feature engineering
     lives in prepare(df) with statistics fitted on the training data only.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
y = (train[TARGET] == POSITIVE).astype(int).to_numpy()
GLOB = float(y.mean())

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
HOUR_LEVELS = pd.Index([str(h) for h in sorted((train["DepTime"] // 100).unique())])
TE_COLS = ["UniqueCarrier", "Origin", "Dest", "Route", "CarrierOrigin", "CarrierDest"]
TE_M = 60


def te_key(df: pd.DataFrame, c: str) -> pd.Series:
    if c == "Route":
        return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    if c == "CarrierOrigin":
        return df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str)
    if c == "CarrierDest":
        return df["UniqueCarrier"].astype(str) + "_" + df["Dest"].astype(str)
    return df[c].astype(str)


def fit_te(idx: np.ndarray) -> dict:
    t = train.iloc[idx]
    yy = y[idx]
    maps = {}
    for c in TE_COLS:
        g = pd.DataFrame({"k": te_key(t, c), "y": yy}).groupby("k")["y"].agg(["sum", "count"])
        maps[c] = (g["sum"] + TE_M * GLOB) / (g["count"] + TE_M)
    return maps


FULL_TE_MAPS = fit_te(np.arange(len(train)))


def prepare(df: pd.DataFrame, maps=None) -> pd.DataFrame:
    if maps is None:
        maps = FULL_TE_MAPS
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])
    dt = df["DepTime"].astype("int64")
    mins = (dt // 100 * 60 + dt % 100) % 1440
    X["DepTimeMin"] = mins
    ang = 2 * np.pi * mins / 1440.0
    X["DepTimeSin"] = np.sin(ang)
    X["DepTimeCos"] = np.cos(ang)
    X["Hour"] = pd.Categorical((dt // 100).astype(str), categories=HOUR_LEVELS)
    dist = df["Distance"].astype(float)
    X["Distance"] = dist
    X["LogDist"] = np.log1p(dist)
    for c in TE_COLS:
        X["TE_" + c] = te_key(df, c).map(maps[c]).astype(float).fillna(GLOB).to_numpy()
    return X


# OOF target encoding for the training matrix (no target leakage into train features)
t0 = time.time()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
X_parts = []
for tr_idx, ho_idx in skf.split(train, y):
    X_parts.append(prepare(train.iloc[ho_idx], fit_te(tr_idx)))
X = pd.concat(X_parts).sort_index()
print(f"Feature build time: {time.time() - t0:.1f}s")

fit_idx, val_idx = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y, random_state=SEED)

model = xgb.XGBClassifier(
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=10,
    min_child_weight=10,
    subsample=0.85,
    colsample_bytree=0.75,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=50,
)

t0 = time.time()
model.fit(X.iloc[fit_idx], y[fit_idx], eval_set=[(X.iloc[val_idx], y[val_idx])], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y := (evald[TARGET] == POSITIVE).astype(int).to_numpy(), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
