"""XGBoost binary classifier on airline delays. Only file the agent edits.

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
from sklearn.model_selection import StratifiedKFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

CAT_COLS = ["Month", "dt15", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
TE_GROUPS = ["Origin", "Dest", "UniqueCarrier", "route"]   # target-encoded groups
TE_M = 20.0                                                # TE smoothing
ORDER = ["doy", "dom", "Month", "dt15", "DayOfWeek", "UniqueCarrier", "Origin", "Dest",
         "DepTime", "Distance", "Origin_te", "Dest_te", "UniqueCarrier_te", "route_te"]


def _dt(df: pd.DataFrame) -> pd.Series:
    dt = df["DepTime"].astype(np.int32)
    return dt.where(dt < 2400, dt - 2400)


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Base features (no TE). Called on train (module level, for fitting) and inside predict_proba."""
    X = pd.DataFrame(index=df.index)
    month = df["Month"].str[2:].astype(np.int32)
    dom = df["DayofMonth"].str[2:].astype(np.int32)
    X["doy"] = month * 31 + dom
    X["dom"] = dom.to_numpy()
    X["Month"] = df["Month"]
    dt = _dt(df)
    X["dt15"] = (dt // 15).astype(str)           # scheduled time in 15-min bins (categorical)
    X["DayOfWeek"] = df["DayOfWeek"]
    X["UniqueCarrier"] = df["UniqueCarrier"]
    X["Origin"] = df["Origin"]
    X["Dest"] = df["Dest"]
    X["DepTime"] = dt.to_numpy()
    X["Distance"] = df["Distance"].to_numpy()
    return X


def _te_keys(df: pd.DataFrame) -> pd.DataFrame:
    K = pd.DataFrame(index=df.index)
    K["Origin"] = df["Origin"].astype(str)
    K["Dest"] = df["Dest"].astype(str)
    K["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    K["route"] = df["Origin"].astype(str) + ">" + df["Dest"].astype(str)
    return K


y_all = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(y_all.mean())


def _te_maps(df_fit: pd.DataFrame, y_fit: np.ndarray, m: float = TE_M) -> dict:
    K = _te_keys(df_fit)
    maps = {}
    for g in TE_GROUPS:
        stats = pd.DataFrame({"k": K[g], "y": y_fit}).groupby("k")["y"].agg(["sum", "count"])
        maps[g] = ((stats["sum"] + m * PRIOR) / (stats["count"] + m)).to_dict()
    return maps


def _te_apply(df: pd.DataFrame, maps: dict) -> pd.DataFrame:
    K = _te_keys(df)
    T = pd.DataFrame(index=df.index)
    for g in TE_GROUPS:
        T[g + "_te"] = K[g].map(maps[g]).astype(np.float32).fillna(PRIOR)
    return T


# fit TE maps on training data only
maps_full = _te_maps(train, y_all)
# out-of-fold TE for the training rows (leak-free)
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
te_oof = pd.DataFrame(index=train.index, columns=[g + "_te" for g in TE_GROUPS], dtype=np.float64)
for fit_idx, app_idx in skf.split(train, y_all):
    maps_fold = _te_maps(train.iloc[fit_idx], y_all[fit_idx])
    te_oof.iloc[app_idx] = _te_apply(train.iloc[app_idx], maps_fold)
te_oof = te_oof.astype(np.float32)

# fit category levels on training data only
_fit = _engineer(train)
cat_levels = {c: pd.Index(sorted(_fit[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame, te_frame: pd.DataFrame = None) -> pd.DataFrame:
    """Base features + TE columns. By default uses full-train maps; pass te_frame to supply
    precomputed (OOF) TE columns for the training fit."""
    X = _engineer(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    T = _te_apply(df, maps_full) if te_frame is None else te_frame
    return pd.concat([X, T], axis=1)[ORDER]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
MEMBERS = [  # lr-mix ensemble of depth-14 models (M7)
    dict(learning_rate=0.03),
    dict(learning_rate=0.04),
    dict(learning_rate=0.025),
]
models = []

t0 = time.time()
X_fit = prepare(train, te_frame=te_oof)   # leak-free OOF-encoded training features
for i, member in enumerate(MEMBERS):
    m = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=14,
        reg_lambda=10.0,
        reg_alpha=0.5,
        subsample=0.8,
        colsample_bytree=0.9,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + i,
        n_jobs=N_JOBS,
        **member,
    )
    m.fit(X_fit, y_all)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
