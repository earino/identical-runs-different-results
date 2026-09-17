"""XGBoost binary classifier on airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

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
y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "Route"]
TE_KEYS = ["UniqueCarrier", "Origin", "Dest", "Route", "CarrierHour", "OriginHour",
           "DayOfWeek", "DepHour", "RouteHour", "DestHour", "CarrierDoW", "Month", "DayofMonth"]
TE_SMOOTH = 20.0

cat_levels = {
    "Month": pd.Index(sorted(train["Month"].unique())),
    "DayofMonth": pd.Index(sorted(train["DayofMonth"].unique())),
    "DayOfWeek": pd.Index(sorted(train["DayOfWeek"].unique())),
    "UniqueCarrier": pd.Index(sorted(train["UniqueCarrier"].unique())),
    "Origin": pd.Index(sorted(train["Origin"].unique())),
    "Dest": pd.Index(sorted(train["Dest"].unique())),
    "Route": pd.Index(sorted((train["Origin"] + "_" + train["Dest"]).unique())),
}


def base_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic per-row features (no fitted statistics)."""
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = pd.to_numeric(df["DepTime"], errors="coerce")
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["DepHour"] = (X["DepTime"] // 100).fillna(-1).astype(int)
    X["DepMin"] = X["DepTime"] % 100
    X["LogDist"] = np.log1p(X["Distance"].clip(lower=0))
    X["UniqueCarrier"] = df["UniqueCarrier"]
    X["Origin"] = df["Origin"]
    X["Dest"] = df["Dest"]
    X["Route"] = df["Origin"] + "_" + df["Dest"]
    X["CarrierHour"] = X["UniqueCarrier"] + "_" + X["DepHour"].astype(str)
    X["OriginHour"] = X["Origin"] + "_" + X["DepHour"].astype(str)
    X["RouteHour"] = X["Route"] + "_" + X["DepHour"].astype(str)
    X["DestHour"] = X["Dest"] + "_" + X["DepHour"].astype(str)
    X["CarrierDoW"] = X["UniqueCarrier"] + "_" + df["DayOfWeek"].astype(str)
    for c in ["Month", "DayofMonth", "DayOfWeek"]:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    for c in ["UniqueCarrier", "Origin", "Dest", "Route"]:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def te_map(keys: pd.Series, y: np.ndarray, smooth: float = TE_SMOOTH) -> pd.Series:
    """Smoothed target mean per key level: (sum + smooth*prior) / (count + smooth)."""
    prior = y.mean()
    df = pd.DataFrame({"k": keys, "y": y})
    g = df.groupby("k")["y"].agg(["sum", "count"])
    enc = (g["sum"] + smooth * prior) / (g["count"] + smooth)
    return enc


def te_apply(keys: pd.Series, mapping: pd.Series, prior: float) -> np.ndarray:
    return keys.astype("object").map(mapping).fillna(prior).astype(float).to_numpy()


# --- fit target encodings on TRAINING DATA ONLY ------------------------------
train_base = base_frame(train)
prior = float(y_train.mean())

# out-of-fold encodings for the training rows (avoid self-fit leakage)
oof_te = {k: np.zeros(len(train)) for k in TE_KEYS}
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for tr_idx, va_idx in kf.split(train_base):
    y_tr = y_train[tr_idx]
    for k in TE_KEYS:
        m = te_map(train_base[k].iloc[tr_idx], y_tr)
        oof_te[k][va_idx] = te_apply(train_base[k].iloc[va_idx], m, prior)
# full-train encodings used for eval / hidden holdout inside predict_proba
full_te = {k: te_map(train_base[k], y_train) for k in TE_KEYS}


def build_X(b: pd.DataFrame, te: dict) -> pd.DataFrame:
    X = b.drop(columns=TE_KEYS).copy()
    for k in TE_KEYS:
        X["TE_" + k] = te[k]
    return X


X_train = build_X(train_base, oof_te)
feature_cols = list(X_train.columns)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    b = base_frame(df)
    te = {k: te_apply(b[k], full_te[k], prior) for k in TE_KEYS}
    return build_X(b, te)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(X_train, y_train)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
