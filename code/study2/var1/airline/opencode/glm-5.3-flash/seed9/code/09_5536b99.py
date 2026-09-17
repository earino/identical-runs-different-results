"""XGBoost binary classifier on airline delays. The only file the agent edits.

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
N_FOLDS = 5

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

RAW_FEATURE_COLS = [c for c in train.columns if c not in ID_COLS + [TARGET]]

# --- target-encoding setup (fitted on training data only) ----------------------
TE_COLS = ["UniqueCarrier", "Origin", "Dest", "Route", "CarrierOrigin", "CarrierDest", "OriginHour",
           "Month", "DayofMonth", "DayOfWeek", "dep_hour", "DestHour",
           "OriginMonth", "OriginDOW", "CarrierHour", "DestMonth", "RouteMonth"]
TE_SMOOTH = {"UniqueCarrier": 20, "Origin": 20, "Dest": 20, "Route": 50,
             "CarrierOrigin": 30, "CarrierDest": 30, "OriginHour": 100,
             "Month": 20, "DayofMonth": 30, "DayOfWeek": 20, "dep_hour": 20, "DestHour": 100,
             "OriginMonth": 100, "OriginDOW": 50, "CarrierHour": 50, "DestMonth": 100, "RouteMonth": 200}
TE_MAPS = {}      # col -> dict(level -> encoded value), full-train mapping
TE_PRIOR = 0.0


def _dep_hour(df: pd.DataFrame) -> pd.Series:
    return df["DepTime"].fillna(-1).astype(int) // 100


def _combo(df: pd.DataFrame, col: str) -> pd.Series:
    if col == "Route":
        return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    if col == "CarrierOrigin":
        return df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str)
    if col == "CarrierDest":
        return df["UniqueCarrier"].astype(str) + "_" + df["Dest"].astype(str)
    if col == "OriginHour":
        return df["Origin"].astype(str) + "_" + _dep_hour(df).astype(str)
    if col == "DestHour":
        return df["Dest"].astype(str) + "_" + _dep_hour(df).astype(str)
    if col == "OriginMonth":
        return df["Origin"].astype(str) + "_" + df["Month"].astype(str)
    if col == "OriginDOW":
        return df["Origin"].astype(str) + "_" + df["DayOfWeek"].astype(str)
    if col == "CarrierHour":
        return df["UniqueCarrier"].astype(str) + "_" + _dep_hour(df).astype(str)
    if col == "DestMonth":
        return df["Dest"].astype(str) + "_" + df["Month"].astype(str)
    if col == "RouteMonth":
        return df["Origin"].astype(str) + "_" + df["Dest"].astype(str) + "_" + df["Month"].astype(str)
    if col == "dep_hour":
        return _dep_hour(df).astype(str)
    return df[col].astype(str)


def fit_target_encoding(df: pd.DataFrame, y: np.ndarray) -> None:
    global TE_PRIOR
    TE_PRIOR = float(y.mean())
    for col in TE_COLS:
        levels = _combo(df, col)
        stats = pd.DataFrame({"lvl": levels, "y": y}).groupby("lvl")["y"].agg(["sum", "count"])
        m = TE_SMOOTH[col]
        enc = (stats["sum"] + TE_PRIOR * m) / (stats["count"] + m)
        TE_MAPS[col] = enc.to_dict()


def te_values(df: pd.DataFrame, col: str, use_oof=None) -> pd.Series:
    """Encoded values for `col`. If use_oof is a list of (val_idx, map_dict) folds, use fold maps
    (out-of-fold, computed on training data only) for the rows in each fold's validation index."""
    levels = _combo(df, col)
    if use_oof is None:
        return levels.map(TE_MAPS[col]).astype(float).fillna(TE_PRIOR)
    out = pd.Series(np.full(len(df), np.nan), index=df.index)
    for val_idx, fold_map in use_oof:
        out.iloc[val_idx] = levels.iloc[val_idx].map(fold_map).astype(float).fillna(TE_PRIOR).to_numpy()
    return out


CAT_LEVELS = {}


def fit_cat_levels(df: pd.DataFrame) -> None:
    for c in ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]:
        CAT_LEVELS[c] = pd.Index(sorted(df[c].dropna().astype(str).unique()))
    CAT_LEVELS["dep_hour"] = pd.Index(sorted(_dep_hour(df).unique().astype(str)))


COUNT_MAPS = {}


def fit_counts(df: pd.DataFrame) -> None:
    for col in ["Route", "Origin", "Dest", "UniqueCarrier"]:
        COUNT_MAPS[col] = _combo(df, col).value_counts().to_dict()


def count_feature(df: pd.DataFrame, col: str) -> pd.Series:
    return np.log1p(_combo(df, col).map(COUNT_MAPS[col]).fillna(0).astype(float))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    deptime = df["DepTime"].fillna(-1).astype(int)
    dhour = _dep_hour(df)

    # numeric
    X["DepTime"] = deptime
    X["Distance"] = df["Distance"]
    X["log_dist"] = np.log1p(df["Distance"].clip(lower=0))
    minutes = dhour * 60 + deptime % 100
    X["hour_sin"] = np.sin(2 * np.pi * minutes / 1440.0)
    X["hour_cos"] = np.cos(2 * np.pi * minutes / 1440.0)

    # native categoricals
    for c in ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]:
        X[c] = pd.Categorical(df[c].astype(str), categories=CAT_LEVELS[c])
    X["dep_hour"] = pd.Categorical(dhour.astype(str), categories=CAT_LEVELS["dep_hour"])

    # flight counts (popularity)
    for col in ["Route", "Origin", "Dest", "UniqueCarrier"]:
        X["cnt_" + col] = count_feature(df, col)

    # target encodings (full-train maps; OOF values are substituted for training rows by the trainer)
    for col in TE_COLS:
        X["te_" + col] = te_values(df, col)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- fit encoders on train only ------------------------------------------------
y = to_y(train)
fit_cat_levels(train)
fit_target_encoding(train, y)
fit_counts(train)

# OOF target encodings for the training matrix (no leakage into the fitted model)
oof_store = {col: [] for col in TE_COLS}
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for fit_idx, val_idx in kf.split(train):
    sub = train.iloc[fit_idx]
    sub_y = y[fit_idx]
    for col in TE_COLS:
        levels = _combo(sub, col)
        stats = pd.DataFrame({"lvl": levels, "yv": sub_y}).groupby("lvl")["yv"].agg(["sum", "count"])
        m = TE_SMOOTH[col]
        enc = (stats["sum"] + TE_PRIOR * m) / (stats["count"] + m)
        oof_store[col].append((val_idx, enc.to_dict()))

# --- model --------------------------------------------------------------------
MODEL_SPECS = [
    dict(n_estimators=300, learning_rate=0.05, max_depth=20, random_state=42,
         colsample_bytree=0.2, colsample_bynode=0.8),
    dict(n_estimators=300, learning_rate=0.05, max_depth=20, random_state=7,
         colsample_bytree=0.2, colsample_bynode=0.8),
    dict(n_estimators=300, learning_rate=0.05, max_depth=20, random_state=3,
         colsample_bytree=0.2, colsample_bynode=0.8),
]

t0 = time.time()
X_all = prepare(train)
for col in TE_COLS:  # substitute leak-free OOF encodings for the training rows
    X_all["te_" + col] = te_values(train, col, use_oof=oof_store[col]).to_numpy()
models = []
for spec in MODEL_SPECS:
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS, **spec)
    m.fit(X_all, y)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    preds = [m.predict_proba(prepare(df))[:, 1] for m in models]
    return np.mean(preds, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
