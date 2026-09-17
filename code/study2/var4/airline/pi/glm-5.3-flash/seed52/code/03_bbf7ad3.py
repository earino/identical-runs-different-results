"""XGBoost binary classifier for airline delay. ONLY FILE THE AGENT EDITS.

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

# target-encoded keys: column name -> list of raw columns making up the key
TE_KEYS = {
    "TE_Origin": ["Origin"],
    "TE_Dest": ["Dest"],
    "TE_Route": ["Route"],
    "TE_Carrier": ["UniqueCarrier"],
    "TE_CarrierOrigin": ["UniqueCarrier", "Origin"],
    "TE_CarrierDest": ["UniqueCarrier", "Dest"],
    "TE_DepHour": ["DepHour"],
}
TE_SMOOTH = 20.0


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["Month"] = df["Month"]
    X["DayofMonth"] = df["DayofMonth"]
    X["DayOfWeek"] = df["DayOfWeek"]
    X["DepTime"] = df["DepTime"]
    X["Distance"] = df["Distance"]
    X["UniqueCarrier"] = df["UniqueCarrier"]
    X["Origin"] = df["Origin"]
    X["Dest"] = df["Dest"]
    X["DepHour"] = df["DepTime"] // 100
    X["Route"] = df["Origin"] + "_" + df["Dest"]
    return X


train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()

# category levels + target-encoding maps fitted on TRAINING data only
_base_train = base_features(train)
cat_levels = {
    c: pd.Index(sorted(_base_train[c].dropna().unique()))
    for c in ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "Route"]
}
GLOBAL_MEAN = float(np.mean(y_train))


def _key(df_base: pd.DataFrame, cols: list) -> pd.Series:
    return df_base[cols[0]].astype(str) if len(cols) == 1 else \
        df_base[cols[0]].astype(str) + "_" + df_base[cols[1]].astype(str)


# full-train encoding maps: key -> smoothed mean
te_maps = {}
for name, cols in TE_KEYS.items():
    g = pd.DataFrame({"k": _key(_base_train, cols), "y": y_train}).groupby("k")["y"].agg(["sum", "count"])
    te_maps[name] = ((g["sum"] + TE_SMOOTH * GLOBAL_MEAN) / (g["count"] + TE_SMOOTH)).to_dict()

# out-of-fold encodings for the training rows (computed from train only)
te_oof = {name: np.full(len(train), np.nan) for name in TE_KEYS}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for tr_idx, va_idx in skf.split(_base_train, y_train):
    sub = _base_train.iloc[tr_idx]
    ysub = y_train[tr_idx]
    for name, cols in TE_KEYS.items():
        g = pd.DataFrame({"k": _key(sub, cols), "y": ysub}).groupby("k")["y"].agg(["sum", "count"])
        m = ((g["sum"] + TE_SMOOTH * GLOBAL_MEAN) / (g["count"] + TE_SMOOTH)).to_dict()
        te_oof[name][va_idx] = _key(_base_train.iloc[va_idx], cols).map(m).to_numpy()
del _base_train


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = base_features(df)
    for c in cat_levels:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    for name, cols in TE_KEYS.items():
        X[name] = _key(X, cols).map(te_maps[name]).fillna(GLOBAL_MEAN).astype("float32")
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=3000,
    max_depth=4,
    learning_rate=0.1,
    colsample_bytree=0.7,
    reg_alpha=8.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=100,
    eval_metric="auc",
)

t0 = time.time()
X_train = prepare(train)
# replace TE columns with out-of-fold values for training (no self-leakage)
for name in TE_KEYS:
    X_train[name] = te_oof[name]
model.fit(X_train, y_train, eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s")
print(f"Best iteration: {model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
