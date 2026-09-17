"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
TE_COLS = ["UniqueCarrier", "Origin", "Dest", "route"]
NUM_COLS = ["DepTime", "hour", "sin_t", "cos_t", "Distance"]
FEATURES = NUM_COLS + CAT_COLS + [c + "_te" for c in TE_COLS]

SMOOTH = 50.0


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(float)
    hour = (dep // 100) % 24
    minute = dep % 100
    X["DepTime"] = dep
    X["hour"] = hour
    frac = (hour * 60 + minute) / 1440.0
    X["sin_t"] = np.sin(2 * np.pi * frac)
    X["cos_t"] = np.cos(2 * np.pi * frac)
    X["Distance"] = df["Distance"].astype(float)
    X["route"] = (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).astype("category")
    for c in CAT_COLS:
        X[c] = df[c].astype(str)
    for c in ["UniqueCarrier", "Origin", "Dest"]:
        X[c] = df[c].astype(str)
    X["route"] = X["route"].astype(str)
    return X


_fitted = _engineer(train)
cat_levels = {c: pd.Index(sorted(_fitted[c].dropna().unique())) for c in CAT_COLS}

# --- target encoding: full-train stats for prediction, OOF stats for training rows
y_train = (train[TARGET] == POSITIVE).astype(float)
prior = float(y_train.mean())


def _te_map(values: pd.Series, y: pd.Series) -> dict:
    g = y.groupby(values).agg(["sum", "count"])
    return ((g["sum"] + SMOOTH * prior) / (g["count"] + SMOOTH)).to_dict()


def _apply_te(df: pd.DataFrame, col: str, mapping: dict) -> np.ndarray:
    return df[col].map(mapping).fillna(prior).to_numpy()


te_maps = {c: _te_map(_fitted[c], y_train) for c in TE_COLS}
oof = pd.DataFrame(index=train.index)
rng = np.random.RandomState(SEED)
fold = rng.randint(0, 5, len(train))
oof_all = _engineer(train)
for c in TE_COLS:
    oof[c + "_te"] = prior
    for k in range(5):
        m = _te_map(_fitted[c][fold != k], y_train[fold != k])
        oof.loc[fold == k, c + "_te"] = _apply_te(oof_all, c, m)[fold == k]
te_oof = oof  # column order: c + "_te" for each TE col


def prepare(df: pd.DataFrame, te_frame: pd.DataFrame = None) -> pd.DataFrame:
    """te_frame: precomputed OOF target-encoded columns for training rows (train only)."""
    X = _engineer(df)
    out = X[NUM_COLS].copy()
    for c in CAT_COLS:
        out[c] = pd.Categorical(X[c], categories=cat_levels[c])
    if te_frame is not None:
        for c in TE_COLS:
            out[c + "_te"] = te_frame[c + "_te"].to_numpy()
    else:
        for c in TE_COLS:
            out[c + "_te"] = _apply_te(X, c, te_maps[c])
    return out


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
rng = np.random.RandomState(SEED)
val_mask = rng.rand(len(train)) < 0.1
tr_idx = np.where(~val_mask)[0]
va_idx = np.where(val_mask)[0]

model = xgb.XGBClassifier(
    n_estimators=1500,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_all = prepare(train, te_frame=te_oof)
y_all = to_y(train)
model.fit(X_all.iloc[tr_idx], y_all[tr_idx], eval_set=[(X_all.iloc[va_idx], y_all[va_idx])], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s, best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
