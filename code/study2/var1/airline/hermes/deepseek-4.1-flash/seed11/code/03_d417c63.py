"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

exp7: add smoothed target encoding for UniqueCarrier / Origin / Dest / route. In-sample target encoding
leaks, so train rows get out-of-fold values (5-fold on the training frame) while the live path
(predict_proba) uses the full-train maps. Smoothing toward the global prior is heavy because each airport
only has ~350 training rows, so raw level means are mostly noise.
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
DROP = ["Month"]                      # year-specific seasonality, does not transfer to 2006
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET] + DROP]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
num_cols = [c for c in feature_cols if c not in obj_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

TE_COLS = ["UniqueCarrier", "Origin", "Dest", "route"]
SMOOTH = 60.0
TE_ROUTE_SEP = "_"


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + TE_ROUTE_SEP + df["Dest"].astype(str)


def _key(df: pd.DataFrame, col: str) -> pd.Series:
    return _route(df) if col == "route" else df[col].astype(str)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


y_train = to_y(train)
PRIOR = float(y_train.mean())


def _encode(y: np.ndarray, keys: pd.Series) -> dict:
    """Smoothed mean target per level: (sum_y + prior * SMOOTH) / (n + SMOOTH)."""
    df = pd.DataFrame({"k": keys.to_numpy(), "y": y})
    g = df.groupby("k")["y"].agg(["sum", "count"])
    return ((g["sum"] + PRIOR * SMOOTH) / (g["count"] + SMOOTH)).to_dict()


# full-train maps -> used by the live prediction path
TE_MAP = {c: _encode(y_train, _key(train, c)) for c in TE_COLS}

# out-of-fold maps -> used only to build the training matrix
N_FOLD = 5
_fold = np.random.RandomState(SEED).randint(0, N_FOLD, size=len(train))
OOF = {}
for f in range(N_FOLD):
    mask = _fold != f
    for c in TE_COLS:
        m = _encode(y_train[mask], _key(train[mask], c))
        OOF.setdefault(c, np.full(len(train), np.nan))[~mask] = _key(train[~mask], c).map(m).to_numpy()
OOF = {c: np.nan_to_num(v, nan=PRIOR) for c, v in OOF.items()}


def prepare(df: pd.DataFrame, oof: bool = False) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[cat_cols + num_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    for c in TE_COLS:
        if oof:
            X["te_" + c] = OOF[c]
        else:
            X["te_" + c] = _key(df, c).map(TE_MAP[c]).fillna(PRIOR).to_numpy()
    return X


# --- model --------------------------------------------------------------------
# Seed-averaged ensemble: each member sees a different subsample/colsample draw; averaging cancels part
# of the variance that otherwise shows up as a year-specific fit.
N_MODELS = 8
COMMON = dict(
    n_estimators=30,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
X_train = prepare(train, oof=True)

t0 = time.time()
models = []
for i in range(N_MODELS):
    m = xgb.XGBClassifier(random_state=SEED + i, **COMMON)
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
