"""XGBoost binary classifier for airline delay prediction.

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

# --- feature configuration ----------------------------------------------------
# categorical columns kept as native XGBoost categoricals
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# target-encoded interaction keys (values in parentheses are joined with "|")
TE_COLS = [
    ("Origin", "Dest", "_hour"),  # route x scheduled hour of day
    ("Origin",),
    ("Dest",),
    ("Origin", "_hour"),
    ("Dest", "_hour"),
]
TE_K = 20.0
N_TE_FOLDS = 5


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def _hour_of(df: pd.DataFrame) -> pd.Series:
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    return (dep // 100).clip(0, 23)


def _te_keys(df: pd.DataFrame) -> dict:
    hour = _hour_of(df)
    out = {}
    for cols in TE_COLS:
        parts = []
        for c in cols:
            parts.append(hour.astype(str) if c == "_hour" else df[c].astype(str))
        s = parts[0]
        for p in parts[1:]:
            s = s + "|" + p
        out[cols] = s
    return out


y_train = to_y(train)
PRIOR = float(y_train.mean())
_te_keys_train = _te_keys(train)


def _te_encoding(keys: pd.Series, y: np.ndarray) -> pd.Series:
    g = pd.DataFrame({"k": keys.values, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + PRIOR * TE_K) / (g["count"] + TE_K)


# full-training encodings used for prediction on unseen rows
TE_MAPS = {cols: _te_encoding(_te_keys_train[cols], y_train) for cols in TE_COLS}

# out-of-fold encodings used for the rows the model is trained on (avoids target leakage)
TRAIN_OOF = {}
_skf = StratifiedKFold(n_splits=N_TE_FOLDS, shuffle=True, random_state=SEED)
_folds = list(_skf.split(np.zeros(len(y_train)), y_train))
for cols in TE_COLS:
    vals = np.full(len(y_train), PRIOR, dtype=float)
    keys = _te_keys_train[cols]
    for a, b in _folds:
        enc = _te_encoding(keys.iloc[a], y_train[a])
        vals[b] = keys.iloc[b].map(enc).fillna(PRIOR).to_numpy()
    TRAIN_OOF[cols] = vals


def prepare(df: pd.DataFrame, oof: bool = False) -> pd.DataFrame:
    """All feature engineering lives here so predict_proba() reproduces it on unseen rows."""
    X = pd.DataFrame(index=df.index)

    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["Distance"] = dist

    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    dep_min = _hour_of(df) * 60 + (dep % 100).clip(0, 59)
    X["dep_minutes"] = dep_min
    X["dep_sin"] = np.sin(2 * np.pi * dep_min / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * dep_min / 1440.0)

    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])

    keys = _te_keys(df)
    for i, cols in enumerate(TE_COLS):
        name = "te_%d" % i
        if oof:
            X[name] = TRAIN_OOF[cols]
        else:
            X[name] = keys[cols].map(TE_MAPS[cols]).fillna(PRIOR).to_numpy(dtype=float)
    return X


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=1200,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.9,
    colsample_bytree=0.2,
    min_child_weight=3,
    reg_lambda=5.0,
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=1,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train, oof=True), y_train)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
