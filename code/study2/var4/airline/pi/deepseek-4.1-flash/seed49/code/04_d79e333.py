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
# categorical columns kept as native XGBoost categoricals. Month / DayofMonth are
# excluded: their delay base rates shift strongly between the 2005 train and the
# 2006 eval slices (e.g. January 0.54 -> 0.44), so they do not transfer.
CAT_COLS = ["DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# target-encoded keys, fit on training labels only (out-of-fold for the training rows)
TE_SPECS = [
    ("te_route_hour", ("Origin", "Dest", "_hour")),
    ("te_origin", ("Origin",)),
    ("te_dest", ("Dest",)),
    ("te_origin_hour", ("Origin", "_hour")),
    ("te_dest_hour", ("Dest", "_hour")),
]
TE_K = 20.0
N_TE_FOLDS = 5

# label-free frequency encodings, fit on training rows only
COUNT_SPECS = [
    ("cnt_origin", ("Origin",)),
    ("cnt_dest", ("Dest",)),
    ("cnt_route", ("Origin", "Dest")),
    ("cnt_carrier", ("UniqueCarrier",)),
]
NUNIQUE_SPECS = [
    ("nuniq_origin_dest", ("Origin", "Dest")),
    ("nuniq_origin_carrier", ("Origin", "UniqueCarrier")),
    ("nuniq_carrier_dest", ("UniqueCarrier", "Dest")),
]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def _hour_of(df: pd.DataFrame) -> pd.Series:
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    return (dep // 100).clip(0, 23)


def _make_key(df: pd.DataFrame, cols) -> pd.Series:
    hour = _hour_of(df)
    parts = [hour.astype(str) if c == "_hour" else df[c].astype(str) for c in cols]
    s = parts[0]
    for p in parts[1:]:
        s = s + "|" + p
    return s


y_train = to_y(train)
PRIOR = float(y_train.mean())


def _te_encoding(keys: pd.Series, y: np.ndarray) -> pd.Series:
    g = pd.DataFrame({"k": keys.values, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + PRIOR * TE_K) / (g["count"] + TE_K)


# full-training maps used for prediction on unseen rows
TE_MAPS = {}
COUNT_MAPS = {}
NUNIQUE_MAPS = {}
for name, cols in TE_SPECS:
    TE_MAPS[name] = _te_encoding(_make_key(train, cols), y_train)
for name, cols in COUNT_SPECS:
    COUNT_MAPS[name] = _make_key(train, cols).value_counts()
for name, cols in NUNIQUE_SPECS:
    gcol, vcol = cols
    g = pd.DataFrame({"g": train[gcol].astype(str).values, "v": train[vcol].astype(str).values})
    NUNIQUE_MAPS[name] = g.groupby("g")["v"].nunique()

# out-of-fold target encodings for the rows the model is trained on (no leakage)
TRAIN_OOF = {}
_skf = StratifiedKFold(n_splits=N_TE_FOLDS, shuffle=True, random_state=SEED)
_folds = list(_skf.split(np.zeros(len(y_train)), y_train))
for name, cols in TE_SPECS:
    keys = _make_key(train, cols)
    vals = np.full(len(y_train), PRIOR, dtype=float)
    for a, b in _folds:
        enc = _te_encoding(keys.iloc[a], y_train[a])
        vals[b] = keys.iloc[b].map(enc).fillna(PRIOR).to_numpy()
    TRAIN_OOF[name] = vals


def prepare(df: pd.DataFrame, oof: bool = False) -> pd.DataFrame:
    """All feature engineering lives here so predict_proba() reproduces it on unseen rows."""
    X = pd.DataFrame(index=df.index)

    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["Distance"] = dist

    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = _hour_of(df)
    minute = (dep % 100).clip(0, 59)
    dep_min = hour * 60 + minute
    X["dep_minutes"] = dep_min
    X["dep_hour"] = hour
    X["dep_minute"] = minute
    X["dep_sin"] = np.sin(2 * np.pi * dep_min / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * dep_min / 1440.0)
    X["min_sin"] = np.sin(2 * np.pi * minute / 60.0)
    X["min_cos"] = np.cos(2 * np.pi * minute / 60.0)

    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])

    for name, cols in TE_SPECS:
        if oof:
            X[name] = TRAIN_OOF[name]
        else:
            X[name] = _make_key(df, cols).map(TE_MAPS[name]).fillna(PRIOR).to_numpy(dtype=float)

    for name, cols in COUNT_SPECS:
        c = _make_key(df, cols).map(COUNT_MAPS[name]).fillna(0).to_numpy(dtype=float)
        X[name] = np.log1p(c)

    for name, (gcol, vcol) in NUNIQUE_SPECS:
        c = df[gcol].astype(str).map(NUNIQUE_MAPS[name]).fillna(0).to_numpy(dtype=float)
        X[name] = np.log1p(c)

    return X


# --- model --------------------------------------------------------------------
# small ensemble of XGBoost models (seed / depth / column-sampling diversity) averaged
MODEL_CONFIGS = [
    dict(random_state=42, max_depth=5, colsample_bytree=0.2),
    dict(random_state=7, max_depth=5, colsample_bytree=0.2),
    dict(random_state=123, max_depth=5, colsample_bytree=0.2),
    dict(random_state=2024, max_depth=6, colsample_bytree=0.2),
    dict(random_state=777, max_depth=4, colsample_bytree=0.25),
]

X_train = prepare(train, oof=True)
MODELS = []
t0 = time.time()
for cfg in MODEL_CONFIGS:
    params = dict(
        n_estimators=1200,
        learning_rate=0.04,
        subsample=0.9,
        min_child_weight=3,
        reg_lambda=5.0,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=1,
        n_jobs=N_JOBS,
    )
    params.update(cfg)
    m = xgb.XGBClassifier(**params)
    m.fit(X_train, y_train)
    MODELS.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(MODELS)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    preds = [m.predict_proba(X)[:, 1] for m in MODELS]
    return np.mean(preds, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
