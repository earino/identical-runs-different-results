"""XGBoost binary classifier for airline departure-delay prediction.

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

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# out-of-fold target encodings for stable keys (scheduled hour interactions)
SMOOTH = 150.0
y_all = (train[TARGET] == POSITIVE).astype(int).to_numpy()
prior = float(y_all.mean())
TE_KEYS = {
    "te_hour": ["DepTime"],
    "te_carrier": ["UniqueCarrier"],
    "te_origin_hour": ["Origin", "DepTime"],
    "te_dest_hour": ["Dest", "DepTime"],
    "te_carrier_hour": ["UniqueCarrier", "DepTime"],
    "te_db_hour": ["Distance", "DepTime"],
}


def _key(df, cols):
    parts = []
    for c in cols:
        if c == "DepTime":
            parts.append((pd.to_numeric(df["DepTime"], errors="coerce").to_numpy() // 100 % 24).astype(int).astype(str))
        elif c == "Distance":
            parts.append((pd.to_numeric(df["Distance"], errors="coerce").to_numpy() // 500).astype(int).astype(str))
        else:
            parts.append(df[c].astype(str).to_numpy())
    out = parts[0]
    for p in parts[1:]:
        out = out + "|" + p
    return out


def _fit_te_maps(df, y):
    maps = {}
    for name, cols in TE_KEYS.items():
        g = pd.DataFrame({"k": _key(df, cols), "y": y}).groupby("k")["y"].agg(["sum", "count"])
        maps[name] = ((g["sum"] + prior * SMOOTH) / (g["count"] + SMOOTH)).to_dict()
    return maps


TE_MAPS = _fit_te_maps(train, y_all)


def _add_te(X, df):
    for name, cols in TE_KEYS.items():
        X[name] = pd.Series(_key(df, cols)).map(TE_MAPS[name]).fillna(prior).astype("float64").to_numpy()
    return X


def _oof_te(df):
    out = {name: np.full(len(df), prior) for name in TE_KEYS}
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, va in kf.split(df):
        maps = _fit_te_maps(df.iloc[tr], y_all[tr])
        for name, cols in TE_KEYS.items():
            out[name][va] = pd.Series(_key(df.iloc[va], cols)).map(maps[name]).fillna(prior).to_numpy()
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    X = _add_te(X, df)
    return X


X_train = prepare(train)
oof = _oof_te(train)
for name in TE_KEYS:
    X_train[name] = oof[name]
y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model ensemble -----------------------------------------------------------
CONFIGS = []
for depth in range(3, 11):
    for seed in (101, 202, 303, 404):
        r = np.random.RandomState(depth * 1000 + seed)
        gp = "lossguide" if (depth + seed) % 2 == 0 else "depthwise"
        CONFIGS.append((depth, seed, float(r.uniform(0.6, 0.95)), float(r.uniform(0.55, 0.95)),
                        int(r.choice([3, 5, 10, 20])), gp))

models = []
t0 = time.time()
for depth, seed, ss, cs, mcw, gp in CONFIGS:
    m = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=depth,
        max_leaves=int(2 ** (depth - 1)),
        grow_policy=gp,
        learning_rate=0.04,
        subsample=ss,
        colsample_bytree=cs,
        min_child_weight=mcw,
        reg_lambda=1.0,
        max_bin=512,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.zeros(len(X))
    for m in models:
        p += m.predict_proba(X)[:, 1]
    return p / len(models)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
