"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# baseline: treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# --- schedule-density features (counts on train, no labels) --------------------
def _count_map(values: pd.Series) -> dict:
    return values.value_counts().to_dict()


_hour = (train["DepTime"] // 100).clip(0, 24)
CNT_MAPS = {
    "cnt_Origin": _count_map(train["Origin"].astype(str)),
    "cnt_Dest": _count_map(train["Dest"].astype(str)),
    "cnt_route": _count_map(train["Origin"].astype(str) + "_" + train["Dest"].astype(str)),
    "cnt_carrier": _count_map(train["UniqueCarrier"].astype(str)),
    "cnt_Origin_hour": _count_map(train["Origin"].astype(str) + "@" + _hour.astype(str)),
}
CNT_KEYS = {
    "cnt_Origin": lambda df: df["Origin"].astype(str),
    "cnt_Dest": lambda df: df["Dest"].astype(str),
    "cnt_route": lambda df: df["Origin"].astype(str) + "_" + df["Dest"].astype(str),
    "cnt_carrier": lambda df: df["UniqueCarrier"].astype(str),
    "cnt_Origin_hour": lambda df: df["Origin"].astype(str) + "@" + (df["DepTime"] // 100).clip(0, 24).astype(str),
}

# --- interaction target encodings (OOF on train rows, full-train maps for new rows)
TE_K = 100.0
_y_all = (train[TARGET] == POSITIVE).astype(float)
_p_global = float(_y_all.mean())
TE_NAMES = ["te_carrier_hour", "te_origin_hour", "te_dest_hour", "te_origin_dow", "te_dest_dow", "te_carrier_dow"]


BASE_COL = {"carrier": "UniqueCarrier", "origin": "Origin", "dest": "Dest"}


def _grp_key(df: pd.DataFrame, name: str) -> pd.Series:
    h = (df["DepTime"] // 100).clip(0, 24).astype(str)
    dow = df["DayOfWeek"].astype(str)
    kind, base, _ = name.split("_")
    col = BASE_COL[base]
    if kind == "te" and name.endswith("_hour"):
        return df[col].astype(str) + "@" + h
    if kind == "te" and name.endswith("_dow"):
        return df[col].astype(str) + "@" + dow
    raise ValueError(name)


def _te_map(keys: pd.Series, y: pd.Series, k: float) -> dict:
    g = pd.DataFrame({"v": keys.to_numpy(), "y": y.to_numpy()}).groupby("v")["y"].agg(["sum", "count"])
    enc = (g["sum"] + _p_global * k) / (g["count"] + k)
    return enc.to_dict()


TE_MAPS = {n: _te_map(_grp_key(train, n), _y_all, TE_K) for n in TE_NAMES}

train_te = pd.DataFrame(index=train.index)
from sklearn.model_selection import KFold  # noqa: E402

for tr_idx, oof_idx in KFold(5, shuffle=True, random_state=SEED).split(train):
    tr_part, oof_part = train.iloc[tr_idx], train.iloc[oof_idx]
    y_part = (tr_part[TARGET] == POSITIVE).astype(float)
    for n in TE_NAMES:
        m = _te_map(_grp_key(tr_part, n), y_part, TE_K)
        train_te.loc[oof_idx, n] = _grp_key(oof_part, n).map(m).to_numpy()
train_te = train_te.fillna(_p_global)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for name, keyfn in CNT_KEYS.items():
        X[name] = np.log1p(keyfn(df).map(CNT_MAPS[name]).fillna(0.0).to_numpy())
    is_train = len(df) == len(train) and np.array_equal(df["Origin"].astype(str).to_numpy(), train["Origin"].astype(str).to_numpy())
    for n in TE_NAMES:
        if is_train:
            X[n] = train_te[n].to_numpy()
        else:
            X[n] = _grp_key(df, n).map(TE_MAPS[n]).fillna(_p_global).to_numpy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
COMMON = dict(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)
CONFIGS = [
    dict(n_estimators=100, max_depth=4, learning_rate=0.05, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0),
    dict(n_estimators=150, max_depth=3, learning_rate=0.05, min_child_weight=10, subsample=0.7, colsample_bytree=0.7, reg_lambda=10.0),
    dict(n_estimators=150, max_depth=5, learning_rate=0.03, min_child_weight=5, subsample=0.85, colsample_bytree=0.85, reg_lambda=3.0),
    dict(n_estimators=120, max_depth=6, learning_rate=0.05, min_child_weight=20, subsample=0.7, colsample_bytree=0.7, reg_lambda=8.0),
]

t0 = time.time()
models = []
for cfg in CONFIGS:
    for seed in [42, 7, 123, 2024]:
        m = xgb.XGBClassifier(**COMMON, **cfg, random_state=seed)
        m.fit(prepare(train), to_y(train))
        models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  n_models={len(models)}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
