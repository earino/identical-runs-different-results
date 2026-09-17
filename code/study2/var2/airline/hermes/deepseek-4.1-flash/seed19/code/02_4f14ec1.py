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


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def _tod(df: pd.DataFrame) -> np.ndarray:
    """Scheduled departure as minutes past midnight (hhmm integers; 2400+ wraps to 0)."""
    dt = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).to_numpy()
    return ((dt // 100) % 24) * 60 + np.minimum(dt % 100, 59)


# group keys for traffic-volume features (how busy an airport/route/hour is)
SPECS = [
    ("Origin",),
    ("Dest",),
    ("UniqueCarrier",),
    ("Origin", "Dest"),
    ("Origin", "Hour"),
    ("Dest", "Hour"),
    ("UniqueCarrier", "Hour"),
]


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=df.index)
    d["Hour"] = _tod(df) // 60
    for c in ["Origin", "Dest", "UniqueCarrier"]:
        d[c] = df[c].astype(str)
    return d


def _key(d: pd.DataFrame, spec) -> np.ndarray:
    k = d[spec[0]].to_numpy().astype(str)
    for c in spec[1:]:
        k = np.char.add(np.char.add(k, "|"), d[c].to_numpy().astype(str))
    return k


_dtrain = _derive(train)
CNT_MAPS = {}
for _spec in SPECS:
    _k = _key(_dtrain, _spec)
    _vc = pd.Series(_k).value_counts()
    CNT_MAPS[_spec] = np.log1p(_vc).to_dict()
    del _vc


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    d = _derive(df)
    for spec in SPECS:
        X[f"CNT_{'_'.join(spec)}"] = pd.Series(_key(d, spec)).map(CNT_MAPS[spec]).fillna(0.0).to_numpy()
    return X


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    reg_lambda=5.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
