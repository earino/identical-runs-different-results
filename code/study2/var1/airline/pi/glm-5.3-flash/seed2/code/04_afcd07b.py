"""XGBoost binary classifier: freq encodings; PROBE over shallow depths / tree counts.

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
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# frequency (count) statistics fitted on TRAIN ONLY, applied inside prepare()
FREQ_COLS = [c for c in ("UniqueCarrier", "Origin", "Dest") if c in cat_cols]
freq_maps = {c: train[c].value_counts() for c in FREQ_COLS}
route_key = ("Origin", "Dest") if "Origin" in cat_cols and "Dest" in cat_cols else None
if route_key:
    freq_maps["route"] = train.groupby(list(route_key)).size().to_dict()  # route frequency (count of rows)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN

    # frequency encodings (stats from train only)
    for c in FREQ_COLS:
        X[c + "_freq"] = np.log1p(df[c].map(freq_maps[c]).astype("float64"))
    if route_key:
        X["route_freq"] = np.log1p(df[list(route_key)].apply(tuple, axis=1).map(freq_maps["route"]).astype("float64"))

    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


BASE_PARAMS = dict(
    learning_rate=0.05,
    min_child_weight=1,
    subsample=0.9,
    reg_lambda=1.0,
    reg_alpha=0.0,
    gamma=0.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
)

# --- model --------------------------------------------------------------------
t0 = time.time()
X = prepare(train)
y = to_y(train)
Xe = prepare(evald)
ye = to_y(evald)

PROBES = [
    dict(max_depth=2, n_estimators=400),
    dict(max_depth=2, n_estimators=1000),
    dict(max_depth=3, n_estimators=150),
    dict(max_depth=3, n_estimators=400),
    dict(max_depth=3, n_estimators=250, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=3, n_estimators=250, min_child_weight=10),
]
results = {}
for pr in PROBES:
    m = xgb.XGBClassifier(**{**BASE_PARAMS, **pr})
    m.fit(X, y)
    auc = roc_auc_score(ye, m.predict_proba(Xe)[:, 1])
    key = f"d{pr['max_depth']}_n{pr['n_estimators']}_lr{pr.get('learning_rate', 0.05)}"
    results[key] = (auc, m)
    print(f"PROBE {key} eval_auc={auc:.4f}")

best_key = max(results, key=lambda k: results[k][0])
model = results[best_key][1]
print(f"BEST {best_key}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


eval_auc = roc_auc_score(ye, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
