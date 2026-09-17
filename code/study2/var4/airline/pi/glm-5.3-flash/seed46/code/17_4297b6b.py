"""XGBoost binary classifier on the airline dataset. THIS IS THE ONLY FILE THE AGENT EDITS.

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


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Derived categorical columns (pure string concatenation, no fitted statistics)."""
    d = pd.DataFrame(index=df.index)
    d["route"] = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    d["car_orig"] = df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str)
    d["car_dest"] = df["UniqueCarrier"].astype(str) + "_" + df["Dest"].astype(str)
    return d


# --- smoothed target encoding + frequency counts (fit on TRAIN ONLY, applied inside prepare) ---
GLOBAL_MEAN = float((train[TARGET] == POSITIVE).mean())
te_maps = {}
for _c, _m in {"UniqueCarrier": 100.0, "Origin": 100.0, "Dest": 100.0}.items():
    _g = (train[TARGET] == POSITIVE).astype(float).groupby(train[_c]).agg(["sum", "count"])
    te_maps[_c] = (_g["sum"] + _m * GLOBAL_MEAN) / (_g["count"] + _m)


DROP = frozenset()  # feature groups disabled for the current probe


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    if "te" not in DROP:
        # smoothed target encoding (unseen -> global mean)
        for c, mp in te_maps.items():
            X["te_" + c] = df[c].map(mp).astype(float).fillna(GLOBAL_MEAN)
    # scheduled time of day: raw hhmm is poorly ordered for trees -> hour + cyclical encoding
    hh = (X["DepTime"] // 100).astype("int32")
    mm = (X["DepTime"] % 100).astype("int32")
    tod = (hh * 60 + mm) / 1440.0
    X["hour"] = hh
    X["tod_sin"] = np.sin(2 * np.pi * tod)
    X["tod_cos"] = np.cos(2 * np.pi * tod)
    X["late_sched"] = (X["DepTime"] >= 2400).astype("int8")
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Early stopping uses data/eval.csv (2006-slice1) only to pick the number of trees: the hidden
# holdout is 2006-slice2, so calibrating tree count against 2006 should transfer better than a
# random 2005 split (2005 val AUC was 0.76 vs 0.71 on 2006: strong year shift).
PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=20,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
    early_stopping_rounds=50,
)

y_all = to_y(train)
X_train = prepare(train)
y_eval = to_y(evald)
X_eval = prepare(evald)


def fit_es(params: dict) -> xgb.XGBClassifier:
    p = {**PARAMS, **params}
    m = xgb.XGBClassifier(**p)
    m.fit(X_train, y_all, eval_set=[(X_eval, y_eval)], verbose=False)
    auc = roc_auc_score(y_eval, m.predict_proba(X_eval)[:, 1])
    print(f"probe {params}: best_iter={m.best_iteration} auc={auc:.4f}")
    return m


t0 = time.time()
# mixed ensemble: hist + lossguide members, all around csbn 0.3
LG = {"grow_policy": "lossguide", "max_leaves": 64}
ENSEMBLE = [
    {"colsample_bynode": 0.3},
    {"colsample_bynode": 0.3, "min_child_weight": 10},
    {"colsample_bynode": 0.3, "max_depth": 5},
    {"colsample_bynode": 0.5},
    {"colsample_bynode": 0.3, "min_child_weight": 40},
    {**LG, "colsample_bynode": 0.3},
    {**LG, "colsample_bynode": 0.3, "min_child_weight": 10, "random_state": 7},
    {**LG, "colsample_bynode": 0.3, "random_state": 13},
    {**LG, "colsample_bynode": 0.3, "subsample": 0.8, "random_state": 99},
    {**LG, "colsample_bynode": 0.3, "min_child_weight": 40, "random_state": 21},
    {"colsample_bynode": 0.3, "learning_rate": 0.03, "random_state": 31},
    {**LG, "colsample_bynode": 0.3, "early_stopping_rounds": 150, "random_state": 55},
    {"colsample_bynode": 0.3, "colsample_bytree": 0.8, "random_state": 77},
]
models = [fit_es(c) for c in ENSEMBLE]
print(f"Total fit: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = np.mean([m.predict_proba(prepare(df))[:, 1] for m in models], axis=0)
    return P


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
