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
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

ENGINEERED = ["dep_min", "hour", "dep_sin", "dep_cos", "month_sin", "month_cos",
              "dom_sin", "dom_cos", "dow_sin", "dow_cos", "log_dist",
              "weekend", "dom", "dist_bin", "midnight"]


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    dep = X["DepTime"].astype("float64")
    X["dep_min"] = (dep // 100) * 60 + (dep % 100)
    X["hour"] = (dep // 100).astype("int64")
    ang = 2 * np.pi * X["dep_min"] / 1440.0
    X["dep_sin"] = np.sin(ang)
    X["dep_cos"] = np.cos(ang)
    m = X["Month"].str.slice(2).astype("int64")
    ma = 2 * np.pi * m / 12.0
    X["month_sin"] = np.sin(ma)
    X["month_cos"] = np.cos(ma)
    d = X["DayofMonth"].str.slice(2).astype("int64")
    da = 2 * np.pi * d / 30.44
    X["dom_sin"] = np.sin(da)
    X["dom_cos"] = np.cos(da)
    w = X["DayOfWeek"].str.slice(2).astype("int64")
    wa = 2 * np.pi * w / 7.0
    X["dow_sin"] = np.sin(wa)
    X["dow_cos"] = np.cos(wa)
    X["log_dist"] = np.log1p(X["Distance"].astype("float64"))
    X["weekend"] = w.isin([6, 7]).astype("int64")
    X["dom"] = d
    X["dist_bin"] = pd.cut(X["Distance"].astype("float64"), bins=[0, 250, 500, 1000, 2000, 100000], labels=False).astype("float64")
    X["midnight"] = (X["dep_min"] == 0).astype("int64")
    return X


# --- count/busyness features (fit on TRAIN only, label-free) -------------------
_tr = add_engineered(train)
count_maps = {
    "n_origin_hour": _tr.groupby([_tr["Origin"], _tr["hour"]]).size(),
    "n_dest_hour": _tr.groupby([_tr["Dest"], _tr["hour"]]).size(),
    "n_carrier": _tr.groupby(_tr["UniqueCarrier"]).size(),
    "n_route": _tr.groupby([_tr["Origin"], _tr["Dest"]]).size(),
}
COUNT_COLS = list(count_maps)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = add_engineered(df)
    X = X[feature_cols + ENGINEERED]
    X["n_origin_hour"] = count_maps["n_origin_hour"].reindex(pd.MultiIndex.from_arrays([X["Origin"], X["hour"]])).to_numpy()
    X["n_dest_hour"] = count_maps["n_dest_hour"].reindex(pd.MultiIndex.from_arrays([X["Dest"], X["hour"]])).to_numpy()
    X["n_carrier"] = X["UniqueCarrier"].map(count_maps["n_carrier"]).to_numpy()
    X["n_route"] = count_maps["n_route"].reindex(pd.MultiIndex.from_arrays([X["Origin"], X["Dest"]])).to_numpy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["hour"] = pd.Categorical(X["hour"].astype(str), categories=[str(h) for h in range(24)])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def make_model(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=20,
        gamma=2.0,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_lambda=10.0,
        reg_alpha=1.0,
        tree_method="hist",
        enable_categorical=True,
        max_bin=512,
        random_state=seed,
        n_jobs=N_JOBS,
    )


t0 = time.time()
X_train = prepare(train)
y_train = to_y(train)
ensemble = []
for seed in (42, 43, 44, 45, 46):
    m = make_model(seed)
    m.fit(X_train, y_train)
    ensemble.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in ensemble], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
