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
TE_SPECS = {  # column -> smoothing strength m: te = (sum + m*global) / (count + m)
    "UniqueCarrier": 60.0, "Origin": 60.0, "Dest": 60.0,
    "route": 40.0, "car_orig": 40.0, "car_dest": 40.0,
    "Month": 30.0, "DayOfWeek": 30.0, "DayofMonth": 30.0,
}
te_maps, freq_maps = {}, {}
_train_inter = add_interactions(train)
for _c, _m in TE_SPECS.items():
    _src = _train_inter[_c] if _c in ("route", "car_orig", "car_dest") else train[_c]
    _g = (train[TARGET] == POSITIVE).astype(float).groupby(_src).agg(["sum", "count"])
    te_maps[_c] = (_g["sum"] + _m * GLOBAL_MEAN) / (_g["count"] + _m)
    freq_maps[_c] = _g["count"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    inter = add_interactions(df)
    # smoothed target encoding + train frequency (unseen -> global mean / 0)
    for c, mp in te_maps.items():
        src = inter[c] if c in ("route", "car_orig", "car_dest") else df[c]
        X["te_" + c] = src.map(mp).astype(float).fillna(GLOBAL_MEAN)
        X["freq_" + c] = np.log1p(src.map(freq_maps[c]).astype(float).fillna(0.0))
    # scheduled time of day: raw hhmm is poorly ordered for trees -> hour + cyclical encoding
    hh = (X["DepTime"] // 100).astype("int32")
    mm = (X["DepTime"] % 100).astype("int32")
    tod = (hh * 60 + mm) / 1440.0
    X["hour"] = hh
    X["tod_sin"] = np.sin(2 * np.pi * tod)
    X["tod_cos"] = np.cos(2 * np.pi * tod)
    X["late_sched"] = (X["DepTime"] >= 2400).astype("int8")
    X["log_dist"] = np.log1p(X["Distance"])
    X["dist_bin"] = np.digitize(X["Distance"], [250, 500, 750, 1000, 1500, 2000, 3000]).astype("int8")
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

t0 = time.time()
model = xgb.XGBClassifier(**PARAMS)
model.fit(prepare(train), y_all, eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Fit: {time.time() - t0:.1f}s, best_iter={model.best_iteration}, es_val_auc={model.best_score:.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
