"""XGBoost binary classifier for airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- feature spec (fitted on train only) --------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


TOD_BINS = [-0.01, 240, 420, 600, 780, 960, 1140, 1320, 1500]
TOD_LABELS = [0, 1, 2, 3, 4, 5, 6, 7]
TE_SMOOTH = 30.0
_y = to_y(train)
_global_mean = float(_y.mean())
_dep = pd.to_numeric(train["DepTime"], errors="coerce")
_tr_tod = pd.cut((_dep // 100) * 60 + (_dep % 100), bins=TOD_BINS, labels=TOD_LABELS).astype(float)


def _te_fit(keys: pd.Series) -> dict:
    df = pd.DataFrame({"k": keys.astype(str).values, "y": _y})
    g = df.groupby("k")["y"].agg(["sum", "count"])
    return ((g["sum"] + TE_SMOOTH * _global_mean) / (g["count"] + TE_SMOOTH)).to_dict()


te_origin = _te_fit(train["Origin"])
te_dest = _te_fit(train["Dest"])
te_carrier = _te_fit(train["UniqueCarrier"])
te_month = _te_fit(train["Month"])
te_dow = _te_fit(train["DayOfWeek"])
te_tod = _te_fit(_tr_tod)


def _int_from_c(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["Month_i"] = _int_from_c(X["Month"].astype(str))
    X["Day_i"] = _int_from_c(X["DayofMonth"].astype(str))
    X["Dow_i"] = _int_from_c(X["DayOfWeek"].astype(str))
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    X["dep_min"] = (dep // 100) * 60 + (dep % 100)
    X["dep_min"] = X["dep_min"].where(X["dep_min"] <= 24 * 60 - 1)
    ang = 2 * np.pi * X["dep_min"] / 1440.0
    X["dep_sin"] = np.sin(ang)
    X["dep_cos"] = np.cos(ang)
    # smoothed target encoding (fitted on train only; unseen -> smoothed global mean)
    tod_bin = pd.cut(X["dep_min"], bins=TOD_BINS, labels=TOD_LABELS).astype(float)
    X["te_origin"] = X["Origin"].astype(str).map(te_origin).astype(float)
    X["te_dest"] = X["Dest"].astype(str).map(te_dest).astype(float)
    X["te_carrier"] = X["UniqueCarrier"].astype(str).map(te_carrier).astype(float)
    X["te_month"] = X["Month"].astype(str).map(te_month).astype(float)
    X["te_dow"] = X["DayOfWeek"].astype(str).map(te_dow).astype(float)
    X["te_tod"] = tod_bin.astype(str).map(te_tod).astype(float)
    return X


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.1,
    min_child_weight=50,
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


eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
