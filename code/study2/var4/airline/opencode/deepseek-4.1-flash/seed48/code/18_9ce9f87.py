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
CAL_COLS = {"Month": "month", "DayofMonth": "day", "DayOfWeek": "dow"}
cat_cols = [c for c in obj_cols if c not in CAL_COLS and train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _freq_map(s: pd.Series) -> pd.Series:
    return s.astype(str).value_counts(normalize=True)


FREQ = {
    "freq_origin": _freq_map(train["Origin"]),
    "freq_dest": _freq_map(train["Dest"]),
    "freq_carrier": _freq_map(train["UniqueCarrier"]),
    "freq_route": _freq_map(train["Origin"].astype(str) + "_" + train["Dest"].astype(str)),
}

_y = (train[TARGET] == POSITIVE).astype(float)
_PRIOR = float(_y.mean())


def _te_map(s: pd.Series, w: float = 100.0) -> pd.Series:
    g = pd.DataFrame({"k": s.astype(str).to_numpy(), "y": _y.to_numpy()}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + _PRIOR * w) / (g["count"] + w)


TE = {
    "te_carrier": _te_map(train["UniqueCarrier"]),
    "te_origin": _te_map(train["Origin"]),
    "te_dest": _te_map(train["Dest"]),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    for c, name in CAL_COLS.items():
        X[name] = _cnum(df[c])
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(int)
    X["dep_hour"] = (dep // 100) % 24
    X["dep_minute"] = dep % 100
    X["doy"] = X["day"] + (X["month"] - 1) * 30.4
    X["is_weekend"] = (X["dow"] >= 6).astype(int)
    ro = df["Origin"].astype(str)
    de = df["Dest"].astype(str)
    X["freq_origin"] = ro.map(FREQ["freq_origin"]).astype(float)
    X["freq_dest"] = de.map(FREQ["freq_dest"]).astype(float)
    X["freq_carrier"] = df["UniqueCarrier"].astype(str).map(FREQ["freq_carrier"]).astype(float)
    X["freq_route"] = (ro + "_" + de).map(FREQ["freq_route"]).astype(float)
    X["te_carrier"] = df["UniqueCarrier"].astype(str).map(TE["te_carrier"]).fillna(_PRIOR).astype(float)
    X["te_origin"] = ro.map(TE["te_origin"]).fillna(_PRIOR).astype(float)
    X["te_dest"] = de.map(TE["te_dest"]).fillna(_PRIOR).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=20,
    learning_rate=0.02,
    min_child_weight=30,
    reg_lambda=5.0,
    reg_alpha=5.0,
    subsample=0.8,
    colsample_bytree=0.5,
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
