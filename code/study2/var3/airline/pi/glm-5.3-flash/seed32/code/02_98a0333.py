"""XGBoost binary classifier for airline delays. THE ONLY FILE THE AGENT EDITS.

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
cat_cols = [c for c in obj_cols if train[c].nunique() <= 5000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
# --- target encodings (fit on train only) -------------------------------------
TE_SMOOTH = 50.0
_y_tr = (train[TARGET] == POSITIVE).astype(int)
_TE_PRIOR = float(_y_tr.mean())


def _te_map(keys: pd.Series) -> pd.Series:
    g = _y_tr.groupby(keys).agg(["mean", "count"])
    return (g["mean"] * g["count"] + _TE_PRIOR * TE_SMOOTH) / (g["count"] + TE_SMOOTH)


def _h(df):
    return (pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype("int64") // 100).clip(0, 23).astype("int32")


_TE_KEYS = {
    "TE_Hour": lambda d: _h(d).astype(str),
    "TE_HourxDOW": lambda d: _h(d).astype(str) + "|" + d["DayOfWeek"].astype(str),
    "TE_HourxMonth": lambda d: _h(d).astype(str) + "|" + d["Month"].astype(str),
    "TE_Origin": lambda d: d["Origin"].astype(str),
    "TE_Dest": lambda d: d["Dest"].astype(str),
    "TE_Carrier": lambda d: d["UniqueCarrier"].astype(str),
}
_TE_MAPS = {k: _te_map(v(train)) for k, v in _TE_KEYS.items()}
cat_levels["DepHour"] = pd.Index(sorted(_h(train).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # time of day from DepTime (hhmm as integer)
    dt = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype("int64")
    hour = dt // 100
    minute = dt % 100
    tod = (hour * 60 + minute).clip(0, 1439)
    X["DepHour"] = hour.clip(0, 23).astype("int32")
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    # cyclical date features (c-<n> strings -> int)
    month = pd.to_numeric(df["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dow = pd.to_numeric(df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dom = pd.to_numeric(df["DayofMonth"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    X["Month_sin"] = np.sin(2 * np.pi * month / 12.0)
    X["Month_cos"] = np.cos(2 * np.pi * month / 12.0)
    X["DOW_sin"] = np.sin(2 * np.pi * dow / 7.0)
    X["DOW_cos"] = np.cos(2 * np.pi * dow / 7.0)
    X["DOM_sin"] = np.sin(2 * np.pi * dom / 31.0)
    X["DOM_cos"] = np.cos(2 * np.pi * dom / 31.0)
    X["logDistance"] = np.log1p(pd.to_numeric(df["Distance"], errors="coerce"))
    X["DepHour"] = (pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype("int64") // 100).clip(0, 23).astype("int32")
    X["DepHour"] = pd.Categorical(X["DepHour"], categories=cat_levels["DepHour"])
    for _col in ["TE_Hour", "TE_HourxDOW", "TE_HourxMonth", "TE_Origin", "TE_Dest", "TE_Carrier"]:
        X[_col] = _TE_MAPS[_col].reindex(_TE_KEYS[_col](df)).fillna(_TE_PRIOR).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
VALID_FRAC = 0.2
EARLY_ROUNDS = 100


def fit_xgb(X, y):
    params = dict(
        n_estimators=4000,
        learning_rate=0.05,
        max_depth=6,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
        eval_metric="auc",
        early_stopping_rounds=EARLY_ROUNDS,
    )
    rng = np.random.RandomState(SEED)
    mask = rng.rand(len(X)) >= VALID_FRAC
    m = xgb.XGBClassifier(**params)
    m.fit(X[mask], y[mask], eval_set=[(X[~mask], y[~mask])], verbose=False)
    best = m.best_iteration
    params.pop("early_stopping_rounds")
    params["n_estimators"] = int(best + 1)
    full = xgb.XGBClassifier(**params)
    full.fit(X, y)
    print(f"best_iteration={best}")
    imp = sorted(zip(X.columns, full.feature_importances_), key=lambda t: -t[1])
    print("importances:", [(k, round(v, 3)) for k, v in imp[:15]])
    return full


t0 = time.time()
model = fit_xgb(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")
def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
