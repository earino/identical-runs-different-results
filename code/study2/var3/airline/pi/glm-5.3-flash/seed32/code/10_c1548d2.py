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


def _te_map(keys: pd.Series, M: float = None) -> pd.Series:
    M = TE_SMOOTH if M is None else M
    g = _y_tr.groupby(keys).agg(["mean", "count"])
    return (g["mean"] * g["count"] + _TE_PRIOR * M) / (g["count"] + M)


def _h(df):
    return (pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype("int64") // 100).clip(0, 23).astype("int32")


_TE_KEYS = {
    "TE_Hour": lambda d: _h(d).astype(str),
    "TE_HourxDOW": lambda d: _h(d).astype(str) + "|" + d["DayOfWeek"].astype(str),
    "TE_HourxMonth": lambda d: _h(d).astype(str) + "|" + d["Month"].astype(str),
    "TE_Origin": lambda d: d["Origin"].astype(str),
    "TE_Dest": lambda d: d["Dest"].astype(str),
    "TE_Carrier": lambda d: d["UniqueCarrier"].astype(str),
    "TE_OriginxHour": lambda d: d["Origin"].astype(str) + "|" + _h(d).astype(str),
    "TE_CarrierxHour": lambda d: d["UniqueCarrier"].astype(str) + "|" + _h(d).astype(str),
}
_TE_M = {"TE_OriginxHour": 100.0, "TE_CarrierxHour": 100.0}
_TE_MAPS_A = {k: _te_map(v(train), _TE_M.get(k)) for k, v in _TE_KEYS.items()}
_TE_MAPS_B = {k: _te_map(v(train), 3.0 * _TE_M.get(k, TE_SMOOTH)) for k, v in _TE_KEYS.items()}
cat_levels["DepHour"] = pd.Index(sorted(_h(train).unique()))


def prepare(df: pd.DataFrame, te_maps=None) -> pd.DataFrame:
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
    if te_maps is None:
        te_maps = _TE_MAPS_A
    for _col in ["TE_Hour", "TE_HourxDOW", "TE_HourxMonth", "TE_Origin", "TE_Dest", "TE_Carrier",
                 "TE_OriginxHour", "TE_CarrierxHour"]:
        X[_col] = te_maps[_col].reindex(_TE_KEYS[_col](df)).fillna(_TE_PRIOR).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
VALID_FRAC = 0.4
EARLY_ROUNDS = 50


def fit_one(dsub, dval, dall, seed, depth, mcw, colsample, subsample, lr=0.04):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": lr,
        "max_depth": depth,
        "min_child_weight": mcw,
        "colsample_bytree": colsample,
        "subsample": subsample,
        "tree_method": "hist",
        "max_bin": 256,
        "seed": seed,
        "nthread": N_JOBS,
    }
    bst = xgb.train(params, dsub, num_boost_round=8000, evals=[(dval, "v")],
                    early_stopping_rounds=EARLY_ROUNDS, verbose_eval=False)
    best = bst.best_iteration
    full = xgb.train(params, dall, num_boost_round=best + 1)
    print(f"seed={seed} depth={depth} best_iteration={best}")
    return full


CONFIGS = [
    (42, 8, 10, 0.9, 1.0, "A"),
    (43, 6, 10, 0.9, 1.0, "B"),
    (58, 8, 10, 0.9, 1.0, "A"),
    (45, 8, 10, 0.7, 1.0, "B"),
    (52, 8, 1, 0.9, 1.0, "A"),
    (49, 8, 1, 0.8, 1.0, "B"),
]
ENSEMBLE = []


def dm_prepare(df: pd.DataFrame) -> xgb.DMatrix:
    return xgb.DMatrix(prepare(df), enable_categorical=True)


t0 = time.time()
y_all = to_y(train)
rng = np.random.RandomState(SEED)
mask = rng.rand(len(train)) >= VALID_FRAC
DM = {}
for variant, maps in (("A", _TE_MAPS_A), ("B", _TE_MAPS_B)):
    X_all = prepare(train, maps)
    dall = xgb.DMatrix(X_all, label=y_all, enable_categorical=True)
    dsub = xgb.DMatrix(X_all[mask], label=y_all[mask], enable_categorical=True)
    dval = xgb.DMatrix(X_all[~mask], label=y_all[~mask], enable_categorical=True)
    DM[variant] = (dall, dsub, dval)
for cfg in CONFIGS:
    dall, dsub, dval = DM[cfg[5]]
    ENSEMBLE.append((fit_one(dsub, dval, dall, *cfg[:-1]), cfg[5]))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    D = {v: xgb.DMatrix(prepare(df, m), enable_categorical=True) for v, m in (("A", _TE_MAPS_A), ("B", _TE_MAPS_B))}
    P = np.mean([m.predict(D[v]) for m, v in ENSEMBLE], axis=0)
    return P


t0 = time.time()
d_eval = {v: xgb.DMatrix(prepare(evald, m), enable_categorical=True) for v, m in (("A", _TE_MAPS_A), ("B", _TE_MAPS_B))}
for m, v in ENSEMBLE:
    print(f"member_eval_auc={roc_auc_score(to_y(evald), m.predict(d_eval[v])):.4f}")
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
