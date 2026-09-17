"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- fitted-on-train statistics/levels (module constants, applied inside prepare) ---
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in ["UniqueCarrier", "Origin", "Dest"]}
ROUTE_LEVELS = pd.Index(sorted((train["Origin"] + "_" + train["Dest"]).dropna().unique()))

# --- out-of-fold target encodings, fitted on TRAIN only -------------------------
RAW_COLS = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
PRIOR = float((train[TARGET] == POSITIVE).mean())
TE_K = 25.0
N_FOLDS = 5

_y_tr = (train[TARGET] == POSITIVE).astype(int)


def _smoothed_te(keys: pd.Series, y: pd.Series) -> pd.Series:
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y.to_numpy()}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + TE_K * PRIOR) / (g["count"] + TE_K)


def _te_keys(df: pd.DataFrame) -> dict:
    hour = ((df["DepTime"] % 2400) // 100).astype(str)
    return {
        "te_route": df["Origin"].astype(str) + "_" + df["Dest"].astype(str),
        "te_carrier": df["UniqueCarrier"].astype(str),
        "te_origin": df["Origin"].astype(str),
        "te_dest": df["Dest"].astype(str),
        "te_carrier_hour": df["UniqueCarrier"].astype(str) + "_" + hour,
        "te_origin_hour": df["Origin"].astype(str) + "_" + hour,
    }


_te_specs_tr = _te_keys(train)
rng = np.random.RandomState(SEED)
_fold_id = rng.randint(0, N_FOLDS, len(train))
_tr_hash = pd.util.hash_pandas_object(train[RAW_COLS], index=False)
_oof = {n: np.full(len(train), PRIOR) for n in _te_specs_tr}
for f in range(N_FOLDS):
    src = _fold_id != f
    for n, keys in _te_specs_tr.items():
        m = _smoothed_te(keys[src], _y_tr[src])
        _oof[n][_fold_id == f] = keys[_fold_id == f].map(m).fillna(PRIOR).to_numpy()
_full_te = {n: _smoothed_te(keys, _y_tr) for n, keys in _te_specs_tr.items()}
# train-row lookup: row hash -> OOF value; unseen rows fall back to full-train TE
_oof_lookup = {n: pd.Series(v, index=_tr_hash).groupby(level=0).mean() for n, v in _oof.items()}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(np.float64)
    t = dep % 2400.0                      # 2400+ -> after-midnight times
    hour = (t // 100).astype(np.float64)
    minute = t - hour * 100
    tod = hour + minute / 60.0            # time of day in hours (float)
    X["DepTime"] = dep
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)
    month = df["Month"].astype(str).str.slice(start=2).astype(np.int16).astype(np.float64)
    X["month"] = month
    X["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    X["day"] = df["DayofMonth"].astype(str).str.slice(start=2).astype(np.int16).astype(np.float64)
    X["dow"] = df["DayOfWeek"].astype(str).str.slice(start=2).astype(np.int16).astype(np.float64)
    dist = df["Distance"].astype(np.float64)
    X["Distance"] = dist
    X["log_dist"] = np.log1p(dist)
    # native categoricals
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"], categories=cat_levels["UniqueCarrier"])
    X["Origin"] = pd.Categorical(df["Origin"], categories=cat_levels["Origin"])
    X["Dest"] = pd.Categorical(df["Dest"], categories=cat_levels["Dest"])
    X["route"] = pd.Categorical(df["Origin"].astype(str) + "_" + df["Dest"].astype(str), categories=ROUTE_LEVELS)
    # OOF target encodings (train rows: out-of-fold; unseen rows: full-train smoothed)
    h = pd.util.hash_pandas_object(df[RAW_COLS], index=False)
    for n, keys in _te_keys(df).items():
        X[n] = h.map(_oof_lookup[n]).fillna(keys.map(_full_te[n])).fillna(PRIOR)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=3000,
    max_depth=8,
    learning_rate=0.05,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    early_stopping_rounds=300,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=100)
print(f"Training time: {time.time() - t0:.1f}s, best iters: {model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
