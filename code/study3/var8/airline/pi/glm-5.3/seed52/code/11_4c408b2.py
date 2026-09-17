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
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
# ALL feature engineering lives inside prepare(): predict_proba() calls it on unseen rows.
# Any statistics used here are computed from `train` (2005) only, at module level.
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

_y_tr = (train[TARGET] == POSITIVE).astype(int).to_numpy()
_PRIOR = float(_y_tr.mean())


def _fit_te(keys: pd.Series, y: np.ndarray, k: float) -> dict:
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k", observed=True)["y"].agg(["mean", "count"])
    return ((g["count"] * g["mean"] + k * _PRIOR) / (g["count"] + k)).to_dict()


def _oh(d): return d["Origin"].astype(str) + "_" + (d["DepTime"].astype(int) // 100).astype(str)
def _dh(d): return d["Dest"].astype(str) + "_" + (d["DepTime"].astype(int) // 100).astype(str)
def _ch(d): return d["UniqueCarrier"].astype(str) + "_" + (d["DepTime"].astype(int) // 100).astype(str)
def _bin(d): return d["DepTime"].astype(int) // 25


TE_SPECS = {
    "te_oh": (10.0, _oh),
    "te_dh": (10.0, _dh),
    "te_ch": (50.0, _ch),
    "te_bin": (100.0, _bin),
    "te_chb": (15.0, lambda d: d["UniqueCarrier"].astype(str) + "_" + (d["DepTime"].astype(int) // 25).astype(str)),
    "te_oc": (10.0, lambda d: d["Origin"].astype(str) + "_" + d["UniqueCarrier"].astype(str)),
    "te_dc": (10.0, lambda d: d["Dest"].astype(str) + "_" + d["UniqueCarrier"].astype(str)),
}

# drift-free volume features (counts of train rows per key)
_route_key = lambda d: d["Origin"].astype(str) + "_" + d["Dest"].astype(str)
_n_origin = train["Origin"].value_counts()
_n_dest = train["Dest"].value_counts()
_n_route = _route_key(train).value_counts()

TE_MAPS = {name: _fit_te(keyf(train), _y_tr, k) for name, (k, keyf) in TE_SPECS.items()}

_oof = {name: np.zeros(len(train)) for name in TE_SPECS}
for tr_idx, va_idx in KFold(10, shuffle=True, random_state=SEED).split(train):
    yf = _y_tr[tr_idx]
    for name, (k, keyf) in TE_SPECS.items():
        m = _fit_te(keyf(train.iloc[tr_idx]), yf, k)
        _oof[name][va_idx] = keyf(train.iloc[va_idx]).map(m).to_numpy(dtype=float)
_oof_df = pd.DataFrame(_oof, index=train.index).fillna(_PRIOR)


def _num(s: pd.Series) -> pd.Series:
    """c-7 -> 7"""
    return s.astype(str).str.replace("c-", "", regex=False).astype(int)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(int)
    hour = dep // 100
    minute = dep % 100
    tod = hour * 60 + minute
    X["hour"] = hour
    X["minute"] = minute
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    X["month"] = _num(df["Month"])
    X["dom"] = _num(df["DayofMonth"])
    X["dow"] = _num(df["DayOfWeek"])
    X["Distance"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(X["Distance"])
    for name, (k, keyf) in TE_SPECS.items():
        X[name] = keyf(df).map(TE_MAPS[name]).astype(float).fillna(_PRIOR)
    X["n_origin"] = np.log1p(df["Origin"].map(_n_origin).fillna(0.0))
    X["n_dest"] = np.log1p(df["Dest"].map(_n_dest).fillna(0.0))
    X["n_route"] = np.log1p(_route_key(df).map(_n_route).fillna(0.0))
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
t0 = time.time()
X_tr = prepare(train)
for name in TE_SPECS:  # swap in-fold TE -> out-of-fold TE for training rows only
    X_tr[name] = _oof_df[name].to_numpy()
X_ev = prepare(evald)
y_tr, y_ev = to_y(train), to_y(evald)

model = xgb.XGBClassifier(
    n_estimators=6000,
    learning_rate=0.02,
    max_depth=12,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=3.0,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=200,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)
model.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s, best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
