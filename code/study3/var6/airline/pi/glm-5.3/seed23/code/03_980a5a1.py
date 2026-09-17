"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     ALL feature engineering lives inside prepare() (target-encoding maps are fitted on train only, at module
     level, and are applied inside prepare() to whatever dataframe comes in).
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
TE_M = 50  # smoothing count for target encoding
N_FOLDS = 5

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
ytr = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(ytr.mean())


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def _cnum(s: pd.Series) -> pd.Series:
    """'c-7' -> 7 (float, NaN stays NaN)."""
    return s.astype(str).str.extract(r"(\d+)")[0].astype(float)


def _hour(df: pd.DataFrame) -> pd.Series:
    return (df["DepTime"].astype(float) // 100).clip(0, 24).astype(int)


def _keys_hour(df):          return _hour(df).astype(str)
def _keys_origin_hour(df):   return df["Origin"].astype(str) + "_" + _keys_hour(df)
def _keys_route_hour(df):    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str) + "_" + _keys_hour(df)
def _keys_dest_hour(df):     return df["Dest"].astype(str) + "_" + _keys_hour(df)
def _keys_carrier_hour(df):  return df["UniqueCarrier"].astype(str) + "_" + _keys_hour(df)


TE_SPECS = {
    "te_hour": _keys_hour,
    "te_origin_hour": _keys_origin_hour,
    "te_route_hour": _keys_route_hour,
    "te_dest_hour": _keys_dest_hour,
    "te_carrier_hour": _keys_carrier_hour,
}

# --- fit encoders on TRAIN ONLY ------------------------------------------------
TE_MAPS = {}
OOF_TE = {}
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
for name, keyfn in TE_SPECS.items():
    keys = keyfn(train).to_numpy()
    sums = pd.DataFrame({"k": keys, "y": ytr}).groupby("k")["y"].agg(["sum", "count"])
    TE_MAPS[name] = ((sums["sum"] + PRIOR * TE_M) / (sums["count"] + TE_M))
    oof = np.full(len(keys), PRIOR)
    for tri, tei in kf.split(keys):
        g = pd.DataFrame({"k": keys[tri], "y": ytr[tri]}).groupby("k")["y"].agg(["sum", "count"])
        mp = (g["sum"] + PRIOR * TE_M) / (g["count"] + TE_M)
        oof[tei] = pd.Series(keys[tei]).map(mp).fillna(PRIOR).to_numpy()
    OOF_TE[name] = oof

# --- categorical levels from train only ----------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Month", "DayofMonth", "DayOfWeek"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(float)
    hour = (dep // 100).clip(0, 24)
    tod = hour + (dep % 100) / 60.0
    X["hour"] = hour
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)
    X["minute"] = dep % 100
    X["month"], X["day"], X["dow"] = _cnum(df["Month"]), _cnum(df["DayofMonth"]), _cnum(df["DayOfWeek"])
    X["distance"] = df["Distance"].astype(float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].values, categories=CAT_LEVELS[c])  # unseen levels -> NaN
    for name, keyfn in TE_SPECS.items():
        X[name] = keyfn(df).map(TE_MAPS[name]).astype(float).fillna(PRIOR).to_numpy()
    return X


def prepare_train(df: pd.DataFrame) -> pd.DataFrame:
    """Training frame: same as prepare() but with out-of-fold TE values (no self-leakage)."""
    X = prepare(df)
    for name in OOF_TE:
        X[name] = OOF_TE[name]
    return X


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2500,
    max_depth=5,
    learning_rate=0.03,
    min_child_weight=1,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
    early_stopping_rounds=100,
)

t0 = time.time()
model.fit(prepare_train(train), ytr, eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
