"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Feature engineering: base columns + cyclic hour/date encodings + out-of-fold smoothed target
encodings of categorical x hour / day-of-week interactions (fit on TRAIN ONLY).
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

# --- feature engineering -------------------------------------------------------
BASE_CAT = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in BASE_CAT}

y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(y_train.mean())
SM = 20.0  # smoothing for target encoding


def _num(s):
    return s.astype(str).str.replace("c-", "", regex=False).astype(int)


def _hour(df):
    return df["DepTime"].astype(int) // 100


# interaction keys, all derived from raw columns only
TE_SPECS = {
    "routeH": lambda d: d["Origin"].astype(str) + "_" + d["Dest"].astype(str) + "_" + _hour(d).astype(str),
    "carrierH": lambda d: d["UniqueCarrier"].astype(str) + "_" + _hour(d).astype(str),
    "originH": lambda d: d["Origin"].astype(str) + "_" + _hour(d).astype(str),
    "destH": lambda d: d["Dest"].astype(str) + "_" + _hour(d).astype(str),
    "routeDow": lambda d: d["Origin"].astype(str) + "_" + d["Dest"].astype(str) + "_" + d["DayOfWeek"].astype(str),
    "hourDow": lambda d: _hour(d).astype(str) + "_" + d["DayOfWeek"].astype(str),
    "originHDow": lambda d: d["Origin"].astype(str) + "_" + _hour(d).astype(str) + "_" + d["DayOfWeek"].astype(str),
    "destHDow": lambda d: d["Dest"].astype(str) + "_" + _hour(d).astype(str) + "_" + d["DayOfWeek"].astype(str),
}


def _group_stats(keys, y):
    g = pd.DataFrame({"c": pd.Series(keys).to_numpy(), "y": y}).groupby("c")["y"].agg(["sum", "count"])
    return g["sum"], g["count"]


# full-train smoothed TE + count maps (fit on train only)
TE_MAPS = {}
for name, kf in TE_SPECS.items():
    s, c = _group_stats(kf(train), y_train)
    TE_MAPS[name] = ((s + SM * PRIOR) / (c + SM), c)

# out-of-fold TE for the training rows themselves (honest training signal)
OOF = {}
rng = np.random.RandomState(SEED)
folds = np.array_split(rng.permutation(len(train)), 5)
for name, kf in TE_SPECS.items():
    keys = pd.Series(kf(train))
    oof = np.full(len(train), PRIOR, dtype=float)
    for f in folds:
        mask = np.ones(len(train), bool)
        mask[f] = False
        s, c = _group_stats(keys[mask], y_train[mask])
        oof[f] = keys.iloc[f].map((s + SM * PRIOR) / (c + SM)).fillna(PRIOR).to_numpy()
    OOF[name] = oof


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw DataFrame (any subset of train's columns, target optional) -> feature frame."""
    X = pd.DataFrame(index=df.index)
    for c in BASE_CAT:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    dt = df["DepTime"].astype(int)
    X["DepTime"] = dt.astype(float)
    X["Distance"] = df["Distance"].astype(float)
    h = dt // 100
    min_of_day = (h * 60 + dt % 100).astype(float)
    X["Hour"] = h.astype(float)
    X["MinOfDay"] = min_of_day
    X["sin_h"] = np.sin(2 * np.pi * min_of_day / 1440)
    X["cos_h"] = np.cos(2 * np.pi * min_of_day / 1440)
    doy = (_num(df["Month"]) * 31 + _num(df["DayofMonth"])).astype(float)
    X["sin_doy"] = np.sin(2 * np.pi * doy / 365)
    X["cos_doy"] = np.cos(2 * np.pi * doy / 365)
    for name, kf in TE_SPECS.items():
        k = pd.Series(kf(df), index=df.index)
        te_map, cnt_map = TE_MAPS[name]
        X["TE_" + name] = k.map(te_map).fillna(PRIOR).to_numpy()
        X["N_" + name] = k.map(cnt_map).fillna(0).astype(float).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
)

t0 = time.time()
X_train = prepare(train)
for name in TE_SPECS:  # replace in-sample TE with out-of-fold TE for the training rows
    X_train["TE_" + name] = OOF[name]
model.fit(X_train, y_train, verbose=False)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
