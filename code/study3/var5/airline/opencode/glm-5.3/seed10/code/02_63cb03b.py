"""XGBoost binary classifier — airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
from sklearn.model_selection import train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

y_all = (train[TARGET] == POSITIVE).astype(int)
PRIOR = float(y_all.mean())

# --- encoders fit on TRAIN ONLY ------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Route"]
DOY_CUM = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])


def _route(df):
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


route_tr = _route(train)
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in ["UniqueCarrier", "Origin", "Dest"]}
cat_levels["Route"] = pd.Index(sorted(route_tr.unique()))

# count encoding (log1p of train frequency)
cnt_maps = {
    "Origin": train["Origin"].value_counts(),
    "Dest": train["Dest"].value_counts(),
    "UniqueCarrier": train["UniqueCarrier"].value_counts(),
    "Route": route_tr.value_counts(),
}

# smoothed target encoding from train
TE_M = 30.0
te_maps = {}
for c in ["Origin", "Dest", "UniqueCarrier", "Route"]:
    if c == "Route":
        s = route_tr
    else:
        s = train[c]
    g = y_all.groupby(s)
    te_maps[c] = ((g.sum() + TE_M * PRIOR) / (g.count() + TE_M)).astype(float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    month = df["Month"].str.slice(2).astype(int)
    day = df["DayofMonth"].str.slice(2).astype(int)
    X["month"] = month
    X["day"] = day
    X["dow"] = df["DayOfWeek"].str.slice(2).astype(int)
    doy = DOY_CUM[month - 1] + day
    X["doy"] = doy
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    dep = df["DepTime"].astype(int)
    X["DepTime"] = dep
    hour = (dep // 100) % 24
    minute = dep % 100
    tod = hour + minute / 60.0
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)
    dist = df["Distance"].astype(float)
    X["Distance"] = dist
    X["log_dist"] = np.log1p(dist)
    for c in ["UniqueCarrier", "Origin", "Dest"]:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    route = _route(df)
    X["Route"] = pd.Categorical(route, categories=cat_levels["Route"])
    for c, mp in cnt_maps.items():
        key = route if c == "Route" else df[c]
        X[c + "_cnt"] = np.log1p(key.map(mp).astype(float).fillna(0.0))
    for c, mp in te_maps.items():
        key = route if c == "Route" else df[c]
        X[c + "_te"] = key.map(mp).astype(float).fillna(PRIOR)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=3000,
    max_depth=6,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=50,
)

t0 = time.time()
X_all, y_all_arr = prepare(train), y_all.to_numpy()
Xtr, Xva, ytr, yva = train_test_split(X_all, y_all_arr, test_size=0.1, random_state=SEED, stratify=y_all_arr)
model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s, best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
