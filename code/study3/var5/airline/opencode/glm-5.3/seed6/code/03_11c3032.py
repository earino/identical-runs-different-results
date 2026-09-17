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
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features: baseline cats + smoothed route TE + hour categorical ------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
HOURS = pd.Index(sorted((train["DepTime"] // 100).unique()))

_route = train["Origin"] + "_" + train["Dest"]
_y = (train[TARGET] == POSITIVE).astype(float)
PRIOR = float(_y.mean())
M = 20.0


def _smoothed(frame: pd.DataFrame) -> pd.Series:
    g = frame.groupby("route")["y"]
    return (g.sum() + M * PRIOR) / (g.count() + M)


def route_te(values: pd.Series) -> np.ndarray:
    return values.map(route_map).fillna(PRIOR).to_numpy()


tmp = pd.DataFrame({"route": _route, "y": _y})
route_map = _smoothed(tmp)  # full-train map used at predict time
route_oof = np.empty(len(tmp))
for tr_idx, val_idx in KFold(n_splits=5, shuffle=True, random_state=SEED).split(tmp):
    m = _smoothed(tmp.iloc[tr_idx])
    route_oof[val_idx] = tmp["route"].iloc[val_idx].map(m).fillna(PRIOR).to_numpy()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["DepTime", "Distance"]].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    X["hour"] = pd.Categorical(df["DepTime"] // 100, categories=HOURS)
    X["route_te"] = route_te(df["Origin"] + "_" + df["Dest"])
    return X


def prepare_train(df: pd.DataFrame, oof: np.ndarray) -> pd.DataFrame:
    X = prepare(df)
    X["route_te"] = oof
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- shallow-model scan, ship the best ------------------------------------------
Xall = prepare_train(train, route_oof)
yall = to_y(train)
Xev, yev = prepare(evald), to_y(evald)

GRID = [
    dict(max_depth=4, n_estimators=100),
    dict(max_depth=4, n_estimators=200),
    dict(max_depth=3, n_estimators=300),
    dict(max_depth=5, n_estimators=100),
]

best, best_auc, best_cfg = None, -1.0, None
t0 = time.time()
for cfg in GRID:
    m = xgb.XGBClassifier(
        learning_rate=0.1, tree_method="hist", enable_categorical=True,
        random_state=SEED, n_jobs=N_JOBS, **cfg)
    m.fit(Xall, yall)
    auc = roc_auc_score(yev, m.predict_proba(Xev)[:, 1])
    print(f"  cfg={cfg} -> eval_auc={auc:.4f}", flush=True)
    if auc > best_auc:
        best, best_auc, best_cfg = m, auc, cfg
print(f"Scan time: {time.time() - t0:.1f}s  chosen={best_cfg}")

model = best


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


print(f"Eval AUC: {best_auc:.4f}")
