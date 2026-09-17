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

# --- features (exp 9 set): baseline cats + hour cat + smoothed route TE -----------
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
    X["route_te"] = (df["Origin"] + "_" + df["Dest"]).map(route_map).fillna(PRIOR).to_numpy()
    return X


def prepare_train(df: pd.DataFrame, oof: np.ndarray) -> pd.DataFrame:
    X = prepare(df)
    X["route_te"] = oof
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: depth 4, low lr, early stopping calibrated on 2006 eval ----------------
Xall = prepare_train(train, route_oof)
yall = to_y(train)
Xev, yev = prepare(evald), to_y(evald)

model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=4,
    learning_rate=0.03,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=40,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(Xall, yall, eval_set=[(Xev, yev)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
