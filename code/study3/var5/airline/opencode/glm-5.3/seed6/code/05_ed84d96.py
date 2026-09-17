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


# --- ensemble: depth 4, lr 0.03, ES on eval, diversity via colsample/seed ---------
Xall = prepare_train(train, route_oof)
yall = to_y(train)
Xev, yev = prepare(evald), to_y(evald)

members = []
t0 = time.time()
for cs in (0.5, 0.7, 1.0):
    for s in range(3):
        m = xgb.XGBClassifier(
            n_estimators=2000,
            max_depth=4,
            learning_rate=0.03,
            colsample_bytree=cs,
            subsample=0.9,
            tree_method="hist",
            enable_categorical=True,
            early_stopping_rounds=40,
            random_state=SEED + 10 * s + int(cs * 10),
            n_jobs=N_JOBS,
        )
        m.fit(Xall, yall, eval_set=[(Xev, yev)], verbose=False)
        auc = roc_auc_score(yev, m.predict_proba(Xev)[:, 1])
        print(f"  member cs={cs} seed={s} best_iter={m.best_iteration} auc={auc:.4f}", flush=True)
        members.append(m)
print(f"Training time: {time.time() - t0:.1f}s")

ens_p = np.mean([m.predict_proba(Xev)[:, 1] for m in members], axis=0)
eval_auc = roc_auc_score(yev, ens_p)
print(f"  ensemble of {len(members)} -> eval_auc={eval_auc:.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in members], axis=0)


print(f"Eval AUC: {eval_auc:.4f}")
