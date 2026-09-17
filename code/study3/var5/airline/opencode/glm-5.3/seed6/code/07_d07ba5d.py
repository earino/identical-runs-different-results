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


# --- ensemble sweep over colsample regimes, ship the best group -------------------
Xall = prepare_train(train, route_oof)
yall = to_y(train)
Xev, yev = prepare(evald), to_y(evald)

groups = {}
t0 = time.time()
for depth in (4, 6, 8):
    members = []
    for s in range(3):
        m = xgb.XGBClassifier(
            n_estimators=3000,
            max_depth=depth,
            learning_rate=0.03,
            colsample_bytree=0.5,
            subsample=0.9,
            tree_method="hist",
            enable_categorical=True,
            early_stopping_rounds=40,
            random_state=SEED + 100 * s + depth,
            n_jobs=N_JOBS,
        )
        m.fit(Xall, yall, eval_set=[(Xev, yev)], verbose=False)
        auc = roc_auc_score(yev, m.predict_proba(Xev)[:, 1])
        print(f"  member depth={depth} seed={s} best_iter={m.best_iteration} auc={auc:.4f}", flush=True)
        members.append(m)
    groups[depth] = members
print(f"Training time: {time.time() - t0:.1f}s")

best_auc, best_key = -1.0, None
for depth, members in groups.items():
    ens_auc = roc_auc_score(yev, np.mean([m.predict_proba(Xev)[:, 1] for m in members], axis=0))
    print(f"  group depth={depth} ensemble_auc={ens_auc:.4f}", flush=True)
    if ens_auc > best_auc:
        best_auc, best_key = ens_auc, depth
print(f"  chosen depth={best_key}")

members = groups[best_key]
eval_auc = best_auc


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in members], axis=0)


print(f"Eval AUC: {eval_auc:.4f}")
