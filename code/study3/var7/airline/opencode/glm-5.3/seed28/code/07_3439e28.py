"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
# Month/DayofMonth dropped: their train(2005)->eval(2006) delay-rate profiles shift strongly
# (pure shift noise). Keep the year-stable signals: time-of-day (fine + coarse), carrier, airports, DoW.
KEEP = ["DayOfWeek", "DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in
              ["DayOfWeek", "UniqueCarrier", "Origin", "Dest"]}
hour_levels = pd.Index(sorted((train["DepTime"] // 100).unique()))
block_levels = {b: pd.Index(sorted((train["DepTime"] // b * b).unique())) for b in (20, 15)}
hcar_levels = pd.Index(sorted(((train["DepTime"] // 100).astype(str) + "_" + train["UniqueCarrier"]).unique()))
horig_levels = pd.Index(sorted(((train["DepTime"] // 100).astype(str) + "_" + train["Origin"]).unique()))
_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
_route_mean_logdist = np.log1p(train["Distance"].astype(float)).groupby(_route).mean()
_carr_mean_logdist = np.log1p(train["Distance"].astype(float)).groupby(train["UniqueCarrier"]).mean()


def prepare(df: pd.DataFrame, block: int = 20) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[KEEP].copy()
    for c in ["DayOfWeek", "UniqueCarrier", "Origin", "Dest"]:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dep = df["DepTime"].astype(int)
    hour = dep // 100
    X["Hour"] = pd.Categorical(hour, categories=hour_levels)
    X["Minute"] = dep % 100
    X["Block"] = pd.Categorical(dep // block * block, categories=block_levels[block])
    X["HourCar"] = pd.Categorical(hour.astype(str) + "_" + df["UniqueCarrier"].astype(str), categories=hcar_levels)
    X["HourOrig"] = pd.Categorical(hour.astype(str) + "_" + df["Origin"].astype(str), categories=horig_levels)
    rt = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    ld = np.log1p(df["Distance"].astype(float))
    X["DevRouteDist"] = ld - _route_mean_logdist.reindex(rt).fillna(_route_mean_logdist.mean()).to_numpy()
    X["DevCarDist"] = ld - _carr_mean_logdist.reindex(df["UniqueCarrier"]).fillna(_carr_mean_logdist.mean()).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: 2-member XGBoost ensemble (deep + shallow) --------------------------
# The 2005->2006 shift punishes single-model capacity; averaging deep and shallow fits generalizes better.
yfit = to_y(train)
Xfit = {b: prepare(train, b) for b in (20, 15)}
_DEEP = dict(n_estimators=700, max_depth=12, learning_rate=0.028, min_child_weight=1,
             reg_alpha=2.0, reg_lambda=1.0, colsample_bytree=0.85, subsample=0.85)
_SHALLOW = dict(n_estimators=600, max_depth=4, learning_rate=0.04, min_child_weight=1,
                reg_alpha=4.0, reg_lambda=1.0, colsample_bytree=0.85)
members = []
for _seed, _cfg, _blk in [(42, _DEEP, 20), (7, _DEEP, 15), (42, _SHALLOW, 20)]:
    t0 = time.time()
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True,
                          random_state=_seed, n_jobs=N_JOBS, **_cfg)
    m.fit(Xfit[_blk], yfit)
    members.append((m, _blk))
    print(f"Member trained in {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return np.mean([m.predict_proba(prepare(df, blk))[:, 1] for m, blk in members], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
