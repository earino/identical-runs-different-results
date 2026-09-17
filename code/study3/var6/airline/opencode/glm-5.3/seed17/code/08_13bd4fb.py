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

# --- fitted on TRAIN only -------------------------------------------------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
cat_levels["hour_cat"] = pd.Index([str(h) for h in range(0, 28)])
_ch_tr = train["UniqueCarrier"].astype(str) + "_" + np.floor(pd.to_numeric(train["DepTime"]) / 100).astype(int).astype(str)
cat_levels["carrier_hour"] = pd.Index(sorted(_ch_tr.unique()))
cat_levels["slot30"] = pd.Index([str(s) for s in range(0, 90)])
cat_levels["slot15t"] = pd.Index([str(s) for s in range(0, 112)])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    X["DepTime"] = dep
    X["hour"] = np.floor(dep / 100.0)
    X["minute"] = dep % 100.0
    X["hour_cat"] = pd.Categorical(np.floor(dep / 100.0).astype("Int64").astype(str), categories=cat_levels["hour_cat"])
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["slot30"] = pd.Categorical(
        np.floor(dep / 30).fillna(-1).astype(int).astype(str), categories=cat_levels["slot30"]
    )
    tod = np.floor(dep / 100) * 60 + dep % 100
    X["slot15"] = pd.Categorical(
        np.floor(tod / 15).fillna(-1).astype(int).astype(str), categories=cat_levels["slot15t"]
    )
    ch = df["UniqueCarrier"].astype(str) + "_" + np.floor(dep / 100).fillna(-1).astype(int).astype(str)
    X["carrier_hour"] = pd.Categorical(ch, categories=cat_levels["carrier_hour"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def make_model(**kw):
    p = dict(
        n_estimators=4000,
        max_depth=8,
        learning_rate=0.03,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=100,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    p.update(kw)
    return xgb.XGBClassifier(**p)


Xall = prepare(train)
yall = to_y(train)
X_tr, X_val, y_tr, y_val = train_test_split(Xall, yall, test_size=0.1, random_state=SEED, stratify=yall)

t0 = time.time()
# 1) find the optimal number of rounds on an internal validation split
probe = make_model()
probe.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
best_iter = probe.best_iteration
print(f"Probe: best_iter={best_iter}  ({time.time() - t0:.1f}s)")

# 2) retrain on the FULL training set at that iteration count, several seeds, and average
models = []
for seed in (42, 7, 123):
    m = make_model(n_estimators=best_iter + 1, early_stopping_rounds=None, random_state=seed)
    m.fit(Xall, yall)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  ({len(models)} seed models @ {best_iter + 1} rounds)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
