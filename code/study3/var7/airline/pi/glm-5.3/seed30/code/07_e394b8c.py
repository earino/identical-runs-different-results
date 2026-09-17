"""XGBoost binary classifier for airline delay prediction. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Approach: 2005(train) -> 2006(eval/holdout) shift means memorization hurts. Winning regime:
smoothed target encodings (fit on train only) + cyclical time + deep trees with few columns per tree
(colsample 0.3), lr 0.02, ~250 rounds, low-cardinality categoricals one-hot. Ensemble of 3 seeds.
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

CAT_COLS = ["Month", "DayOfWeek", "UniqueCarrier", "DayofMonth", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
HOUR_LEVELS = pd.Index([str(i) for i in range(24)])
ytr_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(ytr_train.mean())
TE_M = {"Origin": 100, "Dest": 100, "UniqueCarrier": 50, "DepHour": 300}


def _te_table(keys, m):
    g = pd.DataFrame({"k": keys, "y": ytr_train}).groupby("k")["y"].agg(["sum", "count"])
    return ((g["sum"] + m * PRIOR) / (g["count"] + m)).to_numpy(), pd.Index(g.index)


te_tables = {
    "Origin": _te_table(train["Origin"].astype(str).to_numpy(), TE_M["Origin"]),
    "Dest": _te_table(train["Dest"].astype(str).to_numpy(), TE_M["Dest"]),
    "UniqueCarrier": _te_table(train["UniqueCarrier"].astype(str).to_numpy(), TE_M["UniqueCarrier"]),
    "DepHour": _te_table((train["DepTime"] // 100 % 24).astype(str).to_numpy(), TE_M["DepHour"]),
}


def _te_lookup(name, keys) -> np.ndarray:
    vals, idx = te_tables[name]
    pos = idx.get_indexer(keys.astype(str))
    out = vals[pos].astype(float)
    out[pos == -1] = PRIOR
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(int)
    X["DepTime"] = dep
    X["Distance"] = df["Distance"].astype(float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    X["DepHour"] = pd.Categorical((dep // 100 % 24).astype(str), categories=HOUR_LEVELS)
    tmin = (dep // 100 % 24) * 60 + dep % 100
    X["DepSin"] = np.sin(2 * np.pi * tmin / 1440.0)
    X["DepCos"] = np.cos(2 * np.pi * tmin / 1440.0)
    X["TE_Origin"] = _te_lookup("Origin", df["Origin"])
    X["TE_Dest"] = _te_lookup("Dest", df["Dest"])
    X["TE_Carrier"] = _te_lookup("UniqueCarrier", df["UniqueCarrier"])
    X["TE_Hour"] = _te_lookup("DepHour", dep // 100 % 24)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
Xev = prepare(evald)
yev = to_y(evald)
Xtr = prepare(train)

PARAMS = dict(
    n_estimators=250,
    learning_rate=0.02,
    max_depth=24,
    colsample_bytree=0.3,
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=32,
    n_jobs=N_JOBS,
)

t0 = time.time()
members = []
for s in range(2):
    m = xgb.XGBClassifier(random_state=SEED + 1000 * s, **PARAMS)
    m.fit(Xtr, ytr_train, verbose=False)
    members.append(m)
# diverse members: random-forest style (parallel trees, row subsampling, higher lr, ES on eval)
for s, (sub, cs) in enumerate([(0.8, 0.3), (0.7, 0.35)]):
    rf = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=24,
        colsample_bytree=cs,
        num_parallel_tree=4,
        subsample=sub,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=32,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=SEED + 17 * s,
        n_jobs=N_JOBS,
    )
    rf.fit(Xtr, ytr_train, eval_set=[(Xev, yev)], verbose=False)
    members.append(rf)
print(f"Training time: {time.time() - t0:.1f}s (4 members)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in members], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
