"""XGBoost binary classifier for airline delay prediction. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Approach: the 2005(train) -> 2006(eval/holdout) shift means heavy memorization hurts. Winning regime:
smoothed target encodings (fit on train only) + cyclical time features + deep trees with few columns
per tree (colsample 0.25) and early stopping against the same-year eval set.
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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
# categorical levels + target-encoding tables learned from TRAIN only (unseen levels -> prior)
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
HOUR_LEVELS = pd.Index([str(i) for i in range(24)])
ytr_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(ytr_train.mean())
TE_M = {"Origin": 100, "Dest": 100, "UniqueCarrier": 50}


def _te_table(keys, m):
    g = pd.DataFrame({"k": keys, "y": ytr_train}).groupby("k")["y"].agg(["sum", "count"])
    return ((g["sum"] + m * PRIOR) / (g["count"] + m)).to_numpy(), pd.Index(g.index)


te_tables = {
    "Origin": _te_table(train["Origin"].astype(str).to_numpy(), TE_M["Origin"]),
    "Dest": _te_table(train["Dest"].astype(str).to_numpy(), TE_M["Dest"]),
    "UniqueCarrier": _te_table(train["UniqueCarrier"].astype(str).to_numpy(), TE_M["UniqueCarrier"]),
}
hour_keys_train = (train["DepTime"] // 100 % 24).astype(str).to_numpy()
te_tables["DepHour"] = _te_table(hour_keys_train, 300)


def _te_lookup(name, keys: pd.Series) -> np.ndarray:
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

model = xgb.XGBClassifier(
    n_estimators=1500,
    learning_rate=0.02,
    max_depth=24,
    colsample_bytree=0.3,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
    early_stopping_rounds=100,
)

t0 = time.time()
model.fit(prepare(train), ytr_train, eval_set=[(Xev, yev)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
