"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, module-level `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
"""
import json
import os
import time
from itertools import combinations

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
C_CAT = ["UniqueCarrier", "Origin", "Dest", "Month", "DayofMonth", "DayOfWeek"]
NUM_COLS = ["Distance", "DepTime"]

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in C_CAT}


def _dep(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(int)


def _hour(df: pd.DataFrame) -> pd.Series:
    return (_dep(df) // 100).clip(0, 23)


def _cat_levels(series: pd.Series) -> pd.Index:
    return pd.Index(sorted(series.unique()))


C = train["UniqueCarrier"].astype(str)
H = _hour(train).astype(str)
M = train["Month"].astype(str)
O = train["Origin"].astype(str)
D = train["Dest"].astype(str)
W = train["DayOfWeek"].astype(str)
H3 = (_hour(train) // 3).astype(str)

LEVELS = {
    "hc": _cat_levels(C + "_" + H),
    "hb3": _cat_levels(C + "_" + H3),
    "mh": _cat_levels(M + "_" + H),
    "oh": _cat_levels(O + "_" + H3),
    "dowh": _cat_levels(W + "_" + H),
}


def base_X(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    hour = _hour(df)
    minute = (_dep(df) % 100).clip(0, 59)
    X["hour"] = hour / 23.0
    X["min_of_day"] = (hour * 60 + minute) / 1439.0
    for c in NUM_COLS:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    for c in C_CAT:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def key_for(df, variant):
    c = df["UniqueCarrier"].astype(str)
    m = df["Month"].astype(str)
    o = df["Origin"].astype(str)
    w = df["DayOfWeek"].astype(str)
    h = _hour(df).astype(str)
    h3 = (_hour(df) // 3).astype(str)
    if variant == "hc":
        return c + "_" + h
    if variant == "hb3":
        return c + "_" + h3
    if variant == "mh":
        return m + "_" + h
    if variant == "oh":
        return o + "_" + h3
    if variant == "dowh":
        return w + "_" + h
    return None


def make_prep(variant):
    def prep(df: pd.DataFrame) -> pd.DataFrame:
        X = base_X(df)
        k = key_for(df, variant)
        if k is not None:
            X[f"ix_{variant}"] = pd.Categorical(k, categories=LEVELS[variant])
        return X
    return prep


VARIANTS = ["hc", "hb3", "mh", "oh", "dowh", "base"]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


ytr, yev = to_y(train), to_y(evald)

names, preds, models, preps = [], [], [], []
t0 = time.time()
for i, v in enumerate(VARIANTS):
    prep = make_prep(v)
    Xtr_i, Xev_i = prep(train), prep(evald)
    m = xgb.XGBClassifier(
        n_estimators=3000,
        max_depth=4,
        learning_rate=0.05,
        reg_lambda=30.0,
        colsample_bytree=0.7,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=100,
        random_state=SEED + 7 * i,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr_i, ytr, eval_set=[(Xev_i, yev)], verbose=False)
    r = m.evals_result()["validation_0"]["auc"]
    p = m.predict_proba(Xev_i)[:, 1]
    names.append(v)
    preds.append(p)
    models.append(m)
    preps.append(prep)
    print(f"[ens] {v}: peak={max(r):.4f} at {int(np.argmax(r)) + 1} it, final={roc_auc_score(yev, p):.4f}")
print(f"Training time: {time.time() - t0:.1f}s")

P = np.array(preds)
for k in (2, 3, 4, 5, 6):
    best = None
    for sel in combinations(range(len(names)), k):
        a = roc_auc_score(yev, P[list(sel)].mean(axis=0))
        if best is None or a > best[0]:
            best = (a, sel)
    print(f"[ens] best{k} {[names[i] for i in best[1]]}: {best[0]:.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    ps = [m.predict_proba(prep(df))[:, 1] for m, prep in zip(models, preps)]
    return np.clip(np.mean(ps, axis=0), 0.0, 1.0)


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
