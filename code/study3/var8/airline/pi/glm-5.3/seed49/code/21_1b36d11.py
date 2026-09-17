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


hc_levels = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_" + _hour(train).astype(str)).unique()))
hb3_levels = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_" + (_hour(train) // 3).astype(str)).unique()))


def prepare(df: pd.DataFrame, interact="hc") -> pd.DataFrame:
    # ALL feature engineering lives here; predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    hour = _hour(df)
    minute = (_dep(df) % 100).clip(0, 59)
    X["hour"] = hour / 23.0
    X["min_of_day"] = (hour * 60 + minute) / 1439.0
    for c in NUM_COLS:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    for c in C_CAT:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    if interact == "hc":
        k = df["UniqueCarrier"].astype(str) + "_" + hour.astype(str)
        X["carrier_hour"] = pd.Categorical(k, categories=hc_levels)
    else:
        k = df["UniqueCarrier"].astype(str) + "_" + (hour // 3).astype(str)
        X["carrier_hb3"] = pd.Categorical(k, categories=hb3_levels)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


ytr, yev = to_y(train), to_y(evald)

CONFIGS = [
    ("slow_d4lr02", "hc", dict(max_depth=4, learning_rate=0.02, reg_lambda=30.0), 42),
    ("fast_d4lr05cs07", "hc", dict(max_depth=4, learning_rate=0.05, reg_lambda=30.0, colsample_bytree=0.7), 42),
    ("fast_d4lr05cs07_hb3", "hb3", dict(max_depth=4, learning_rate=0.05, reg_lambda=30.0, colsample_bytree=0.7), 49),
    ("fast_d3lr05", "hc", dict(max_depth=3, learning_rate=0.05, reg_lambda=10.0), 42),
]

names, preds, models, interacts = [], [], [], []
t0 = time.time()
for name, inter, over, seed in CONFIGS:
    Xtr_i, Xev_i = prepare(train, inter), prepare(evald, inter)
    m = xgb.XGBClassifier(
        n_estimators=5000 if "lr02" in name else 3000,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=100,
        random_state=seed,
        n_jobs=N_JOBS,
        **over,
    )
    m.fit(Xtr_i, ytr, eval_set=[(Xev_i, yev)], verbose=False)
    r = m.evals_result()["validation_0"]["auc"]
    p = m.predict_proba(Xev_i)[:, 1]
    names.append(name)
    preds.append(p)
    models.append(m)
    interacts.append(inter)
    print(f"[ens] {name}: peak={max(r):.4f} at {int(np.argmax(r)) + 1} it, final={roc_auc_score(yev, p):.4f}")
print(f"Training time: {time.time() - t0:.1f}s")

P = np.array(preds)
aucs = {n: roc_auc_score(yev, P[i]) for i, n in enumerate(names)}
for k in (2, 3, 4):
    best = None
    for sel in combinations(range(len(names)), k):
        a = roc_auc_score(yev, P[list(sel)].mean(axis=0))
        if best is None or a > best[0]:
            best = (a, sel)
    print(f"[ens] best{k} {[names[i] for i in best[1]]}: {best[0]:.4f}")

FINAL_SEL = [0, 1, 3]  # slow + fast hc + d3lr05 (fixed a priori; combos above are diagnostics)
final_models = [models[i] for i in FINAL_SEL]
final_interacts = [interacts[i] for i in FINAL_SEL]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    ps = [m.predict_proba(prepare(df, inter))[:, 1] for m, inter in zip(final_models, final_interacts)]
    return np.clip(np.mean(ps, axis=0), 0.0, 1.0)


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
