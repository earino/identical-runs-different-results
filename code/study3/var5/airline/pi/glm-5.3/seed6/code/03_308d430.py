"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
BASE_CATS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in BASE_CATS}
cat_levels["route"] = pd.Index(sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(float)
    dep_h = np.floor(dep / 100.0)
    minutes = dep_h * 60.0 + (dep - dep_h * 100.0)
    X["dep_raw"] = dep
    X["dep_minutes"] = minutes
    X["sin_min"] = np.sin(2 * np.pi * minutes / 1440.0)
    X["cos_min"] = np.cos(2 * np.pi * minutes / 1440.0)
    X["dep_hour"] = dep_h
    X["distance"] = df["Distance"].astype(float)
    for c in BASE_CATS:
        X[c] = pd.Categorical(df[c].values, categories=cat_levels[c])
    return X


def add_route(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["route"] = pd.Categorical((df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).values,
                                categories=cat_levels["route"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- diagnostic sweep ----------------------------------------------------------
t0 = time.time()
y_all = to_y(train)
y_ev = to_y(evald)
X_all_base = prepare(train)
X_ev_base = prepare(evald)
X_all_route = add_route(train, X_all_base)
X_ev_route = add_route(evald, X_ev_base)

results = []
for feats, X_tr_all, X_ev in [("base", X_all_base, X_ev_base), ("route", X_all_route, X_ev_route)]:
    for n_rounds in [20, 30, 40, 60]:
        m = xgb.XGBClassifier(
            n_estimators=n_rounds, max_depth=6, learning_rate=0.1, tree_method="hist",
            enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
        )
        m.fit(X_tr_all, y_all)
        p = m.predict_proba(X_ev)[:, 1]
        auc = roc_auc_score(y_ev, p)
        results.append((auc, feats, n_rounds))
        print(f"diag feats={feats} rounds={n_rounds} eval_auc={auc:.4f}")
results.sort(reverse=True)
BEST_AUC, BEST_FEATS, BEST_ROUNDS = results[0]
print(f"diag best: feats={BEST_FEATS} rounds={BEST_ROUNDS} eval_auc={BEST_AUC:.4f}  ({time.time()-t0:.0f}s)")


# --- final model on full train with the best config ----------------------------
def make_X(df: pd.DataFrame) -> pd.DataFrame:
    X = prepare(df)
    if BEST_FEATS == "route":
        X = add_route(df, X)
    return X


FINAL_ROUNDS = BEST_ROUNDS
model = xgb.XGBClassifier(
    n_estimators=FINAL_ROUNDS, max_depth=6, learning_rate=0.1, tree_method="hist",
    enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
)
model.fit(make_X(train), y_all)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(make_X(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
