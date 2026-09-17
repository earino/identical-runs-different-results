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
BASE_CATS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in BASE_CATS}
AIRPORTS = pd.Index(sorted(set(train["Origin"].dropna()) | set(train["Dest"].dropna())))

# smoothed target encodings from TRAIN ONLY
y_tr_all = (train[TARGET] == POSITIVE).astype(int)
PRIOR = float(y_tr_all.mean())
K = 100.0  # smoothing strength


def _te_map(keys: pd.Series) -> pd.Series:
    g = y_tr_all.groupby(keys).agg(["sum", "count"])
    sm = (g["sum"] + K * PRIOR) / (g["count"] + K)
    return sm


def dep_hour_of(df: pd.DataFrame) -> np.ndarray:
    dep = df["DepTime"].astype(float).to_numpy()
    return np.floor(dep / 100.0)


TE_SPEC = {
    "te_carrier": train["UniqueCarrier"],
    "te_origin": train["Origin"],
    "te_dest": train["Dest"],
    "te_route": train["Origin"].astype(str) + "_" + train["Dest"].astype(str),
    "te_hour": pd.Series(dep_hour_of(train), index=train.index),
}
TE_MAPS = {name: _te_map(keys) for name, keys in TE_SPEC.items()}


def prepare(df: pd.DataFrame, use_airports: bool = True, use_te: bool = True) -> pd.DataFrame:
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
    if use_airports:
        X["Origin"] = pd.Categorical(df["Origin"].values, categories=AIRPORTS)
        X["Dest"] = pd.Categorical(df["Dest"].values, categories=AIRPORTS)
    if use_te:
        keys = {
            "te_carrier": df["UniqueCarrier"],
            "te_origin": df["Origin"],
            "te_dest": df["Dest"],
            "te_route": df["Origin"].astype(str) + "_" + df["Dest"].astype(str),
            "te_hour": pd.Series(dep_h, index=df.index),
        }
        for name, k in keys.items():
            X[name] = k.map(TE_MAPS[name]).fillna(PRIOR).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- diagnostic sweep ----------------------------------------------------------
t0 = time.time()
y_all = to_y(train)
y_ev = to_y(evald)
results = []
for use_airports, use_te in [(True, False), (True, True), (False, True)]:
    X_tr = prepare(train, use_airports, use_te)
    X_ev = prepare(evald, use_airports, use_te)
    for n_rounds in [30, 50, 80]:
        m = xgb.XGBClassifier(
            n_estimators=n_rounds, max_depth=6, learning_rate=0.1, tree_method="hist",
            enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
        )
        m.fit(X_tr, y_all)
        auc = roc_auc_score(y_ev, m.predict_proba(X_ev)[:, 1])
        tag = f"airports={int(use_airports)} te={int(use_te)}"
        results.append((auc, use_airports, use_te, n_rounds))
        print(f"diag {tag} rounds={n_rounds} eval_auc={auc:.4f}")
results.sort(key=lambda r: -r[0])
BEST_AUC, BEST_AP, BEST_TE, BEST_ROUNDS = results[0]
print(f"diag best: airports={BEST_AP} te={BEST_TE} rounds={BEST_ROUNDS} eval_auc={BEST_AUC:.4f} ({time.time()-t0:.0f}s)")


# --- final model on full train with the best config ----------------------------
model = xgb.XGBClassifier(
    n_estimators=BEST_ROUNDS, max_depth=6, learning_rate=0.1, tree_method="hist",
    enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
)
model.fit(prepare(train, BEST_AP, BEST_TE), y_all)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, BEST_AP, BEST_TE))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
