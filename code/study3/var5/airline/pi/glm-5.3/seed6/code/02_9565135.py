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
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "route"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS[:6]}
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
    X["route"] = (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).values
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].values if c != "route" else X["route"].values,
                              categories=cat_levels[c])  # unseen levels -> NaN
    X["route"] = pd.Categorical(X["route"], categories=cat_levels["route"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
N_ROUNDS = 600
model = xgb.XGBClassifier(
    n_estimators=N_ROUNDS,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_all = prepare(train)
y_all = to_y(train)
X_ev = prepare(evald)
y_ev = to_y(evald)
rng = np.random.RandomState(SEED)
val_idx = rng.rand(len(X_all)) < 0.1
X_tr, y_tr = X_all[~val_idx], y_all[~val_idx]
X_val, y_val = X_all[val_idx], y_all[val_idx]
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s")

for it in [50, 100, 200, 300, 400, 600]:
    p_val = model.predict_proba(X_val, iteration_range=(0, it))[:, 1]
    p_ev = model.predict_proba(X_ev, iteration_range=(0, it))[:, 1]
    print(f"diag iter={it} val_auc={roc_auc_score(y_val, p_val):.4f} eval_auc={roc_auc_score(y_ev, p_ev):.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df), iteration_range=(0, N_ROUNDS))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
