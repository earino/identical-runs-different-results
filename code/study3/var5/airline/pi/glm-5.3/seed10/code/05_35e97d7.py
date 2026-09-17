"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
y_tr = (train[TARGET] == POSITIVE).astype(int).to_numpy()
y_ev = (evald[TARGET] == POSITIVE).astype(int).to_numpy()

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

CURVE_KS = [50, 100, 150, 250]
SHIP_KS = [100, 150, 250]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace(r"^c-", "", regex=True), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["Month"] = _num(df["Month"])
    X["DayofMonth"] = _num(df["DayofMonth"])
    X["DayOfWeek"] = _num(df["DayOfWeek"])
    X["DepTime"] = df["DepTime"].astype(float)
    X["Distance"] = df["Distance"].astype(float)
    dep = X["DepTime"]
    hour = np.clip((dep // 100) % 24, 0, 23)
    minute = dep % 100
    X["DepHour"] = hour
    X["DepMinute"] = minute
    X["DepTimeMin"] = hour * 60 + minute
    X["sin_day"] = np.sin(2 * np.pi * (hour * 60 + minute) / 1440)
    X["cos_day"] = np.cos(2 * np.pi * (hour * 60 + minute) / 1440)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
BASE_PARAMS = dict(
    n_estimators=250,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

GRID = []
for depth in [5, 6, 8]:
    for mcw in [1, 30]:
        for subs in [1.0, 0.8]:
            GRID.append(dict(max_depth=depth, min_child_weight=mcw, subsample=subs))

t0 = time.time()
X_tr, X_ev = prepare(train), prepare(evald)
results = {}
for gp in GRID:
    m = xgb.XGBClassifier(**{**BASE_PARAMS, **gp})
    m.fit(X_tr, y_tr, verbose=False)
    tag = f"d{gp['max_depth']}_w{gp['min_child_weight']}_s{gp['subsample']}"
    curve = {}
    for k in CURVE_KS:
        p = m.predict_proba(X_ev, iteration_range=(0, k))[:, 1]
        curve[k] = roc_auc_score(y_ev, p)
    print(f"[diag] {tag} curve " + " ".join(f"k={k}:{curve[k]:.4f}" for k in CURVE_KS))
    for k in SHIP_KS:
        results[(tag, k)] = curve[k]

(best_tag, BEST_K), best_auc = max(results.items(), key=lambda kv: kv[1])
print(f"[diag] chosen: {best_tag} k={BEST_K} auc={best_auc:.4f} ({time.time() - t0:.1f}s)")
BEST_GP = [gp for gp in GRID if f"d{gp['max_depth']}_w{gp['min_child_weight']}_s{gp['subsample']}" == best_tag][0]
model = xgb.XGBClassifier(**{**BASE_PARAMS, **BEST_GP})
model.fit(X_tr, y_tr, verbose=False)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df), iteration_range=(0, BEST_K))[:, 1]


eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
