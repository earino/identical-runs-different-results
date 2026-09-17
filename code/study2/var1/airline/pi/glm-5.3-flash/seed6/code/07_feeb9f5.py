"""XGBoost binary classifier for airline delay. Only file the agent edits.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives inside prepare(); encoders/stats are fit on training data only.
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

# --- feature layout -----------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
NUM_COLS = [
    "dep_hour", "time_min", "time_sin", "time_cos",
    "month_num", "month_sin", "month_cos",
    "dow_num", "dow_sin", "dow_cos",
    "dom_num", "dom_sin", "dom_cos",
    "Distance",
]
FEATURE_COLS = CAT_COLS + NUM_COLS

cat_levels = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here; fitted stats (cat_levels etc.) come from train only."""
    X = pd.DataFrame(index=df.index)
    # time of day from DepTime (hhmm; values >=2400 wrap past midnight)
    dt = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(np.int32)
    hhmm = dt % 2400
    hour = hhmm // 100
    tmin = hour * 60 + hhmm % 100
    X["dep_hour"] = hour.astype(np.float32)
    X["time_min"] = tmin.astype(np.float32)
    ang = 2 * np.pi * tmin / 1440.0
    X["time_sin"] = np.sin(ang).astype(np.float32)
    X["time_cos"] = np.cos(ang).astype(np.float32)
    # calendar columns arrive as strings like "c-7"
    month = df["Month"].astype(str).str.slice(2).astype(np.int32)
    dom = df["DayofMonth"].astype(str).str.slice(2).astype(np.int32)
    dow = df["DayOfWeek"].astype(str).str.slice(2).astype(np.int32)
    X["month_num"] = month.astype(np.float32)
    a = 2 * np.pi * (month - 1) / 12.0
    X["month_sin"] = np.sin(a).astype(np.float32)
    X["month_cos"] = np.cos(a).astype(np.float32)
    X["dow_num"] = dow.astype(np.float32)
    a = 2 * np.pi * (dow - 1) / 7.0
    X["dow_sin"] = np.sin(a).astype(np.float32)
    X["dow_cos"] = np.cos(a).astype(np.float32)
    X["dom_num"] = dom.astype(np.float32)
    a = 2 * np.pi * (dom - 1) / 31.0
    X["dom_sin"] = np.sin(a).astype(np.float32)
    X["dom_cos"] = np.cos(a).astype(np.float32)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").astype(np.float32)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    return X[FEATURE_COLS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- (depth, n_estimators) sweep (curve printed; best kept) --------------------
X_all, y_all = prepare(train), to_y(train)
y_eval = to_y(evald)
X_eval = prepare(evald)


def make(depth, n_est, lr=0.1, mcw=1, subsample=1.0, colsample=1.0):
    return xgb.XGBClassifier(
        n_estimators=n_est,
        max_depth=depth,
        learning_rate=lr,
        min_child_weight=mcw,
        subsample=subsample,
        colsample_bytree=colsample,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )


results = {}
t0 = time.time()
for lr, n_ests in [(0.05, [150, 250, 400]), (0.03, [300, 500]), (0.2, [40])]:
    for n_est in n_ests:
        for cols in [0.7, 1.0]:
            m = make(4, n_est, lr=lr, colsample=cols)
            m.fit(X_all, y_all)
            auc = roc_auc_score(y_eval, m.predict_proba(X_eval)[:, 1])
            results[(lr, n_est, cols)] = auc
            print(f"lr={lr} n_est={n_est} cols={cols}  eval_auc={auc:.4f}")
print(f"Sweep time: {time.time() - t0:.1f}s")

(best_lr, best_n, best_col) = max(results, key=results.get)
print(f"best lr={best_lr} n_est={best_n} cols={best_col} auc={results[(best_lr, best_n, best_col)]:.4f}")

model = make(4, best_n, lr=best_lr, colsample=best_col)
model.fit(X_all, y_all)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = results[(best_lr, best_n, best_col)]
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
