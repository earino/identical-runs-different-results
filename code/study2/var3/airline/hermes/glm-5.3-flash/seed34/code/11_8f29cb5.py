"""XGBoost binary classifier for the airline delay task.

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

# --- feature spec (fitted on train only) --------------------------------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _num(s):
    return s.astype(str).str.replace("c-", "", regex=False).astype(float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    X["Month_n"] = _num(df["Month"])
    X["Day_n"] = _num(df["DayofMonth"])
    X["DOW_n"] = _num(df["DayOfWeek"])
    dep = df["DepTime"].astype(float)
    X["DepTime"] = dep
    X["Dep_hour"] = pd.Categorical(
        ((dep // 100) % 24).astype(int).astype(str),
        categories=[str(i) for i in range(24)],
    )
    X["Dep_min"] = dep % 100
    X["Dep_minOfDay"] = (dep // 100) * 60 + dep % 100
    X["IsLateNight"] = ((dep >= 2350) | (dep < 600)).astype(float)
    X["Distance"] = df["Distance"].astype(float)
    X["Dist_log"] = np.log1p(df["Distance"].astype(float))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=1600,
    learning_rate=0.01,
    max_depth=6,
    min_child_weight=50,
    subsample=0.7,
    colsample_bytree=0.6,
    reg_lambda=15.0,
    reg_alpha=2.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
)

t0 = time.time()
X_full = prepare(train)
y_full = to_y(train)
SEEDS = [42, 7, 2026]
models = []
for s in SEEDS:
    m = xgb.XGBClassifier(**{**model.get_params(), "random_state": s})
    m.fit(X_full, y_full)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Ps = [m.predict_proba(prepare(df))[:, 1] for m in models]
    return np.mean(Ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
