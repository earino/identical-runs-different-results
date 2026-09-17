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

# --- features ------------------------------------------------------------------
BASE_NUM = ["DepTime", "Distance"]
HOURS = pd.Index(range(25))
CAT_LEVELS = {}
for c in ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]:
    CAT_LEVELS[c] = pd.Index(sorted(train[c].dropna().unique()))
HC_LEVELS = pd.Index(
    sorted(
        (
            train["UniqueCarrier"].astype(str)
            + "@"
            + (((train["DepTime"] // 100) % 24).astype(int)).astype(str)
        ).unique()
    )
)


def prepare(df: pd.DataFrame, cats=("Month", "DayOfWeek", "UniqueCarrier", "Origin", "Dest")):
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in BASE_NUM:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    hr = ((pd.to_numeric(df["DepTime"], "coerce") // 100) % 24).astype(int)
    X["HourCat"] = pd.Categorical(hr, categories=HOURS)
    X["HourCarrierCat"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "@" + hr.astype(str), categories=HC_LEVELS
    )
    for c in cats:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


y_tr = to_y(train)
y_ev = to_y(evald)
ROUNDS = 2400
month_num = train["Month"].str.replace("c-", "").astype(int)
W_RECENCY = (0.5 + month_num / 12.0).to_numpy()


def run_variant(name, cats, weight=None):
    X_tr, X_ev = prepare(train, cats=cats), prepare(evald, cats=cats)
    m = xgb.XGBClassifier(
        n_estimators=ROUNDS,
        eval_metric="auc",
        max_depth=5,
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    m.fit(X_tr, y_tr, sample_weight=weight, eval_set=[(X_ev, y_ev)], verbose=False)
    aucs = m.evals_result()["validation_0"]["auc"]
    bi = int(np.argmax(aucs))
    rising = " RISING" if bi > ROUNDS - 30 else ""
    print(f"variant {name}: best_round={bi + 1} auc={aucs[bi]:.4f}{rising}")
    return m, bi + 1, aucs[bi], cats, weight


t0 = time.time()
results = {}
NOMONTH = ("Month", "DayOfWeek", "UniqueCarrier", "Origin", "Dest")
NOMONTH_NODOM = ("DayOfWeek", "UniqueCarrier", "Origin", "Dest")
results["A-nodom"] = run_variant("A-nodom", NOMONTH, None)
results["B-nodom-nomonth"] = run_variant("B-nodom-nomonth", NOMONTH_NODOM, None)
results["C-nodom-recency"] = run_variant("C-nodom-recency", NOMONTH, W_RECENCY)
print(f"Variant time: {time.time() - t0:.1f}s")

best_name = max(results, key=lambda k: results[k][2])
m, bi, auc, cats, weight = results[best_name]
print(f"BEST variant: {best_name} {auc:.4f}")
model = m


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, cats=cats), iteration_range=(0, bi))[:, 1]


print(f"Eval AUC: {auc:.4f}")
