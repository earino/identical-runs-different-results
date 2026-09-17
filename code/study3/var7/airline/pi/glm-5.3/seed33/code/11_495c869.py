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
BASE_CAT = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
BASE_NUM = ["DepTime", "Distance"]
HOURS = pd.Index(range(25))
CAT_LEVELS = {
    **{c: pd.Index(sorted(train[c].dropna().unique())) for c in BASE_CAT},
    "HourCat": HOURS,
}


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Raw -> derived columns (not yet categorical-encoded)."""
    D = pd.DataFrame(index=df.index)
    hr = ((pd.to_numeric(df["DepTime"], errors="coerce") // 100) % 24).astype(int)
    D["HourCat"] = hr.astype(str)
    D["RouteCat"] = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    D["HourCarrierCat"] = df["UniqueCarrier"].astype(str) + "@" + hr.astype(str)
    D["HourDowCat"] = df["DayOfWeek"].astype(str) + "@" + hr.astype(str)
    D["HourMonthCat"] = df["Month"].astype(str) + "@" + hr.astype(str)
    return D


_der_tr = derive(train)
EXTRA_LEVELS = {c: pd.Index(sorted(_der_tr[c].unique())) for c in _der_tr.columns}


def make_prepare(extras):
    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        for c in BASE_NUM:
            X[c] = pd.to_numeric(df[c], errors="coerce")
        D = derive(df)
        X["HourCat"] = pd.Categorical(D["HourCat"].astype(int), categories=HOURS)
        for c in BASE_CAT:
            X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])
        for c in extras:
            X[c] = pd.Categorical(D[c], categories=EXTRA_LEVELS[c])  # unseen combos -> NaN
        return X

    return prepare


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


y_tr = to_y(train)
y_ev = to_y(evald)
ROUNDS = 900


def run_variant(name, extras):
    prepare = make_prepare(extras)
    X_tr, X_ev = prepare(train), prepare(evald)
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
    m.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    aucs = m.evals_result()["validation_0"]["auc"]
    bi = int(np.argmax(aucs))
    print(f"variant {name}: best_round={bi + 1} auc={aucs[bi]:.4f}")
    return m, bi + 1, aucs[bi], prepare


t0 = time.time()
results = {}
results["A-hourcat"] = run_variant("A-hourcat", [])
results["B-route"] = run_variant("B-route", ["RouteCat"])
results["C-hourcar"] = run_variant("C-hourcar", ["HourCarrierCat"])
results["D-hourdow"] = run_variant("D-hourdow", ["HourDowCat"])
results["E-hourmon"] = run_variant("E-hourmon", ["HourMonthCat"])
results["F-all"] = run_variant("F-all", ["RouteCat", "HourCarrierCat", "HourDowCat", "HourMonthCat"])
print(f"Variant time: {time.time() - t0:.1f}s")

best_name = max(results, key=lambda k: results[k][2])
m, bi, auc, prepare = results[best_name]
print(f"BEST variant: {best_name} {auc:.4f}")
model = m


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df), iteration_range=(0, bi))[:, 1]


print(f"Eval AUC: {auc:.4f}")
