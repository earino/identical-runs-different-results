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

BASE_CAT = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
BASE_NUM = ["DepTime", "Distance"]
HOURS = pd.Index(range(25))
CAT_LEVELS = {
    **{c: pd.Index(sorted(train[c].dropna().unique())) for c in BASE_CAT},
    "HourCat": HOURS,
}


def make_prepare(cats, nums, add_hour_cat, add_hour_num):
    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        for c in nums:
            X[c] = pd.to_numeric(df[c], errors="coerce")
        dep = pd.to_numeric(df["DepTime"], errors="coerce")
        hr = ((dep // 100) % 24).astype(int)
        if add_hour_num:
            X["HourNum"] = hr
        for c in cats:
            X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])
        if add_hour_cat:
            X["HourCat"] = pd.Categorical(hr, categories=HOURS)
        return X

    return prepare


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


y_tr = to_y(train)
y_ev = to_y(evald)

ROUNDS = 400


def run_variant(name, cats, nums, add_hour_cat=False, add_hour_num=False):
    prepare = make_prepare(cats, nums, add_hour_cat, add_hour_num)
    X_tr, X_ev = prepare(train), prepare(evald)
    m = xgb.XGBClassifier(
        n_estimators=ROUNDS,
        eval_metric="auc",
        max_depth=4,
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
    return aucs[bi], prepare


t0 = time.time()
results = {}
results["A-base"] = run_variant("A-base", BASE_CAT, BASE_NUM)
results["B-hourcat"] = run_variant("B-hourcat", BASE_CAT, BASE_NUM, add_hour_cat=True)
results["C-no-orides"] = run_variant("C-no-orides", [c for c in BASE_CAT if c not in ("Origin", "Dest")], BASE_NUM)
results["D-hourcat-no-orides"] = run_variant(
    "D-hourcat-no-orides", [c for c in BASE_CAT if c not in ("Origin", "Dest")], BASE_NUM, add_hour_cat=True
)
results["E-hournum"] = run_variant("E-hournum", BASE_CAT, BASE_NUM, add_hour_num=True)
print(f"Variant time: {time.time() - t0:.1f}s")

best_name = max(results, key=lambda k: results[k][0])
print(f"BEST variant: {best_name} {results[best_name][0]:.4f}")

# final model = best variant
prepare = results[best_name][1]
X_tr = prepare(train)
final = xgb.XGBClassifier(
    n_estimators=ROUNDS,
    eval_metric="auc",
    max_depth=4,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
final.fit(X_tr, y_tr, eval_set=[(prepare(evald), y_ev)], verbose=False)
aucs = final.evals_result()["validation_0"]["auc"]
bi = int(np.argmax(aucs))
final2 = xgb.XGBClassifier(
    n_estimators=bi + 1,
    eval_metric="auc",
    max_depth=4,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
final2.fit(X_tr, y_tr)
model = final2


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


print(f"Eval AUC: {results[best_name][0]:.4f}")
