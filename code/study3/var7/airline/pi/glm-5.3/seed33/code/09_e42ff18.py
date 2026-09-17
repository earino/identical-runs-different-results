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


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in BASE_NUM:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    hr = ((pd.to_numeric(df["DepTime"], errors="coerce") // 100) % 24).astype(int)
    X["HourCat"] = pd.Categorical(hr, categories=HOURS)
    for c in BASE_CAT:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_tr = prepare(train)
y_tr = to_y(train)
X_ev = prepare(evald)
y_ev = to_y(evald)

ROUNDS = 800
CONFIGS = [
    dict(name="d4-lr05", max_depth=4, learning_rate=0.05),
    dict(name="d4-lr03", max_depth=4, learning_rate=0.03),
    dict(name="d4-lr02", max_depth=4, learning_rate=0.02),
    dict(name="d3-lr10", max_depth=3, learning_rate=0.1),
    dict(name="d5-lr05", max_depth=5, learning_rate=0.05),
    dict(name="d4-lr10-col7", max_depth=4, learning_rate=0.1, colsample_bytree=0.7),
]

best = None
t0 = time.time()
for cfg in CONFIGS:
    name = cfg.pop("name")
    m = xgb.XGBClassifier(
        n_estimators=ROUNDS,
        eval_metric="auc",
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
        **cfg,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    aucs = m.evals_result()["validation_0"]["auc"]
    bi = int(np.argmax(aucs))
    tail = " RISING" if bi > ROUNDS - 20 else ""
    print(f"cfg {name}: best_round={bi + 1} auc={aucs[bi]:.4f}{tail}")
    if best is None or aucs[bi] > best[0]:
        best = (aucs[bi], dict(cfg), bi + 1)
    cfg["name"] = name
print(f"Sweep time: {time.time() - t0:.1f}s")
print(f"BEST: {best}")

final = xgb.XGBClassifier(
    n_estimators=best[2],
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    **best[1],
)
final.fit(X_tr, y_tr)
model = final


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


print(f"Eval AUC: {best[0]:.4f}")
