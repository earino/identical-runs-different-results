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
N_MODELS = 9

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Month", "DayofMonth", "DayOfWeek"]
# interaction categoricals: levels fitted on the training data only
_hour = lambda d: ((d["DepTime"].astype(int) // 100) % 24).astype(str)          # noqa: E731
INTERACTIONS = {
    "Hour": _hour,
    "HourCarrier": lambda d: _hour(d) + "|" + d["UniqueCarrier"].astype(str),
    "CarrierDow": lambda d: d["UniqueCarrier"].astype(str) + "|" + d["DayOfWeek"].astype(str),
    "HourOrigin": lambda d: _hour(d) + "|" + d["Origin"].astype(str),
    "DepBin10": lambda d: (((d["DepTime"].astype(int) // 100) % 24 * 60 + d["DepTime"].astype(int) % 100) // 10).astype(str),
}
interaction_levels = {
    name: pd.Index(sorted(set(fn(train)))) for name, fn in INTERACTIONS.items()
}
feature_cols = CAT_COLS + list(INTERACTIONS) + ["DepTime", "DepMinutes", "Distance", "RouteN"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
cat_levels.update(interaction_levels)
# route frequency, fitted on training data only
route_counts = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    dep = df["DepTime"].astype(int)
    for name, fn in INTERACTIONS.items():
        X[name] = pd.Categorical(fn(df), categories=cat_levels[name])
    X["DepTime"] = dep
    X["DepMinutes"] = (dep // 100) % 24 * 60 + dep % 100
    X["Distance"] = df["Distance"].astype(float)
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["RouteN"] = route.map(route_counts).fillna(0).to_numpy(dtype=float)
    return X[feature_cols]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: seed ensemble of XGB classifiers with column diversity -------------
PARAMS = dict(
    n_estimators=600,
    max_depth=8,
    learning_rate=0.05,
    min_child_weight=10,
    colsample_bytree=0.7,
    reg_alpha=4.0,
    grow_policy="lossguide",
    max_leaves=128,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_all, y_all = prepare(train), to_y(train)
models = []
for s in range(N_MODELS):
    m = xgb.XGBClassifier(random_state=s, **PARAMS)
    m.fit(X_all, y_all, verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = sum(m.predict_proba(X)[:, 1] for m in models)
    return ps / len(models)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
