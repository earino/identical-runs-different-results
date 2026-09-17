"""Greedy backward elimination round 2 + DepTime-as-categorical variant.

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

feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET, "Month"]]
str_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
num_cols = [c for c in feature_cols if c not in str_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in str_cols}
DEPTIME_LEVELS = pd.Index(sorted(train["DepTime"].dropna().unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in num_cols:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    for c in str_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def with_dt_cat(df: pd.DataFrame) -> pd.DataFrame:
    X = prepare(df)
    X["DepTime"] = pd.Categorical(df["DepTime"], categories=DEPTIME_LEVELS)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


X_num = prepare(train)
Xe_num = prepare(evald)
X_cat = with_dt_cat(train)
Xe_cat = with_dt_cat(evald)
y = to_y(train)
ye = to_y(evald)

BASE6 = [c for c in X_num.columns if c != "DayofMonth"]
VARIANTS = [
    ("ctrl_6feat", BASE6, X_num, Xe_num),
    ("drop_Distance", [c for c in BASE6 if c != "Distance"], X_num, Xe_num),
    ("drop_DayOfWeek", [c for c in BASE6 if c != "DayOfWeek"], X_num, Xe_num),
    ("drop_UniqueCarrier", [c for c in BASE6 if c != "UniqueCarrier"], X_num, Xe_num),
    ("drop_Origin", [c for c in BASE6 if c != "Origin"], X_num, Xe_num),
    ("drop_Dest", [c for c in BASE6 if c != "Dest"], X_num, Xe_num),
    ("deptime_categorical", BASE6, X_cat, Xe_cat),
]

results = []
best_auc, best_model, best_name, best_feats, best_Xs = -1.0, None, None, None, None
for name, feats, Xtr, Xev in VARIANTS:
    t0 = time.time()
    m = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr[feats], y)
    auc = roc_auc_score(ye, m.predict_proba(Xev[feats])[:, 1])
    print(f"variant={name}  eval_auc={auc:.4f}  ({time.time() - t0:.1f}s)")
    if auc > best_auc:
        best_auc, best_model, best_name, best_feats, best_Xs = auc, m, name, feats, (Xtr, Xev)

print(f"BEST: {best_name} eval_auc={best_auc:.4f}")
model = best_model
FEATURES = best_feats
DT_AS_CAT = best_name == "deptime_categorical"


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = with_dt_cat(df) if DT_AS_CAT else prepare(df)
    return model.predict_proba(Xp[FEATURES])[:, 1]


print(f"Eval AUC: {best_auc:.4f}")
