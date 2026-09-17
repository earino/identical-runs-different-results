"""XGBoost airline delay classifier (agent-edited). See program.md for the contract.

predict_proba(df) -> P(dep_delayed_15min == 'Y'). All feature engineering lives in prepare(df)
so the hidden holdout gets identical treatment. Encoders/statistics are fitted on train only.

This run: baseline features; ensemble of 5 XGBoost models (seed bag + subsample decorrelation),
mean of predicted probabilities. Final model = XGBoost ensemble (allowed).
"""
import warnings

warnings.filterwarnings("ignore")

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
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# heterogeneous bag: (n_estimators, lr, depth) x seeds, subsample/colsample decorrelation
MEMBERS = (
    [(30, 0.1, 6)] * 3 + [(60, 0.05, 6)] * 3 + [(30, 0.1, 8)] * 3 + [(60, 0.05, 8)] * 3
)
CONFIG = dict(subsample=0.9, colsample_bytree=0.85)

X = prepare(train)
y = to_y(train)

t0 = time.time()
models = []
for k, (ne, lr, md) in enumerate(MEMBERS):
    m = xgb.XGBClassifier(
        tree_method="hist", enable_categorical=True, random_state=SEED + k, n_jobs=N_JOBS,
        n_estimators=ne, learning_rate=lr, max_depth=md, **CONFIG,
    )
    m.fit(X, y, verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    probs = np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)
    return probs


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
