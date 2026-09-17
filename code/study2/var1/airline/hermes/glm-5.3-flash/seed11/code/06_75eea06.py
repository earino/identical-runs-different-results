"""XGBoost airline delay classifier (agent-edited). See program.md for the contract.

predict_proba(df) -> P(dep_delayed_15min == 'Y'). All feature engineering lives in prepare(df)
so the hidden holdout gets identical treatment. Encoders/statistics are fitted on train only.

This run: internal 4-fold CV grid over a few model configs (inside the 120s budget),
winner refit on the full 100k train.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

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
# frozen category levels, fitted on TRAIN only (unseen levels -> NaN in predict_proba)
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
GRID = [
    dict(n_estimators=30, learning_rate=0.1, max_depth=6, subsample=1.0),
    dict(n_estimators=60, learning_rate=0.05, max_depth=6, subsample=1.0),
    dict(n_estimators=100, learning_rate=0.05, max_depth=6, subsample=1.0),
    dict(n_estimators=100, learning_rate=0.1, max_depth=6, subsample=1.0),
    dict(n_estimators=30, learning_rate=0.1, max_depth=8, subsample=1.0),
    dict(n_estimators=60, learning_rate=0.05, max_depth=8, subsample=1.0),
    dict(n_estimators=100, learning_rate=0.05, max_depth=6, subsample=0.9),
    dict(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=1.0),
]


def make_model(**kw):
    params = dict(
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    params.update(kw)
    return xgb.XGBClassifier(**params)


X_full = prepare(train)
y_full = to_y(train)

t0 = time.time()
kf = KFold(n_splits=4, shuffle=True, random_state=SEED)
splits = list(kf.split(X_full))
results = []
for i, cfg in enumerate(GRID):
    aucs = []
    for tr_idx, va_idx in splits:
        m = make_model(**cfg)
        m.fit(X_full.iloc[tr_idx], y_full[tr_idx], verbose=False)
        aucs.append(roc_auc_score(y_full[va_idx], m.predict_proba(X_full.iloc[va_idx])[:, 1]))
    results.append((float(np.mean(aucs)), i, cfg))
    print(f"cfg{i} {cfg} cv_auc={np.mean(aucs):.4f} (+-{np.std(aucs):.4f})")
print(f"CV time: {time.time() - t0:.1f}s")

results.sort(reverse=True)
best_cv, best_i, best_cfg = results[0]
print(f"BEST cfg{best_i} cv={best_cv:.4f} {best_cfg}")

t0 = time.time()
model = make_model(**best_cfg)
model.fit(X_full, y_full, verbose=False)
print(f"Final fit time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
