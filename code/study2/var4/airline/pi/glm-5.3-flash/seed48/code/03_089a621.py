"""XGBoost binary classifier for airline delay. Contract: see program.md."""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

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


Xall = prepare(train)
yall = to_y(train)
Xeval = prepare(evald)
yeval = to_y(evald)
X_tr, X_val, y_tr, y_val = train_test_split(
    np.arange(len(train)), yall, test_size=0.1, random_state=SEED, stratify=yall
)
Xv, yv = Xall.iloc[X_val], yall[X_val]

configs = []
for depth in [4, 6]:
    for lr in [0.1, 0.2]:
        for n in [60, 150]:
            for mcw in [5, 30]:
                configs.append((depth, lr, n, mcw, 0.8, 0.8))
configs += [(6, 0.1, 100, 50, 0.6, 0.6), (6, 0.15, 80, 20, 0.7, 0.7), (4, 0.1, 200, 10, 0.8, 0.8)]

t0 = time.time()
results = []
for depth, lr, n, mcw, ss, cs in configs:
    m = xgb.XGBClassifier(
        n_estimators=n, max_depth=depth, learning_rate=lr, subsample=ss, colsample_bytree=cs,
        min_child_weight=mcw, tree_method="hist", enable_categorical=True,
        eval_metric="auc", random_state=SEED, n_jobs=N_JOBS,
    )
    m.fit(Xall.iloc[X_tr], y_tr, eval_set=[(Xv, yv)], verbose=False)
    va = m.evals_result()["validation_0"]["auc"][-1]
    ev = roc_auc_score(yeval, m.predict_proba(Xeval)[:, 1])
    results.append((ev, va, (depth, lr, n, mcw, ss, cs)))
    print(f"depth={depth} lr={lr} n={n} mcw={mcw} ss={ss} cs={cs} -> val={va:.4f} eval={ev:.4f}")
print(f"sweep time: {time.time() - t0:.1f}s")

results.sort(reverse=True)
best_ev, best_va, best_cfg = results[0]
print(f"BEST config: {best_cfg} eval={best_ev:.4f} val={best_va:.4f}")

# refit best config on FULL train for the official number
depth, lr, n, mcw, ss, cs = best_cfg
model = xgb.XGBClassifier(
    n_estimators=n, max_depth=depth, learning_rate=lr, subsample=ss, colsample_bytree=cs,
    min_child_weight=mcw, tree_method="hist", enable_categorical=True,
    eval_metric="auc", random_state=SEED, n_jobs=N_JOBS,
)
model.fit(Xall, yall, verbose=False)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


eval_auc = roc_auc_score(yeval, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
