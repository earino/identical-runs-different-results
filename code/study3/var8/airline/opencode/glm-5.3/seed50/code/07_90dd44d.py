"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Model: bag of 5 leaf-wise (lossguide) XGBoost models with column subsampling; three members are
trained on feature subsets (random-subspace decorrelation), two on the full set.
Features: scheduled departure minute-of-day (+sin/cos), Distance, carrier/origin/dest as native
categoricals (unseen levels -> NaN). Target encodings/route/date features were tested and HURT
generalization 2005->2006, so they are deliberately excluded.
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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()

# --- feature spec (fit on train only) --------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here: predict_proba() calls this on unseen rows."""
    X = pd.DataFrame(index=df.index)
    mins = (df["DepTime"].astype(int) // 100) * 60 + (df["DepTime"].astype(int) % 100)
    X["dep_minutes"] = mins
    X["sin_dep"] = np.sin(2 * np.pi * mins / 1440.0)
    X["cos_dep"] = np.cos(2 * np.pi * mins / 1440.0)
    X["dist"] = df["Distance"].astype(float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: subspace-diversified bag of leaf-wise XGBoost models -----------------
# (seed, columns to drop); dropping one group per member decorrelates bag errors.
SPECS = [
    (1, []),
    (2, []),
    (3, []),
    (4, ["UniqueCarrier"]),
    (5, ["Dest"]),
    (6, ["dist"]),
]


def make_model(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=450,
        max_depth=0,
        grow_policy="lossguide",
        max_leaves=512,
        learning_rate=0.05,
        min_child_weight=1,
        subsample=1.0,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.5,
        tree_method="hist",
        max_bin=256,
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


t0 = time.time()
X_train = prepare(train)
models = []
for seed, drop in SPECS:
    m = make_model(seed)
    m.fit(X_train.drop(columns=drop) if drop else X_train, y_train)
    models.append((m, drop))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X.drop(columns=drop) if drop else X)[:, 1] for m, drop in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
