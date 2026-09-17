"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

CAT_COLS = ["Month", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
_ch = (train["UniqueCarrier"].astype(str) + "_" + (train["DepTime"].astype("float64") // 100).astype(int).astype(str))
carrier_hour_levels = pd.Index(sorted(set(_ch)))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    dt = df["DepTime"].astype("float64")
    dep_min = (dt // 100) * 60 + (dt % 100)
    X["dep_min"] = dep_min
    X["dep_hour"] = dep_min // 60
    X["dep_sin"] = np.sin(2 * np.pi * dep_min / 1440)
    X["dep_cos"] = np.cos(2 * np.pi * dep_min / 1440)
    X["Distance"] = df["Distance"].astype("float64")
    X["carrier_hour"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "_" + (df["DepTime"].astype("float64") // 100).astype(int).astype(str),
        categories=carrier_hour_levels,
    )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


X_all, y_all = prepare(train), to_y(train)

# --- model --------------------------------------------------------------------
def make_model(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        tree_method="hist",
        max_bin=512,
        enable_categorical=True,
        max_cat_threshold=32,
        random_state=seed,
        n_jobs=N_JOBS,
        eval_metric="auc",
        early_stopping_rounds=50,
    )


t0 = time.time()
models = []
kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_all)):
    m = make_model(SEED + fold)
    m.fit(X_all.iloc[tr_idx], y_all[tr_idx], eval_set=[(X_all.iloc[va_idx], y_all[va_idx])], verbose=False)
    m.n_estimators = m.best_iteration + 1
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  best_iters: {[m.best_iteration for m in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    preds = [m.predict_proba(prepare(df))[:, 1] for m in models]
    return np.mean(preds, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
