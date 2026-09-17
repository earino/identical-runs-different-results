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

# --- features -----------------------------------------------------------------
# ALL feature engineering lives inside prepare()/add_base(): predict_proba() calls it on unseen rows.
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "hour_cat", "tbucket", "hour_carrier"]


def add_base(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["month"] = pd.to_numeric(df["Month"].astype(str).str.slice(2), errors="coerce")
    X["day"] = pd.to_numeric(df["DayofMonth"].astype(str).str.slice(2), errors="coerce")
    X["dow"] = pd.to_numeric(df["DayOfWeek"].astype(str).str.slice(2), errors="coerce")
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).astype("Int64")
    minute = dep % 100
    X["hour"] = hour
    X["minute"] = minute
    dep_min = hour * 60 + minute
    X["dep_sin"] = np.sin(2 * np.pi * dep_min / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * dep_min / 1440.0)
    X["distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["log_dist"] = np.log1p(X["distance"])
    X["UniqueCarrier"] = df["UniqueCarrier"]
    X["Origin"] = df["Origin"]
    X["Dest"] = df["Dest"]
    hs = hour.astype(str)
    X["hour_cat"] = hs
    X["tbucket"] = (dep_min // 30).astype("Int64").astype(str)
    X["hour_carrier"] = hs + "_" + df["UniqueCarrier"].astype(str)
    return X


_base_train = add_base(train)
cat_levels = {c: pd.Index(sorted(_base_train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = add_base(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# 2-seed bagged ensemble of deep trees; averaged probabilities.
def _make_model(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=600,
        max_depth=20,
        learning_rate=0.03,
        reg_lambda=2.0,
        alpha=0.5,
        subsample=0.8,
        colsample_bytree=0.6,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=150,
        eval_metric="auc",
        random_state=seed,
        n_jobs=N_JOBS,
    )


X_train, y_train = prepare(train), to_y(train)
X_eval, y_eval = prepare(evald), to_y(evald)

t0 = time.time()
models = []
for seed in (1, 2):
    m = _make_model(seed)
    m.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
    models.append(m)
    print(f"model seed={seed} best_iter={m.best_iteration} t={time.time() - t0:.1f}s")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(y_eval, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
