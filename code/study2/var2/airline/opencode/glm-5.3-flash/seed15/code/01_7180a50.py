"""XGBoost binary classifier with engineered features. Only file the agent edits.

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

FEATURE_COLS = ["Month", "DayOfMonth", "DayOfWeek", "DepTime", "UniqueCarrier",
                "Origin", "Dest", "Distance"]
train["_route"] = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique()))
              for c in ["UniqueCarrier", "Origin", "Dest", "_route"]}


def _parse_c(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.slice(2), errors="coerce")


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering applied identically everywhere (train/eval/holdout)."""
    X = pd.DataFrame(index=df.index)
    X["month"] = _parse_c(df["Month"])
    X["day"] = _parse_c(df["DayofMonth"])
    X["dow"] = _parse_c(df["DayOfWeek"])
    dt = df["DepTime"].astype(float) % 2400
    X["deptime"] = dt
    X["hour"] = (dt // 100).astype(int)
    X["minute"] = (dt % 100).astype(int)
    ang = 2 * np.pi * (dt / 1440.0)
    X["dt_sin"] = np.sin(ang)
    X["dt_cos"] = np.cos(ang)
    X["distance"] = df["Distance"].astype(float)
    X["logdist"] = np.log1p(X["distance"])
    X["carrier"] = df["UniqueCarrier"]
    X["origin"] = df["Origin"]
    X["dest"] = df["Dest"]
    X["route"] = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows."""
    X = base_features(df)
    X["carrier"] = pd.Categorical(X["carrier"], categories=CAT_LEVELS["UniqueCarrier"])
    X["origin"] = pd.Categorical(X["origin"], categories=CAT_LEVELS["Origin"])
    X["dest"] = pd.Categorical(X["dest"], categories=CAT_LEVELS["Dest"])
    X["route"] = pd.Categorical(X["route"], categories=CAT_LEVELS["_route"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.06,
    subsample=0.9,
    colsample_bytree=0.8,
    min_child_weight=5,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
Xtr = prepare(train)
ytr = to_y(train)
rng = np.random.RandomState(SEED)
va_mask = rng.rand(len(Xtr)) < 0.15
model.fit(Xtr[~va_mask], ytr[~va_mask],
          eval_set=[(Xtr[va_mask], ytr[va_mask])], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
