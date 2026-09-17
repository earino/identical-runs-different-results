"""XGBoost binary classifier for airline delays. THE ONLY FILE THE AGENT EDITS.

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
from sklearn.model_selection import train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


y_train = to_y(train)
PRIOR = float(y_train.mean())
SMOOTH = 30.0


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row derived features. No fitted statistics in here."""
    out = pd.DataFrame(index=df.index)
    month = _num(df["Month"])
    dom = _num(df["DayofMonth"])
    dow = _num(df["DayOfWeek"])
    deptime = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (deptime // 100)
    dep_min = hour * 60 + (deptime % 100)
    out["month"] = month
    out["dom"] = dom
    out["dow"] = dow
    out["deptime"] = deptime
    out["hour"] = hour
    out["dep_sin"] = np.sin(2 * np.pi * dep_min / 1440.0)
    out["dep_cos"] = np.cos(2 * np.pi * dep_min / 1440.0)
    out["dom_sin"] = np.sin(2 * np.pi * (dom - 1) / 31.0)
    out["dom_cos"] = np.cos(2 * np.pi * (dom - 1) / 31.0)
    out["dow_sin"] = np.sin(2 * np.pi * (dow - 1) / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * (dow - 1) / 7.0)
    out["distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    out["log_distance"] = np.log1p(out["distance"])
    out["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    out["Origin"] = df["Origin"].astype(str)
    out["Dest"] = df["Dest"].astype(str)
    out["route"] = out["Origin"] + "_" + out["Dest"]
    return out


feat_train = base_features(train)
_tmp = feat_train.assign(_y=y_train)

CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "route"]
TE_KEYS = ["UniqueCarrier", "Origin", "Dest", "route", "hour", "month", "dow", "dom"]

cat_levels = {c: pd.Index(sorted(feat_train[c].unique())) for c in CAT_COLS}

te_maps = {}
for key in TE_KEYS:
    g = _tmp.groupby(key)["_y"].agg(["size", "sum"])
    te_maps[key] = ((g["sum"] + SMOOTH * PRIOR) / (g["size"] + SMOOTH)).to_dict()

BASE_NUMERIC = [c for c in feat_train.columns if c not in CAT_COLS]
NUMERIC = BASE_NUMERIC + ["te_" + k for k in TE_KEYS]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    f = base_features(df)
    X = pd.DataFrame(index=df.index)
    for c in BASE_NUMERIC:
        X[c] = pd.to_numeric(f[c], errors="coerce")
    for key in TE_KEYS:
        X["te_" + key] = f[key].map(te_maps[key]).astype(float).fillna(PRIOR)
    for c in CAT_COLS:
        X[c] = pd.Categorical(f[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    prepare(train), y_train, test_size=0.15, random_state=SEED, stratify=y_train
)

model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=40,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
print(f"trees={model.best_iteration} features={X_tr.shape[1]}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
