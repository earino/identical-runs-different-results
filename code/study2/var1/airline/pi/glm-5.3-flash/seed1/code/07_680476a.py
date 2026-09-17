"""XGBoost binary classifier on the airline dataset. THE ONLY FILE THE AGENT EDITS.

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

# --- encoder statistics fit on TRAINING data only (module level) ---------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Route"]
train["Route"] = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw schema -> feature frame. All engineering lives here so predict_proba reproduces it."""
    out = pd.DataFrame(index=df.index)
    # calendar features (c-<n> strings -> ints)
    out["month"] = df["Month"].str.slice(2).astype(int)
    out["day"] = df["DayofMonth"].str.slice(2).astype(int)
    out["dow"] = df["DayOfWeek"].str.slice(2).astype(int)
    out["weekend"] = (out["dow"] >= 6).astype(int)
    # scheduled departure time -> minutes since midnight (wrap bogus >=2400 values)
    dtm = df["DepTime"].astype(int) % 2400
    out["deptime"] = dtm
    out["dep_hour"] = dtm // 100
    out["dep_min"] = dtm % 100
    ang = 2 * np.pi * (out["dep_hour"] * 60 + out["dep_min"]) / 1440.0
    out["dep_sin"] = np.sin(ang)
    out["dep_cos"] = np.cos(ang)
    # distance
    d = df["Distance"].astype(float)
    out["distance"] = d
    out["distance_log"] = np.log1p(d)
    # categorical features (native categoricals; levels frozen from train)
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    out["UniqueCarrier"] = df["UniqueCarrier"]
    out["Origin"] = df["Origin"]
    out["Dest"] = df["Dest"]
    out["Route"] = route
    for c in CAT_COLS:
        out[c] = pd.Categorical(out[c], categories=cat_levels[c])
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    return add_features(df)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
BASE = dict(
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
    eval_metric="auc",
)
# diverse ensemble members: fixed tree counts, varied depth/regularization/seed
MEMBERS = [
    dict(max_depth=8, learning_rate=0.05, n_estimators=1200, min_child_weight=10,
         subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, random_state=SEED),
    dict(max_depth=10, learning_rate=0.05, n_estimators=800, min_child_weight=20,
         subsample=0.7, colsample_bytree=0.7, reg_lambda=5.0, random_state=7),
    dict(max_depth=12, learning_rate=0.05, n_estimators=700, min_child_weight=10,
         subsample=0.7, colsample_bytree=0.7, reg_lambda=3.0, random_state=777),
    dict(max_depth=8, learning_rate=0.05, n_estimators=1200, min_child_weight=10,
         subsample=0.5, colsample_bytree=0.5, reg_lambda=2.0, random_state=99),
    dict(max_depth=12, learning_rate=0.05, n_estimators=600, min_child_weight=30,
         subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0, random_state=123),
    dict(max_depth=9, learning_rate=0.03, n_estimators=1600, min_child_weight=15,
         subsample=0.7, colsample_bytree=0.7, reg_lambda=4.0, random_state=202),
]

t0 = time.time()
X = prepare(train)
y = to_y(train)
Xe = prepare(evald)
models = []
for i, mp in enumerate(MEMBERS):
    ts = time.time()
    m = xgb.XGBClassifier(**{**BASE, **mp})
    m.fit(X, y)
    models.append(m)
    solo = roc_auc_score(to_y(evald), m.predict_proba(Xe)[:, 1])
    print(f"model {i}: {time.time() - ts:.1f}s solo_eval_auc={solo:.5f}")
print(f"total fit: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    p = np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)
    return p


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
