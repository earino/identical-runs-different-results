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
    out["dep_slot"] = dtm // 15   # 15-min slot index, numeric (0..95)
    minutes = (dtm // 100) * 60 + (dtm % 100)
    ang = 2 * np.pi * minutes / 1440.0
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
    max_bin=512,
)
# diverse ensemble members: fixed tree counts, varied depth/regularization/seed
MEMBERS = [
    dict(max_depth=10, learning_rate=0.05, n_estimators=800, min_child_weight=20,
         subsample=0.7, colsample_bytree=0.7, reg_lambda=5.0, random_state=7),
    dict(max_depth=12, learning_rate=0.05, n_estimators=700, min_child_weight=10,
         subsample=0.7, colsample_bytree=0.7, reg_lambda=3.0, random_state=777),
    dict(max_depth=12, learning_rate=0.05, n_estimators=700, min_child_weight=10,
         subsample=0.7, colsample_bytree=0.7, reg_lambda=3.0, random_state=888),
    dict(max_depth=8, learning_rate=0.05, n_estimators=1000, min_child_weight=10,
         subsample=0.5, colsample_bytree=0.5, reg_lambda=2.0, random_state=99),
    dict(max_depth=12, learning_rate=0.05, n_estimators=600, min_child_weight=30,
         subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0, random_state=123),
    dict(max_depth=9, learning_rate=0.03, n_estimators=1300, min_child_weight=15,
         subsample=0.7, colsample_bytree=0.7, reg_lambda=4.0, random_state=202),
    (dict(max_depth=12, learning_rate=0.05, n_estimators=700, min_child_weight=10,
          subsample=0.7, colsample_bytree=0.7, reg_lambda=3.0, random_state=4242), 7),
]

t0 = time.time()
X = prepare(train)
y = to_y(train)
Xe = prepare(evald)
# recency weights: eval is 2006, so later-2005 months are more representative
_m = train["Month"].str.slice(2).astype(int).to_numpy()
w = 1.0 + 0.08 * (_m - 1)   # Jan=1.0 ... Dec=1.88
models = []
for i, mp in enumerate(MEMBERS):
    ts = time.time()
    params, month_min = mp if isinstance(mp, tuple) else (mp, 1)
    m = xgb.XGBClassifier(**{**BASE, **params})
    mask = _m >= month_min
    m.fit(X.loc[mask], y[mask], sample_weight=w[mask])
    models.append(m)
    print(f"model {i}: {time.time() - ts:.1f}s months>={month_min}")
print(f"total fit: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    p = np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)
    return p


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
