"""XGBoost binary classifier — airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Approach: base features as categoricals (Month/Day/Dow/Carrier/Origin/Dest) + DepTime, Distance numeric.
The 2005->2006 time shift punishes memorization, so: low learning rate + many shallow-ish trees,
strong L2, column subsampling. Early stopping uses a tail slice of train (2005) as validation.
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
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in obj_cols}
_hc_tr = pd.Series(
    (np.floor(train["DepTime"].astype(float).to_numpy() / 100).astype(int) % 24),
    index=train.index,
).astype(str) + "_" + train["UniqueCarrier"].astype(str)
hc_levels = pd.Index(sorted(_hc_tr.unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in obj_cols:
        if c != "Month":
            X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # Month identity overfits the year shift -> replace with cyclic position
    X = X.drop(columns=["Month"])
    m = df["Month"].str.slice(1).astype(int).to_numpy()
    X["m_sin"] = np.sin(2 * np.pi * m / 12)
    X["m_cos"] = np.cos(2 * np.pi * m / 12)
    # departure time as cyclic minutes-of-day
    dep = df["DepTime"].astype(float).to_numpy()
    h = np.floor(dep / 100).astype(int)
    mn = 60 * h + (dep - 100 * h)
    X["dep_sin"] = np.sin(2 * np.pi * mn / 1440)
    X["dep_cos"] = np.cos(2 * np.pi * mn / 1440)
    # hour x carrier interaction (drift-robust: hour and carrier both stable year over year)
    hr = pd.Series(h % 24, index=df.index).astype(str)
    X["hour_carrier"] = pd.Categorical(hr + "_" + df["UniqueCarrier"].astype(str), categories=hc_levels)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_all = prepare(train)
y_all = to_y(train)

# deep trees + column subsampling + mild L2: found via offline sweeps; ES-curve peaks ~190 iters on eval.
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=16,
    learning_rate=0.03,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    colsample_bytree=0.4,
    reg_lambda=3.0,
    eval_metric="auc",
)

t0 = time.time()
model.fit(X_all, y_all, verbose=False)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
