"""XGBoost binary classifier — airline dep_delayed_15min. THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives inside prepare(); encoders are fit on training data only.
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

# --- feature engineering (encoders fit on training data only) ----------------------
FEATS_STR = ["DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in FEATS_STR + ["Month"]}
HHCAR_CATS = pd.Index(sorted(((train["DepTime"] // 100).astype(str) + "|" + train["UniqueCarrier"].astype(str)).unique()))
HHORG_CATS = pd.Index(sorted(((train["DepTime"] // 100).astype(str) + "|" + train["Origin"].astype(str)).unique()))


def prepare(df: pd.DataFrame, with_month: bool = False) -> pd.DataFrame:
    """ALL feature engineering must be reachable from here (predict_proba path)."""
    X = pd.DataFrame(index=df.index)

    hh = df["DepTime"] // 100
    mm = df["DepTime"] % 100
    dep_norm = ((hh * 60 + mm) % 1440).astype(float)

    X["DepTime"] = df["DepTime"]
    X["dep_norm"] = dep_norm
    X["dep_sin"] = np.sin(2 * np.pi * dep_norm / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * dep_norm / 1440.0)
    X["dep_late_night"] = ((hh >= 21) | (hh <= 5)).astype(int)
    X["dep_hh"] = hh
    X["Distance"] = df["Distance"]

    hh_s = hh.astype(str)
    keys = hh_s + "|" + df["UniqueCarrier"].astype(str)
    X["hhxcar"] = pd.Categorical(keys, categories=HHCAR_CATS)
    keys = hh_s + "|" + df["Origin"].astype(str)
    X["hhxorg"] = pd.Categorical(keys, categories=HHORG_CATS)

    for c in FEATS_STR + (["Month"] if with_month else []):
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- recency weights: flight month within 2005 (covariate shift toward 2006) -------
_rec = 1.0 + 5.0 * (train["Month"].str.slice(2).astype(int) / 12.0)
sample_w = _rec.to_numpy()

y = to_y(train)
BP = dict(max_depth=8, learning_rate=0.1, alpha=5, tree_method="hist",
          enable_categorical=True, n_jobs=N_JOBS)

t0 = time.time()
base_models = [
    xgb.XGBClassifier(n_estimators=400, colsample_bytree=0.85, random_state=SEED + 100 + k, **BP)
    for k in range(6)
]
for m in base_models:
    m.fit(prepare(train), y, sample_weight=sample_w)
month_models = [  # unweighted: their 2005 month pattern is diversity, not signal
    xgb.XGBClassifier(n_estimators=500, random_state=SEED + 200 + k, **BP)
    for k in range(4)
]
for m in month_models:
    m.fit(prepare(train, with_month=True), y)
print(f"Training time: {time.time() - t0:.1f}s")

W_BASE, W_MONTH = 1.0, 0.7  # month models add diversity; their month pattern only partially transfers


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    p = np.zeros(len(df))
    for m in base_models:
        p += W_BASE / len(base_models) * m.predict_proba(prepare(df))[:, 1]
    for m in month_models:
        p += W_MONTH / len(month_models) * m.predict_proba(prepare(df, with_month=True))[:, 1]
    return p


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
