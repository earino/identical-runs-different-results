"""XGBoost binary classifier for airline departure-delay prediction.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Design notes (from offline exploration):
  - 2005->2006 drift: internal 2005-val early stopping is useless (val rewards year-specific overfit),
    so we train on 100% of train.csv with a FIXED, heavily-decorrelated configuration:
    deep trees (d14) + low colsample_bytree(0.4) + 5-seed ensemble averaging.
  - Label-free count/volume features (route, origin+hour, carrier+hour, route+hour traffic) transfer
    across years; label-based target encodings do NOT (they help 2005, hurt 2006).
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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# count/volume stats: computed from TRAIN (2005) rows only, at module level, used inside prepare()
def _key_fns(df):
    h = (df["DepTime"] // 100).astype(str)
    return {
        "cnt_origin": df["Origin"].astype(str),
        "cnt_route": df["Origin"].astype(str) + "-" + df["Dest"].astype(str),
        "cnt_origh": df["Origin"].astype(str) + "-" + h,
        "cnt_carrh": df["UniqueCarrier"].astype(str) + "-" + h,
        "cnt_routeh": df["Origin"].astype(str) + "-" + df["Dest"].astype(str) + "-" + h,
    }

CNT_MAPS = {nm: vc for nm, vc in _key_fns(train).items()}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering lives here: predict_proba() calls prepare() on unseen rows."""
    X = pd.DataFrame(index=df.index)
    X["month"] = df["Month"].str.replace("c-", "").astype(int)
    X["dom"] = df["DayofMonth"].str.replace("c-", "").astype(int)
    X["dow"] = df["DayOfWeek"].str.replace("c-", "").astype(int)
    X["dep"] = df["DepTime"].astype(int)
    X["hour"] = df["DepTime"] // 100
    X["min"] = df["DepTime"] % 100
    X["dep_min"] = X["hour"] * 60 + X["min"]
    X["sin"] = np.sin(2 * np.pi * X["dep_min"] / 1440).round(6)
    X["cos"] = np.cos(2 * np.pi * X["dep_min"] / 1440).round(6)
    X["dist"] = df["Distance"].astype(int)
    X["logdist"] = np.log1p(df["Distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    for nm, keys in _key_fns(df).items():
        X[nm] = keys.map(CNT_MAPS[nm]).fillna(0).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: 5-seed ensemble of deep, column-decorrelated trees ----------------
N_EST = 250
model_kwargs = dict(
    n_estimators=N_EST,
    max_depth=14,
    learning_rate=0.05,
    min_child_weight=10,
    subsample=0.7,
    colsample_bytree=0.4,
    reg_lambda=5,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
    eval_metric="auc",
)

X_train, y_train = prepare(train), to_y(train)
t0 = time.time()
models = []
for seed in (1, 2, 3, 4, 5):
    m = xgb.XGBClassifier(random_state=seed, **model_kwargs)
    m.fit(X_train, y_train, verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
