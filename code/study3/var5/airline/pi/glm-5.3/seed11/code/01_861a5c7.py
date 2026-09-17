"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, module-level `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives inside `prepare(df)`; fitted statistics come from train.csv only.
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

# --- feature engineering ------------------------------------------------------
NUM_CAT_COLS = ["Month", "DayofMonth", "DayOfWeek"]  # 'c-<n>' strings -> numeric


def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.split("-").str[-1], errors="coerce")


RAW_CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]

_cat_source = train.copy()
_cat_source["Route"] = _cat_source["Origin"] + "_" + _cat_source["Dest"]
cat_levels = {
    "UniqueCarrier": pd.Index(sorted(_cat_source["UniqueCarrier"].dropna().unique())),
    "Origin": pd.Index(sorted(_cat_source["Origin"].dropna().unique())),
    "Dest": pd.Index(sorted(_cat_source["Dest"].dropna().unique())),
    "Route": pd.Index(sorted(_cat_source["Route"].dropna().unique())),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in NUM_CAT_COLS:
        X[c] = _cnum(df[c])
    X["DepTime"] = pd.to_numeric(df["DepTime"], errors="coerce")
    X["hour"] = X["DepTime"] // 100
    X["minute"] = X["DepTime"] % 100
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["log_distance"] = np.log1p(X["Distance"])
    for c in RAW_CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["Route"] = pd.Categorical(df["Origin"] + "_" + df["Dest"], categories=cat_levels["Route"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=30,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model = xgb.XGBClassifier(**PARAMS)
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
