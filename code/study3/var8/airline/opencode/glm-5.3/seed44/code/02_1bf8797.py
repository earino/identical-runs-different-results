"""XGBoost airline delay: feature engineering v1 (time/cyclic/route) + early stopping.

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

# --- feature engineering -------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Route"]
_train_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
cat_levels = {
    "UniqueCarrier": sorted(train["UniqueCarrier"].dropna().astype(str).unique()),
    "Origin": sorted(train["Origin"].dropna().astype(str).unique()),
    "Dest": sorted(train["Dest"].dropna().astype(str).unique()),
    "Route": sorted(_train_route.unique()),
}


def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    month = _cnum(df["Month"]).astype(float)
    dom = _cnum(df["DayofMonth"]).astype(float)
    dow = _cnum(df["DayOfWeek"]).astype(float)
    X["month"] = month
    X["dom"] = dom
    X["dow"] = dow
    tp = 2.0 * np.pi
    X["month_sin"] = np.sin(tp * month / 12.0)
    X["month_cos"] = np.cos(tp * month / 12.0)
    X["dom_sin"] = np.sin(tp * dom / 31.0)
    X["dom_cos"] = np.cos(tp * dom / 31.0)
    X["dow_sin"] = np.sin(tp * dow / 7.0)
    X["dow_cos"] = np.cos(tp * dow / 7.0)
    dep = df["DepTime"].astype(int)
    dep_min = ((dep // 100) * 60 + (dep % 100)) % 1440
    X["dep_min"] = dep_min.astype(float)
    X["hour"] = (dep_min // 60).astype(float)
    X["dep_sin"] = np.sin(tp * dep_min / 1440.0)
    X["dep_cos"] = np.cos(tp * dep_min / 1440.0)
    X["dist"] = df["Distance"].astype(float)
    X["dist_log"] = np.log1p(X["dist"])
    for c in ["UniqueCarrier", "Origin", "Dest"]:
        X[c] = df[c].astype(str)
    X["Route"] = X["Origin"] + "_" + X["Dest"]
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model ---------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=4000,
    max_depth=8,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=200,
    eval_metric="auc",
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
