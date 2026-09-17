"""XGBoost binary classifier — airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- fitted on TRAIN only -------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
route_train = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
cat_levels["route"] = pd.Index(sorted(route_train.unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    X["dep_time"] = dep
    X["hour"] = np.floor(dep / 100.0)
    X["minute"] = dep % 100.0
    X["month"] = df["Month"].astype(str).str.replace("c-", "", regex=False).astype(float)
    X["day"] = df["DayofMonth"].astype(str).str.replace("c-", "", regex=False).astype(float)
    X["dow"] = df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False).astype(float)
    X["distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["log_distance"] = np.log1p(X["distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    X["route"] = pd.Categorical(
        df["Origin"].astype(str) + "_" + df["Dest"].astype(str), categories=cat_levels["route"]
    )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
Xall = prepare(train)
yall = to_y(train)
X_tr, X_val, y_tr, y_val = train_test_split(Xall, yall, test_size=0.1, random_state=SEED, stratify=yall)

model = xgb.XGBClassifier(
    n_estimators=800,
    max_depth=8,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=40,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
