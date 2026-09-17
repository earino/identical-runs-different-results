"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- features -----------------------------------------------------------------
CAT_LEVELS = {
    "UniqueCarrier": pd.Index(sorted(train["UniqueCarrier"].unique())),
    "Origin": pd.Index(sorted(train["Origin"].unique())),
    "Dest": pd.Index(sorted(train["Dest"].unique())),
    "route": pd.Index(sorted((train["Origin"] + "_" + train["Dest"]).unique())),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["month"] = df["Month"].str.slice(2).astype(int)
    X["day"] = df["DayofMonth"].str.slice(2).astype(int)
    X["dow"] = df["DayOfWeek"].str.slice(2).astype(int)
    dep = df["DepTime"].astype(int)
    X["dep_time"] = dep
    X["hour"] = dep // 100
    X["minute"] = dep % 100
    X["distance"] = df["Distance"].astype(int)
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"], categories=CAT_LEVELS["UniqueCarrier"])
    X["Origin"] = pd.Categorical(df["Origin"], categories=CAT_LEVELS["Origin"])
    X["Dest"] = pd.Categorical(df["Dest"], categories=CAT_LEVELS["Dest"])
    X["route"] = pd.Categorical(df["Origin"] + "_" + df["Dest"], categories=CAT_LEVELS["route"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    prepare(train), to_y(train), test_size=0.1, random_state=SEED
)

model = xgb.XGBClassifier(
    n_estimators=1200,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s, best iter: {model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
