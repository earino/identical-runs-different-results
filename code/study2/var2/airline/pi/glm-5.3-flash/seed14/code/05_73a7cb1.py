"""Airline delay XGBoost. Exp 7: depth8 mcw20 with carrier_hour."""
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

# --- fitted statistics (train only) -------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
CH_TRAIN = train["UniqueCarrier"] + "_" + ((train["DepTime"].astype(float) // 100).mod(24)).astype(int).astype(str)
CH_LEVELS = pd.Index(sorted(CH_TRAIN.unique()))


def prep_base(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering lives here (called by predict_proba too)."""
    X = pd.DataFrame(index=df.index)
    X["month"] = df["Month"].str.replace("c-", "", regex=False).astype(int)
    X["dom"] = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int)
    X["dow"] = df["DayOfWeek"].str.replace("c-", "", regex=False).astype(int)
    deptime = df["DepTime"].astype(float)
    hour = (deptime // 100).mod(24)
    minute = deptime.mod(100)
    tod = hour * 60 + minute
    ang = 2 * np.pi * tod / 1440.0
    X["deptime"] = deptime
    X["dep_hour"] = hour
    X["dep_minute"] = minute
    X["tod_sin"] = np.sin(ang)
    X["tod_cos"] = np.cos(ang)
    X["distance"] = df["Distance"].astype(float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    ch = df["UniqueCarrier"] + "_" + ((df["DepTime"].astype(float) // 100).mod(24)).astype(int).astype(str)
    X["carrier_hour"] = pd.Categorical(ch, categories=CH_LEVELS)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=20,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prep_base(train), to_y(train), eval_set=[(prep_base(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prep_base(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
