"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. Module-level `predict_proba(df)`: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives inside prepare(); statistics fit on train.csv only.

Transfer notes (2005 -> 2006 shift): raw high-cardinality interactions (Route identity,
target encodings, dow interactions) overfit 2005 and hurt 2006 generalization.
What transfers: time-of-day buckets (15-min cat + hour cat), hour x carrier interaction,
identity Origin/Dest/carrier, strong L2 regularization.
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

# --- statistics fit on the training data only ----------------------------------
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in ["UniqueCarrier", "Origin", "Dest"]}
_tr_hour = (train["DepTime"].astype(int) // 100) % 24
_tr_carr = train["UniqueCarrier"].astype(str)
HOUR_CARR_CATS = pd.Index(sorted((_tr_hour.astype(str) + "_" + _tr_carr).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(int)
    hour = (dep // 100) % 24
    minute = dep % 100
    mins = hour * 60 + minute
    X["DepTime"] = dep
    X["Distance"] = df["Distance"].astype(float)
    X["logDist"] = np.log1p(X["Distance"])
    X["Month"] = df["Month"].astype(str).str.replace("c-", "", regex=False).astype(int)
    X["DayofMonth"] = df["DayofMonth"].astype(str).str.replace("c-", "", regex=False).astype(int)
    X["DayOfWeek"] = df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False).astype(int)
    X["DepHour"] = hour
    X["DepMinute"] = minute
    X["MinsOfDay"] = mins
    X["sin_tod"] = np.sin(2 * np.pi * mins / 1440.0)
    X["cos_tod"] = np.cos(2 * np.pi * mins / 1440.0)
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"], categories=cat_levels["UniqueCarrier"])
    X["Origin"] = pd.Categorical(df["Origin"], categories=cat_levels["Origin"])
    X["Dest"] = pd.Categorical(df["Dest"], categories=cat_levels["Dest"])
    X["HourCat"] = pd.Categorical(hour, categories=pd.Index(range(24)))
    X["TodCat"] = pd.Categorical((mins // 15).astype(int), categories=pd.Index(range(96)))
    X["HourCarr"] = pd.Categorical(
        hour.astype(str) + "_" + df["UniqueCarrier"].astype(str), categories=HOUR_CARR_CATS
    )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=350,
    learning_rate=0.1,
    max_depth=8,
    min_child_weight=20,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=100,
    gamma=0.5,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
