"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- features -----------------------------------------------------------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
NUM_COLS = ["DepTime", "dep_min", "sin_t", "cos_t", "Distance", "route_n", "route_dist_mu"]
INTER_COLS = ["hour_c", "car_hh", "dow_hour"]
FEATURES = NUM_COLS + CAT_COLS + INTER_COLS


def _hour_str(df: pd.DataFrame) -> pd.Series:
    return ((df["DepTime"].astype(float) // 100) % 24).astype(int).astype(str)


def _hh_str(df: pd.DataFrame) -> pd.Series:
    return ((((df["DepTime"].astype(float) // 100) % 24) * 60 + df["DepTime"].astype(float) % 100) // 30).astype(int).astype(str)


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


# train-fitted lookups (never fit on the dataframe passed to predict_proba)
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
route_cnt = _route(train).value_counts()
route_dist = train.groupby(_route(train))["Distance"].mean()
_car_hh_train = train["UniqueCarrier"].astype(str) + "@" + _hh_str(train)
_dow_hour_train = train["DayOfWeek"].astype(str) + "@" + _hour_str(train)
chh_levels = pd.Index(sorted(_car_hh_train.unique()))
dwh_levels = pd.Index(sorted(_dow_hour_train.unique()))
DIST_DEFAULT = float(train["Distance"].mean())


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(float)
    X["DepTime"] = dep
    X["dep_min"] = ((dep // 100) % 24) * 60 + dep % 100
    X["sin_t"] = np.sin(2 * np.pi * X["dep_min"] / 1440.0)
    X["cos_t"] = np.cos(2 * np.pi * X["dep_min"] / 1440.0)
    X["Distance"] = df["Distance"].astype(float)
    route = _route(df)
    X["route_n"] = route.map(route_cnt).fillna(0)
    X["route_dist_mu"] = route.map(route_dist).fillna(DIST_DEFAULT)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    X["hour_c"] = pd.Categorical(_hour_str(df))
    X["car_hh"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "@" + _hh_str(df), categories=chh_levels)
    X["dow_hour"] = pd.Categorical(
        df["DayOfWeek"].astype(str) + "@" + _hour_str(df), categories=dwh_levels)
    return X[FEATURES]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=1500,
    max_depth=4,
    learning_rate=0.05,
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
