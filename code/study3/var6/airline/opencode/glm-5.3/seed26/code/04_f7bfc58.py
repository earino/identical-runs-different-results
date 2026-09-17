"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
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
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "route"]
_route_train = train["Origin"].astype(str) + "-" + train["Dest"].astype(str)
cat_levels = {
    "route": pd.Index(sorted(_route_train.unique())),
    **{c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS if c != "route"},
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    mon = df["Month"].str.replace("c-", "", regex=False).astype(int)
    dom = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int)
    dow = df["DayOfWeek"].str.replace("c-", "", regex=False).astype(int)
    X["month"] = mon
    X["day"] = dom
    X["dow"] = dow
    dt = df["DepTime"].astype(int)
    hour = dt // 100
    minute = dt % 100
    msm = hour * 60 + minute  # minutes since midnight
    X["hour"] = hour
    X["msm"] = msm
    X["msm_sin"] = np.sin(2 * np.pi * msm / 1440)
    X["msm_cos"] = np.cos(2 * np.pi * msm / 1440)
    X["mon_sin"] = np.sin(2 * np.pi * mon / 12)
    X["mon_cos"] = np.cos(2 * np.pi * mon / 12)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    X["dist"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(df["Distance"].astype(float))
    route = df["Origin"].astype(str) + "-" + df["Dest"].astype(str)
    for c in CAT_COLS:
        vals = route if c == "route" else df[c]
        X[c] = pd.Categorical(vals, categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=40,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s, best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
