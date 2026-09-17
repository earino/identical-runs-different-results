"""XGBoost binary classifier for airline delays. Only file the agent edits.

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

# --- fit feature statistics on TRAIN ONLY -------------------------------------
cat_levels = {
    "Month": pd.Index(sorted(train["Month"].unique())),
    "DayOfWeek": pd.Index(sorted(train["DayOfWeek"].unique())),
    "UniqueCarrier": pd.Index(sorted(train["UniqueCarrier"].unique())),
    "Origin": pd.Index(sorted(train["Origin"].unique())),
    "Dest": pd.Index(sorted(train["Dest"].unique())),
    "route": pd.Index(sorted((train["Origin"] + "_" + train["Dest"]).unique())),
}
# distance deciles from train
DIST_BINS = np.quantile(train["Distance"], np.linspace(0, 1, 11))
DIST_EDGES = np.unique(DIST_BINS)[1:-1]  # inner edges; values outside -> NaN cat? keep as -1/10 index


def _time_feats(d: pd.DataFrame) -> pd.DataFrame:
    dt = d["DepTime"].astype(float)
    hh = np.floor(dt / 100.0).astype(int) % 24
    mm = (dt % 100).astype(int)
    minutes = hh * 60 + mm
    X = pd.DataFrame({
        "DepTime_min": minutes,
        "hour": hh,
        "hour_sin": np.sin(2 * np.pi * hh / 24.0),
        "hour_cos": np.cos(2 * np.pi * hh / 24.0),
    })
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here; predict_proba() calls it on unseen rows."""
    t = _time_feats(df)
    X = pd.DataFrame(index=df.index)
    X["DepTime_min"] = t["DepTime_min"]
    X["hour"] = t["hour"]
    X["hour_sin"] = t["hour_sin"]
    X["hour_cos"] = t["hour_cos"]
    X["Month_num"] = df["Month"].str.replace("c-", "", regex=False).astype(int)
    X["Day_num"] = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int)
    X["Dow_num"] = df["DayOfWeek"].str.replace("c-", "", regex=False).astype(int)
    X["Month_sin"] = np.sin(2 * np.pi * X["Month_num"] / 12.0)
    X["Month_cos"] = np.cos(2 * np.pi * X["Month_num"] / 12.0)
    X["Dow_sin"] = np.sin(2 * np.pi * X["Dow_num"] / 7.0)
    X["Dow_cos"] = np.cos(2 * np.pi * X["Dow_num"] / 7.0)
    X["Distance"] = df["Distance"].astype(float)
    X["Distance_log"] = np.log1p(X["Distance"])
    X["Distance_bin"] = np.searchsorted(DIST_EDGES, X["Distance"], side="right")
    X["Month"] = pd.Categorical(df["Month"], categories=cat_levels["Month"])
    X["DayOfWeek"] = pd.Categorical(df["DayOfWeek"], categories=cat_levels["DayOfWeek"])
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"], categories=cat_levels["UniqueCarrier"])
    X["Origin"] = pd.Categorical(df["Origin"], categories=cat_levels["Origin"])
    X["Dest"] = pd.Categorical(df["Dest"], categories=cat_levels["Dest"])
    X["route"] = pd.Categorical(df["Origin"] + "_" + df["Dest"], categories=cat_levels["route"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10.0,
    reg_lambda=1.0,
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
