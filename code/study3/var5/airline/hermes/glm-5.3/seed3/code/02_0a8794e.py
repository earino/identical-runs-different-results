"""XGBoost binary classifier for airline delay prediction.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
"""
import json
import os
import re
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
C_NUM = {"Month": "month", "DayofMonth": "day", "DayOfWeek": "dow"}  # c-<n> strings -> ints
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
# global category levels from TRAIN ONLY (unseen levels in eval/holdout -> NaN -> missing)
CAT_LEVELS = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
CAT_LEVELS["route"] = pd.Index(sorted(train["Origin"].astype(str) + "_" + train["Dest"].astype(str)))


def _cnum(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"c-(\d+)", expand=False).astype(float)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c, name in C_NUM.items():
        X[name] = _cnum(df[c])
    t = df["DepTime"].astype(float)
    hh = np.clip((t // 100).astype(float), 0, 24)
    mm = t % 100
    minutes = hh * 60 + mm
    X["minutes_of_day"] = minutes
    X["tod_sin"] = np.sin(2 * np.pi * minutes / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * minutes / 1440.0)
    X["distance"] = df["Distance"].astype(float)
    X["log_distance"] = np.log1p(df["Distance"].astype(float))
    X["route"] = pd.Categorical(df["Origin"].astype(str) + "_" + df["Dest"].astype(str), categories=CAT_LEVELS["route"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=CAT_LEVELS[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- smoothed target encodings (fit on TRAIN ONLY) ------------------------------
y_all = to_y(train)
PRIOR = float(y_all.mean())
ALPHA = 20.0

TE_COLS = ["UniqueCarrier", "Origin", "Dest", "route_src"]
train_te = pd.DataFrame({
    "UniqueCarrier": train["UniqueCarrier"],
    "Origin": train["Origin"],
    "Dest": train["Dest"],
    "route_src": train["Origin"].astype(str) + "_" + train["Dest"].astype(str),
})


def fit_te_map(series: pd.Series, y: np.ndarray) -> dict:
    g = pd.DataFrame({"k": series.values, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return ((g["sum"] + PRIOR * ALPHA) / (g["count"] + ALPHA)).to_dict()


TE_MAPS = {c: fit_te_map(train_te[c], y_all) for c in TE_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = build_features(df)
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["te_carrier"] = df["UniqueCarrier"].map(TE_MAPS["UniqueCarrier"]).astype(float).fillna(PRIOR)
    X["te_origin"] = df["Origin"].map(TE_MAPS["Origin"]).astype(float).fillna(PRIOR)
    X["te_dest"] = df["Dest"].map(TE_MAPS["Dest"]).astype(float).fillna(PRIOR)
    X["te_route"] = route.map(TE_MAPS["route_src"]).astype(float).fillna(PRIOR)
    return X


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
Xy = prepare(train)
n_tr = int(len(Xy) * 0.8)
X_tr, y_tr = Xy.iloc[:n_tr], y_all[:n_tr]
X_val, y_val = Xy.iloc[n_tr:], y_all[n_tr:]
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
