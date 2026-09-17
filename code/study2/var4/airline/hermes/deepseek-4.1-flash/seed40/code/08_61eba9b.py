"""XGBoost binary classifier for the airline delay task (autoresearch).

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


# --- feature engineering (fit on train only, applied inside prepare()) ---------
def _num(df, col):
    return df[col].str.replace("c-", "", regex=False).astype(int)


CAT_SRC = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_SRC}
route_levels = pd.Index(sorted((train["Origin"] + "_" + train["Dest"]).unique()))

# traffic volume as seen in the training year (stable structural feature, no label involved)
_cnt = {
    "carrier": train["UniqueCarrier"].value_counts(),
    "origin": train["Origin"].value_counts(),
    "dest": train["Dest"].value_counts(),
    "route": (train["Origin"] + "_" + train["Dest"]).value_counts(),
}

FEATURES = ["hour", "minute", "tmin", "sin_day", "cos_day", "DayOfWeek",
            "sin_dow", "cos_dow", "Distance", "log_distance",
            "cnt_carrier", "cnt_origin", "cnt_dest", "cnt_route", "route_share",
            "UniqueCarrier", "Origin", "Dest", "route"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw DataFrame -> model matrix. Everything here is reproducible on unseen rows."""
    X = pd.DataFrame(index=df.index)
    dt = df["DepTime"].to_numpy()
    hour = (dt // 100) % 24
    minute = dt % 100
    tmin = hour * 60.0 + minute
    X["hour"] = hour
    X["minute"] = minute
    X["tmin"] = tmin
    X["sin_day"] = np.sin(2 * np.pi * tmin / 1440.0)
    X["cos_day"] = np.cos(2 * np.pi * tmin / 1440.0)
    X["Month"] = _num(df, "Month")
    X["DayofMonth"] = _num(df, "DayofMonth")
    X["DayOfWeek"] = _num(df, "DayOfWeek")
    X["sin_month"] = np.sin(2 * np.pi * X["Month"] / 12.0)
    X["cos_month"] = np.cos(2 * np.pi * X["Month"] / 12.0)
    X["sin_dow"] = np.sin(2 * np.pi * X["DayOfWeek"] / 7.0)
    X["cos_dow"] = np.cos(2 * np.pi * X["DayOfWeek"] / 7.0)
    X["Distance"] = df["Distance"].astype(float)
    X["log_distance"] = np.log1p(X["Distance"])
    for c in CAT_SRC:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    route = df["Origin"] + "_" + df["Dest"]
    X["route"] = pd.Categorical(route, categories=route_levels)
    X["cnt_carrier"] = df["UniqueCarrier"].map(_cnt["carrier"]).fillna(0.0).astype(float)
    X["cnt_origin"] = df["Origin"].map(_cnt["origin"]).fillna(0.0).astype(float)
    X["cnt_dest"] = df["Dest"].map(_cnt["dest"]).fillna(0.0).astype(float)
    X["cnt_route"] = route.map(_cnt["route"]).fillna(0.0).astype(float)
    X["route_share"] = X["cnt_route"] / X["cnt_origin"]
    return X[FEATURES]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X = prepare(train)
y = to_y(train)

params = dict(
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
SEEDS = [42, 7, 2024]
models = []
for s in SEEDS:
    m = xgb.XGBClassifier(n_estimators=2000, **{**params, "random_state": s})
    m.fit(X, y, verbose=False)
    models.append(m)
    print(f"  seed {s} done ({time.time() - t0:.1f}s)")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    p = np.zeros(len(Xp))
    for m in models:
        p += m.predict_proba(Xp)[:, 1]
    return p / len(models)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
