"""XGBoost binary classifier for airline delay prediction. THIS IS THE ONLY FILE THE AGENT EDITS.

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
RAW_COLS = [c for c in train.columns if c not in ID_COLS + [TARGET]]
CAT_RAW = ["UniqueCarrier", "Origin", "Dest"]
Cyc = dict(Month=("c-", 12), DayofMonth=("c-", 31), DayOfWeek=("c-", 7))

# smoothed target encodings, fit on TRAIN ONLY
Y = (train[TARGET] == POSITIVE).astype(float)
PRIOR = Y.mean()
M_TE = 20.0


def _te_map(keys: pd.Series) -> dict:
    g = Y.groupby(keys)
    return ((g.sum() + PRIOR * M_TE) / (g.transform("count") + M_TE)).to_dict()


TE_HOUR = _te_map((train["DepTime"] // 100 % 24))
TE_CARRIER = _te_map(train["UniqueCarrier"])
TE_ORIGIN = _te_map(train["Origin"])
TE_DEST = _te_map(train["Dest"])

# flight-frequency features (train-only counts; hub size is stable year-over-year)
CNT_ORIGIN = train["Origin"].value_counts()
CNT_DEST = train["Dest"].value_counts()
CNT_CARRIER = train["UniqueCarrier"].value_counts()
CNT_ROUTE = (train["Origin"] + "->" + train["Dest"]).value_counts()
# airport-hour congestion volumes (stable year-over-year, unlike target-rate interactions)
CNT_ORIG_HOUR = (train["Origin"] + "@" + (train["DepTime"] // 100 % 24).astype(str)).value_counts()
CNT_DEST_HOUR = (train["Dest"] + "@" + (train["DepTime"] // 100 % 24).astype(str)).value_counts()


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace(r"^c-", "", regex=True), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c, (prefix, _max) in Cyc.items():
        X[c] = _num(df[c]).astype("float32")
    dt = df["DepTime"].astype("float32")
    X["DepTime"] = dt
    X["hour"] = dt // 100 % 24
    X["minute_of_day"] = (dt // 100 % 24) * 60 + dt % 100
    X["Distance"] = df["Distance"].astype("float32")
    for c in CAT_RAW:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    hour = df["DepTime"] // 100 % 24
    X["te_hour"] = hour.map(TE_HOUR).astype("float32").fillna(PRIOR)
    X["te_carrier"] = df["UniqueCarrier"].map(TE_CARRIER).astype("float32").fillna(PRIOR)
    X["te_origin"] = df["Origin"].map(TE_ORIGIN).astype("float32").fillna(PRIOR)
    X["te_dest"] = df["Dest"].map(TE_DEST).astype("float32").fillna(PRIOR)
    route = df["Origin"] + "->" + df["Dest"]
    X["cnt_origin"] = df["Origin"].map(CNT_ORIGIN).fillna(0).astype("float32")
    X["cnt_dest"] = df["Dest"].map(CNT_DEST).fillna(0).astype("float32")
    X["cnt_carrier"] = df["UniqueCarrier"].map(CNT_CARRIER).fillna(0).astype("float32")
    X["cnt_route"] = route.map(CNT_ROUTE).fillna(0).astype("float32")
    X["cnt_orig_hour"] = (df["Origin"] + "@" + hour.astype(str)).map(CNT_ORIG_HOUR).fillna(0).astype("float32")
    X["cnt_dest_hour"] = (df["Dest"] + "@" + hour.astype(str)).map(CNT_DEST_HOUR).fillna(0).astype("float32")
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_RAW}

# --- model --------------------------------------------------------------------
# bagged ensemble of XGBoost models with different seeds (variance reduction)
N_BAGS = 5
models = []
t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
for k in range(N_BAGS):
    m = xgb.XGBClassifier(
        n_estimators=4000,
        max_depth=18,
        learning_rate=0.03,
        tree_method="hist",
        enable_categorical=True,
        min_child_weight=1,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        early_stopping_rounds=100,
        random_state=SEED + 1000 * k,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    print(f"  bag {k}: best_iter={m.best_iteration}")
    models.append(m)
model = models[0]  # keep a module-level handle
print(f"Training time: {time.time() - t0:.1f}s")

def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = np.stack([m.predict_proba(X, iteration_range=(0, m.best_iteration + 1))[:, 1] for m in models])
    return ps.mean(axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")