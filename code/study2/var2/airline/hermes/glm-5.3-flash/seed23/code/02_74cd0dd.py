"""XGBoost binary classifier for airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

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
RAW_FEATURES = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in RAW_FEATURES if train[c].dtype == object]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in obj_cols}
NUMERIC_COLS = ["DepTime", "Distance"]

# target encodings, fit on TRAIN only
y01 = (train[TARGET] == POSITIVE).astype(float)
GLOBAL_MEAN = y01.mean()


def route_key(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + ">" + df["Dest"].astype(str)


train_route = route_key(train)
TE_COLS = ["UniqueCarrier", "Origin", "Dest"]
TE_SMOOTH = {"UniqueCarrier": 30, "Origin": 40, "Dest": 40, "route": 60}
te_maps = {}
for c in TE_COLS:
    grp = y01.groupby(train[c])
    n = grp.count()
    m = grp.mean()
    k = TE_SMOOTH[c]
    te_maps[c] = ((m * n + GLOBAL_MEAN * k) / (n + k)).to_dict()
grp = y01.groupby(train_route)
te_maps["route"] = ((grp.mean() * grp.count() + GLOBAL_MEAN * TE_SMOOTH["route"]) / (grp.count() + TE_SMOOTH["route"])).to_dict()
route_counts = train_route.value_counts().to_dict()


def add_numeric_feats(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    dt = df["DepTime"].astype("int64")
    hour = (dt // 100) % 24
    minute = dt % 100
    frac = (hour * 60 + minute) / 1440.0
    X["dep_hour"] = hour.astype("int64")
    X["dep_minute"] = minute.astype("int64")
    X["dep_sin"] = np.sin(2 * np.pi * frac)
    X["dep_cos"] = np.cos(2 * np.pi * frac)
    X["log_dist"] = np.log1p(df["Distance"].astype("float64"))
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in obj_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    for c in NUMERIC_COLS:
        X[c] = df[c].astype("float64")
    X = add_numeric_feats(df, X)
    # target encodings (maps fit on train only; unseen -> global mean)
    for c in TE_COLS:
        X["te_" + c] = df[c].map(te_maps[c]).astype("float64").fillna(GLOBAL_MEAN)
    rte = route_key(df)
    X["te_route"] = rte.map(te_maps["route"]).astype("float64").fillna(GLOBAL_MEAN)
    X["route_count"] = rte.map(route_counts).astype("float64").fillna(0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=50,
)

t0 = time.time()
model.fit(
    prepare(train),
    to_y(train),
    eval_set=[(prepare(evald), to_y(evald))],
    verbose=False,
)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
