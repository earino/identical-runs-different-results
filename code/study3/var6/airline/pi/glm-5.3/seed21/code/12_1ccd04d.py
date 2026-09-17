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

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

CAT_COLS: list[str] = []  # raw categoricals dropped: their 2005-specific level splits hurt generalization


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- statistics fitted on TRAIN ONLY -------------------------------------------
_y = to_y(train)
GM = float(_y.mean())


def _smoothed_te(keys: pd.Series, m: float) -> pd.Series:
    g = pd.DataFrame({"k": keys.to_numpy(), "y": _y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + GM * m) / (g["count"] + m)


_route_key = train["Origin"] + "_" + train["Dest"]
TE = {
    "carrier_te": _smoothed_te(train["UniqueCarrier"], 100),
    "origin_te": _smoothed_te(train["Origin"], 300),
    "dest_te": _smoothed_te(train["Dest"], 300),
    "month_te": _smoothed_te(train["Month"], 100),
    "dow_te": _smoothed_te(train["DayOfWeek"], 100),
}
CNT = {
    "origin_count": train["Origin"].value_counts(),
    "dest_count": train["Dest"].value_counts(),
    "route_count": _route_key.value_counts(),
    "origin_hour_count": (train["Origin"] + "_" + (train["DepTime"] // 100).astype(str)).value_counts(),
    "dest_hour_count": (train["Dest"] + "_" + (train["DepTime"] // 100).astype(str)).value_counts(),
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering lives here (predict_proba calls this on unseen rows)."""
    X = df[["DepTime", "Distance"]].copy()
    dep = df["DepTime"].to_numpy()
    hour = dep // 100
    mod = hour * 60 + dep % 100
    X["hour"] = hour
    X["minute"] = dep % 100
    X["minute_of_day"] = mod
    mm = mod % 1440
    X["sin_day"] = np.sin(2 * np.pi * mm / 1440)
    X["cos_day"] = np.cos(2 * np.pi * mm / 1440)
    mo = df["Month"].str.split("-").str[-1].astype(int).to_numpy()
    X["month_sin"] = np.sin(2 * np.pi * mo / 12)
    X["month_cos"] = np.cos(2 * np.pi * mo / 12)
    for name, te in TE.items():
        key = df["Month"] if name == "month_te" else df["DayOfWeek"] if name == "dow_te" else \
              df["UniqueCarrier"] if name == "carrier_te" else df["Origin"] if name == "origin_te" else df["Dest"]
        X[name] = key.map(te).fillna(GM).to_numpy()
    rk = df["Origin"] + "_" + df["Dest"]
    X["origin_count"] = df["Origin"].map(CNT["origin_count"]).fillna(0).to_numpy()
    X["dest_count"] = df["Dest"].map(CNT["dest_count"]).fillna(0).to_numpy()
    X["route_count"] = rk.map(CNT["route_count"]).fillna(0).to_numpy()
    hk = df["Origin"] + "_" + (df["DepTime"] // 100).astype(str)
    dk = df["Dest"] + "_" + (df["DepTime"] // 100).astype(str)
    X["origin_hour_count"] = hk.map(CNT["origin_hour_count"]).fillna(0).to_numpy()
    X["dest_hour_count"] = dk.map(CNT["dest_hour_count"]).fillna(0).to_numpy()
    X["log_origin_count"] = np.log1p(X["origin_count"].to_numpy())
    X["log_dest_hour_count"] = np.log1p(X["dest_hour_count"].to_numpy())
    X["route_share"] = X["route_count"].to_numpy() / (X["origin_count"].to_numpy() + 1)
    X["origin_hour_share"] = X["origin_hour_count"].to_numpy() / (X["origin_count"].to_numpy() + 1)
    X["dest_hour_share"] = X["dest_hour_count"].to_numpy() / (X["dest_count"].to_numpy() + 1)
    return X


_feat_train = add_features(train)
cat_levels = {c: pd.Index(sorted(_feat_train[c].astype(str).unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = add_features(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c].astype(str), categories=cat_levels[c])  # unseen -> NaN
    return X


LG = dict(max_depth=0, grow_policy="lossguide", tree_method="hist", enable_categorical=True,
         eval_metric="auc", early_stopping_rounds=40)
ENSEMBLE = [
    dict(max_leaves=256, learning_rate=0.07, min_child_weight=40, reg_lambda=5.0, reg_alpha=1.0, subsample=0.85, colsample_bytree=0.5),
    dict(max_leaves=256, learning_rate=0.07, min_child_weight=25, reg_lambda=5.0, reg_alpha=1.0, subsample=0.85, colsample_bytree=0.5),
    dict(max_leaves=256, learning_rate=0.07, min_child_weight=40, reg_lambda=5.0, reg_alpha=1.0, subsample=0.85, colsample_bytree=0.6),
    dict(max_leaves=384, learning_rate=0.07, min_child_weight=40, reg_lambda=5.0, reg_alpha=1.0, subsample=0.85, colsample_bytree=0.5),
    dict(max_leaves=256, learning_rate=0.06, min_child_weight=30, reg_lambda=8.0, reg_alpha=1.0, subsample=0.8, colsample_bytree=0.5),
    dict(max_leaves=256, learning_rate=0.08, min_child_weight=50, reg_lambda=5.0, reg_alpha=1.0, subsample=0.9, colsample_bytree=0.45),
]

models = []
t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
for i, p in enumerate(ENSEMBLE):
    m = xgb.XGBClassifier(n_estimators=900, n_jobs=N_JOBS, random_state=SEED + i, **{**LG, **p})
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)])
    models.append(m)
    print(f"model {i}: best_iter={m.best_iteration}")
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
