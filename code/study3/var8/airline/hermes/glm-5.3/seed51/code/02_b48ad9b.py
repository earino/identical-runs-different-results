"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
GM = float(y_train.mean())  # global mean delay rate (train only)


# --- target encodings: smoothed mean of y per group, fit on train only --------
def _fit_te(keys: pd.Series, y: np.ndarray, alpha: float = 20.0):
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    te = (g["sum"] + alpha * GM) / (g["count"] + alpha)
    return te.to_dict()


def _route(df):
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


def _hour(df):
    return (df["DepTime"] // 100).clip(0, 24).astype(int)


TE_ALPHA = 20.0
TE_MAPS = {
    "te_carrier": _fit_te(train["UniqueCarrier"], y_train, TE_ALPHA),
    "te_origin": _fit_te(train["Origin"], y_train, TE_ALPHA),
    "te_dest": _fit_te(train["Dest"], y_train, TE_ALPHA),
    "te_route": _fit_te(_route(train), y_train, TE_ALPHA),
    "te_hour": _fit_te(_hour(train), y_train, TE_ALPHA),
    "te_origin_hour": _fit_te(train["Origin"].astype(str) + "_" + _hour(train).astype(str), y_train, TE_ALPHA),
    "te_dest_hour": _fit_te(train["Dest"].astype(str) + "_" + _hour(train).astype(str), y_train, TE_ALPHA),
}
CNT_MAPS = {
    "cnt_origin": train["Origin"].value_counts().to_dict(),
    "cnt_dest": train["Dest"].value_counts().to_dict(),
    "cnt_route": _route(train).value_counts().to_dict(),
}


# --- features -----------------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]


def fe(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls fe() on unseen rows,
    # so anything computed on `train`/`evald` outside this function will NOT be applied
    # to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    hour = _hour(df)
    mins = hour * 60 + df["DepTime"] % 100
    X["hour"] = hour
    X["dep_minutes"] = mins
    X["dep_sin"] = np.sin(2 * np.pi * mins / 1440)
    X["dep_cos"] = np.cos(2 * np.pi * mins / 1440)
    X["month"] = df["Month"].str.slice(2).astype(int)
    X["day"] = df["DayofMonth"].str.slice(2).astype(int)
    X["dow"] = df["DayOfWeek"].str.slice(2).astype(int)
    X["distance_log"] = np.log1p(df["Distance"].astype(float))
    X["distance"] = df["Distance"].astype(float)
    for name, keys in [
        ("te_carrier", df["UniqueCarrier"]),
        ("te_origin", df["Origin"]),
        ("te_dest", df["Dest"]),
        ("te_route", _route(df)),
        ("te_hour", hour),
        ("te_origin_hour", df["Origin"].astype(str) + "_" + hour.astype(str)),
        ("te_dest_hour", df["Dest"].astype(str) + "_" + hour.astype(str)),
    ]:
        X[name] = keys.astype(str).map(TE_MAPS[name]).astype(float).fillna(GM)
    for name, keys in [
        ("cnt_origin", df["Origin"]),
        ("cnt_dest", df["Dest"]),
        ("cnt_route", _route(df)),
    ]:
        X[name] = keys.astype(str).map(CNT_MAPS[name]).astype(float).fillna(0.0)
        X[name] = np.log1p(X[name])
    for c in CAT_COLS:
        X[c] = df[c].to_numpy()
    return X


cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = fe(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=2000,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
# internal split for early stopping (train is 2005, eval is 2006; keep time split honest)
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(train))
cut = int(0.9 * len(train))
tr_i, va_i = idx[:cut], idx[cut:]
Xp = prepare(train)
y = to_y(train)
m_es = xgb.XGBClassifier(early_stopping_rounds=50, **PARAMS)
m_es.fit(Xp.iloc[tr_i], y[tr_i], eval_set=[(Xp.iloc[va_i], y[va_i])], verbose=False)
best_round = m_es.best_iteration + 1
print(f"ES rounds: {best_round}, training time {time.time() - t0:.1f}s")

p2 = dict(PARAMS)
p2["n_estimators"] = best_round
model = xgb.XGBClassifier(**p2)
model.fit(Xp, y)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
