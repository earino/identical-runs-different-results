"""XGBoost binary classifier for airline delay prediction.

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
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# smoothed target encoding, fit on TRAIN ONLY (prior = train base rate)
_prior = float((train[TARGET] == POSITIVE).mean())
_K = 30.0


def _te_map(keys: pd.Series, y: np.ndarray) -> pd.Series:
    """smoothed target mean per key, computed from (train keys, train y)"""
    df = pd.DataFrame({"k": keys.to_numpy(), "y": y})
    g = df.groupby("k", observed=True)["y"].agg(["sum", "count"])
    enc = (g["sum"] + _K * _prior) / (g["count"] + _K)
    return enc

_y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
_route_train = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
TE_MAPS = {}
TE_MAPS["Route"] = _te_map(_route_train, _y_train)
TE_MAPS["Origin"] = _te_map(train["Origin"], _y_train)
TE_MAPS["Dest"] = _te_map(train["Dest"], _y_train)
TE_MAPS["UniqueCarrier"] = _te_map(train["UniqueCarrier"], _y_train)
TE_MAPS["hour"] = _te_map((train["DepTime"] // 100).astype(str), _y_train)
TE_KEYS = {
    "te_route": ("Route", lambda X: X["Origin"].astype(str) + "_" + X["Dest"].astype(str)),
    "te_origin": ("Origin", lambda X: X["Origin"].astype(str)),
    "te_dest": ("Dest", lambda X: X["Dest"].astype(str)),
    "te_carrier": ("UniqueCarrier", lambda X: X["UniqueCarrier"].astype(str)),
    "te_hour": ("hour", lambda X: (pd.to_numeric(X["DepTime"], errors="coerce") // 100).astype("Int64").astype(str)),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df.copy()
    dt = pd.to_numeric(X["DepTime"], errors="coerce")
    hour = (dt // 100).astype(float)
    frac_hour = hour + (dt % 100) / 60.0
    month_num = X["Month"].astype(str).str.slice(2).astype(int)
    day_num = X["DayofMonth"].astype(str).str.slice(2).astype(int)
    dow_num = X["DayOfWeek"].astype(str).str.slice(2).astype(int)
    out = pd.DataFrame(index=X.index)
    for c, v in [
        ("DepTime", dt.astype(float)),
        ("Distance", pd.to_numeric(X["Distance"], errors="coerce").astype(float)),
        ("hour", hour),
        ("frac_hour", frac_hour),
        ("hour_sin", np.sin(2 * np.pi * frac_hour / 24.0)),
        ("hour_cos", np.cos(2 * np.pi * frac_hour / 24.0)),
        ("month_num", month_num.astype(float)),
        ("day_num", day_num.astype(float)),
        ("dow_num", dow_num.astype(float)),
        ("dow_sin", np.sin(2 * np.pi * dow_num / 7.0)),
        ("dow_cos", np.cos(2 * np.pi * dow_num / 7.0)),
        ("month_sin", np.sin(2 * np.pi * month_num / 12.0)),
        ("month_cos", np.cos(2 * np.pi * month_num / 12.0)),
    ]:
        out[c] = v
    for c, (key, keyfn) in TE_KEYS.items():
        enc = TE_MAPS[key]
        out[c] = keyfn(X).map(enc).astype(float).fillna(_prior)
    for c in CAT_COLS:
        out[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return out


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=3000,
    max_depth=6,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="auc",
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
model = xgb.XGBClassifier(**PARAMS)
model.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
