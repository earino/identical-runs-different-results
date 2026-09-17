"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
hour_levels = pd.Index(sorted(((train["DepTime"] // 100).astype(int)).astype(str).unique()))

# smooth target encodings, computed on train only
_prior = float((train[TARGET] == POSITIVE).mean())


def _target_encode(col, smoothing: float) -> pd.Series:
    grp = train.groupby(col)[TARGET]
    enc = (grp.sum().eq(POSITIVE).astype(float) + _prior * smoothing) / (grp.size() + smoothing)
    return enc.astype(float)


origin_te = _target_encode("Origin", 500.0)
carrier_te = _target_encode("UniqueCarrier", 100.0)
timebin_te = _target_encode(
    ((train["DepTime"] // 100 * 60 + train["DepTime"] % 100) // 15).astype(str), 200.0
)
dow_te = _target_encode(train["DayOfWeek"].astype(str), 200.0)

# full-train route encoding (for non-train rows)
route_te_full = _target_encode(
    train["Origin"].astype(str) + "_" + train["Dest"].astype(str), 500.0
)
hour_te_full = _target_encode((train["DepTime"] // 100).astype(str), 300.0)
month_te_full = _target_encode(train["Month"].astype(str), 300.0)
day_te_full = _target_encode(train["DayofMonth"].astype(str), 500.0)

# --- out-of-fold target encodings for the TRAINING rows (kills TE self-leakage) ---
TE_SPECS = {
    "OriginTE": (train["Origin"].astype(str), 500.0),
    "CarrierTE": (train["UniqueCarrier"].astype(str), 100.0),
    "TimeBinTE": (((train["DepTime"] // 100 * 60 + train["DepTime"] % 100) // 15).astype(str), 200.0),
    "DowTE": (train["DayOfWeek"].astype(str), 200.0),
    "RouteTE": (
        train["Origin"].astype(str) + "_" + train["Dest"].astype(str), 500.0
    ),
    "HourTE": ((train["DepTime"] // 100).astype(str), 300.0),
    "MonthTE": (train["Month"].astype(str), 300.0),
    "DayTE": (train["DayofMonth"].astype(str), 500.0),
}
_y = (train[TARGET] == POSITIVE).astype(float).to_numpy()
_oof_te: dict = {name: np.zeros(len(train)) for name in TE_SPECS}
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for tr_idx, va_idx in kf.split(train):
    fold = train.iloc[tr_idx]
    prior = float(_y[tr_idx].mean())
    for name, (col_vals, sm) in TE_SPECS.items():
        grp = fold.assign(_v=col_vals.iloc[tr_idx], _y=_y[tr_idx]).groupby("_v")["_y"]
        enc = (grp.sum() + prior * sm) / (grp.size() + sm)
        _oof_te[name][va_idx] = col_vals.iloc[va_idx].map(enc).fillna(prior).to_numpy()
OOF_TE_INDEX = train.index

MONTH_LEN = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
MONTH_START = np.concatenate([[0], np.cumsum(MONTH_LEN)[:-1]])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["Month"] = df["Month"].str.replace("c-", "", regex=False).astype(int)
    X["DayofMonth"] = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int)
    X["DayOfWeek"] = df["DayOfWeek"].str.replace("c-", "", regex=False).astype(int)
    X["DepTime"] = df["DepTime"].astype(int)
    X["Distance"] = df["Distance"].astype(float)
    X["Hour"] = X["DepTime"] // 100
    X["Minute"] = X["DepTime"] % 100
    X["DepTimeMin"] = X["Hour"] * 60 + X["Minute"]
    X["DayOfYear"] = MONTH_START[X["Month"].to_numpy() - 1] + X["DayofMonth"].to_numpy()
    X["HourCat"] = pd.Categorical(X["Hour"].astype(str), categories=hour_levels)
    X["TimeBin"] = X["DepTimeMin"] // 15  # 96 bins over the day
    X["DistPerMin"] = X["Distance"] / (X["DepTimeMin"] + 1)
    X["OriginTE"] = df["Origin"].astype(str).map(origin_te).astype(float)
    X["CarrierTE"] = df["UniqueCarrier"].astype(str).map(carrier_te).astype(float)
    X["TimeBinTE"] = X["TimeBin"].astype(str).map(timebin_te).astype(float)
    X["DowTE"] = df["DayOfWeek"].astype(str).map(dow_te).astype(float)
    X["RouteTE"] = route_te_full.reindex(
        df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    ).fillna(_prior).to_numpy()
    X["HourTE"] = (df["DepTime"].astype(int) // 100).astype(str).map(hour_te_full).astype(float)
    X["MonthTE"] = df["Month"].astype(str).map(month_te_full).astype(float)
    X["DayTE"] = df["DayofMonth"].astype(str).map(day_te_full).astype(float)
    # training rows use out-of-fold TE values; all other rows use the full-train encodings
    if df.index.equals(OOF_TE_INDEX):
        for name in TE_SPECS:
            X[name] = _oof_te[name]
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
BASE_PARAMS = dict(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    subsample=0.5,
    colsample_bytree=0.5,
    min_child_weight=5,
    reg_lambda=3.0,
    reg_alpha=2.0,
    n_jobs=N_JOBS,
)

models = []
t0 = time.time()
CONFIGS = [
    (42, 8, 0.5), (7, 6, 0.5), (2024, 10, 0.5),
    (11, 8, 0.6), (99, 6, 0.6), (123, 10, 0.6),
    (5, 7, 0.4), (77, 9, 0.4), (313, 7, 0.4),
]
for seed, depth, ss in CONFIGS:
    params = {**BASE_PARAMS, "max_depth": depth, "subsample": ss}
    m = xgb.XGBClassifier(random_state=seed, **params)
    m.fit(prepare(train), to_y(train))
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    preds = [m.predict_proba(Xp)[:, 1] for m in models]
    return np.mean(preds, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
