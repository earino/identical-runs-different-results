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

_y = (train[TARGET] == POSITIVE).astype(float).to_numpy()
_prior = float(_y.mean())


def _time_bin(dep_time: pd.Series) -> pd.Series:
    return ((dep_time // 100 * 60 + dep_time % 100) // 15).astype(str)


# TE definitions: name -> (train-col values, smoothing). Computed on train only.
TE_SPECS = {
    "OriginTE": (train["Origin"].astype(str), 500.0),
    "DestTE": (train["Dest"].astype(str), 500.0),
    "CarrierTE": (train["UniqueCarrier"].astype(str), 100.0),
    "TimeBinTE": (_time_bin(train["DepTime"]), 200.0),
    "DowTE": (train["DayOfWeek"].astype(str), 200.0),
    "RouteTE": (train["Origin"].astype(str) + "_" + train["Dest"].astype(str), 500.0),
    "HourTE": ((train["DepTime"] // 100).astype(str), 300.0),
    "MonthTE": (train["Month"].astype(str), 300.0),
    "DayTE": (train["DayofMonth"].astype(str), 500.0),
    "OrigTB": (train["Origin"].astype(str) + "_" + _time_bin(train["DepTime"]), 1000.0),
}


def _te_lookup(vals: pd.Series, y: np.ndarray, prior: float, smoothing: float) -> pd.Series:
    grp = pd.Series(y).groupby(pd.Series(vals).reset_index(drop=True).values)
    return (grp.sum() + prior * smoothing) / (grp.size() + smoothing)


# full-train encodings: what unseen (holdout) rows will use
te_full = {
    name: _te_lookup(vals, _y, _prior, sm) for name, (vals, sm) in TE_SPECS.items()
}

# out-of-fold encodings: only for the training matrix itself (kills TE self-leakage)
oof_te = {name: np.zeros(len(train)) for name in TE_SPECS}
kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
for tr_idx, va_idx in kf.split(train):
    pr = float(_y[tr_idx].mean())
    for name, (vals, sm) in TE_SPECS.items():
        v = pd.Series(vals).reset_index(drop=True)
        enc = _te_lookup(v.iloc[tr_idx], _y[tr_idx], pr, sm)
        oof_te[name][va_idx] = v.iloc[va_idx].map(enc).fillna(pr).to_numpy()

MONTH_LEN = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
MONTH_START = np.concatenate([[0], np.cumsum(MONTH_LEN)[:-1]])


def _te_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Full-train smoothed target encodings for an arbitrary raw dataframe."""
    tb = _time_bin(df["DepTime"])
    vals = {
        "OriginTE": df["Origin"].astype(str),
        "DestTE": df["Dest"].astype(str),
        "CarrierTE": df["UniqueCarrier"].astype(str),
        "TimeBinTE": tb,
        "DowTE": df["DayOfWeek"].astype(str),
        "RouteTE": df["Origin"].astype(str) + "_" + df["Dest"].astype(str),
        "HourTE": (df["DepTime"] // 100).astype(str),
        "MonthTE": df["Month"].astype(str),
        "DayTE": df["DayofMonth"].astype(str),
        "OrigTB": df["Origin"].astype(str) + "_" + tb,
    }
    out = pd.DataFrame(index=df.index)
    for name, v in vals.items():
        out[name] = (
            pd.Series(v).reset_index(drop=True).map(te_full[name]).astype(float).fillna(_prior).to_numpy()
        )
    return out


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
    X = pd.concat([X, _te_columns(df)], axis=1)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
BASE_PARAMS = dict(
    n_estimators=350,
    max_depth=8,
    learning_rate=0.03,
    tree_method="hist",
    enable_categorical=True,
    subsample=0.5,
    colsample_bytree=0.6,
    min_child_weight=5,
    reg_lambda=3.0,
    reg_alpha=2.0,
    n_jobs=N_JOBS,
)

CONFIGS = [
    (42, 8, 0.5), (7, 6, 0.5), (2024, 10, 0.5),
    (11, 8, 0.6), (99, 6, 0.6), (123, 10, 0.6),
    (5, 7, 0.4), (77, 9, 0.4), (313, 7, 0.4),
    (2, 8, 0.5), (17, 6, 0.5), (55, 10, 0.5),
]

models = []
t0 = time.time()
X_train = prepare(train)
for name in TE_SPECS:  # training rows get the honest out-of-fold values
    X_train[name] = oof_te[name]
for seed, depth, ss in CONFIGS:
    params = {**BASE_PARAMS, "max_depth": depth, "subsample": ss}
    m = xgb.XGBClassifier(random_state=seed, **params)
    m.fit(X_train, to_y(train))
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
