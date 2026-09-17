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
from sklearn.model_selection import StratifiedKFold, train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature engineering ------------------------------------------------------
BASE_CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
TIME_BINS = ["night", "early_morning", "morning", "midday", "afternoon", "evening"]
_TODS = TIME_BINS + ["late_night"]
_HOURS = [str(h) for h in range(24)]

DERIVED = [
    "DepHour", "DepMinute", "DepMinSin", "DepMinCos",
    "MonthN", "DayOfMonthN", "DayOfWeekN", "MonthSin", "MonthCos",
    "DayOfWeekSin", "DayOfWeekCos", "LogDistance", "Weekend",
]

CAT_COLS = BASE_CAT_COLS + ["DepTimeBin"]
FEATURE_COLS = CAT_COLS + DERIVED

# ---- target encoding (fit on TRAIN only; maps stored at module level) --------
_SMOOTH_M = {"UniqueCarrier": 20, "Origin": 20, "Dest": 20, "Route": 100, "OriginHour": 50,
             "DestTOD": 100, "CarrierHour": 30, "Hour": 60, "CarrierDist": 40,
             "RouteHour": 150, "CarrierTOD": 60}
TE_COLS = list(_SMOOTH_M)  # + ["RouteCount"] handled separately


def _tod_bin(hour: pd.Series) -> pd.Series:
    b = pd.cut(hour, bins=[-1, 4, 7, 10, 15, 19, 23, 24], labels=TIME_BINS + ["late_night"])
    return b.astype(str).where(hour.notna(), "missing")


def _te_tokens(df: pd.DataFrame) -> dict:
    """Token series per TE column (raw strings, no levels needed)."""
    org = df["Origin"].astype(str)
    dst = df["Dest"].astype(str)
    car = df["UniqueCarrier"].astype(str)
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dt // 100).clip(0, 23)
    tod = _tod_bin(hour)
    return {
        "UniqueCarrier": car,
        "Origin": org,
        "Dest": dst,
        "Route": org + "__" + dst,
        "OriginHour": org + "__" + hour.astype("Int64").astype(str),
        "DestTOD": dst + "__" + tod,
        "CarrierHour": car + "__" + hour.astype("Int64").astype(str),
        "Hour": hour.astype("Int64").astype(str),
        "CarrierDist": car + "__" + (pd.to_numeric(df["Distance"], errors="coerce") // 250).astype("Int64").astype(str),
        "RouteHour": (org + "__" + dst) + "__" + hour.astype("Int64").astype(str),
        "CarrierTOD": car + "__" + tod,
    }


def _te_key(tok: pd.Series, col: str) -> pd.Series:
    return tok.where(tok.isin(TE_MAPS[col].index), "__UNSEEN__")


TE_MAPS = {}
TE_PRIOR = None

y_full = (train[TARGET] == POSITIVE).astype(int).to_numpy()
TE_PRIOR = float(y_full.mean())

_tokens_tr = _te_tokens(train)
for _col in TE_COLS:
    _tok = _tokens_tr[_col]
    _m = _SMOOTH_M[_col]
    _grp = pd.DataFrame({"tok": _tok, "y": y_full}).groupby("tok")["y"].agg(["sum", "count"])
    _enc = (_grp["sum"] + _m * TE_PRIOR) / (_grp["count"] + _m)
    TE_MAPS[_col] = _enc
    # add the encoded column name to FEATURE_COLS once
    if _col not in FEATURE_COLS:
        FEATURE_COLS = FEATURE_COLS + [_col]

# OOF encoding for train rows (avoids self-label leakage in the training matrix)
OOF_TE = {}
_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for _col in TE_COLS:
    _tok = _tokens_tr[_col].to_numpy()
    _m = _SMOOTH_M[_col]
    _oof = np.full(len(_tok), np.nan)
    for _tr_idx, _va_idx in _kf.split(np.zeros(len(_tok)), y_full):
        _g = pd.DataFrame({"tok": _tok[_tr_idx], "y": y_full[_tr_idx]}).groupby("tok")["y"].agg(["sum", "count"])
        _e = (_g["sum"] + _m * TE_PRIOR) / (_g["count"] + _m)
        _oof[_va_idx] = pd.Series(_tok[_va_idx]).map(_e).fillna(TE_PRIOR).to_numpy()
    OOF_TE[_col] = _oof

# route frequency from train
_route_counts = _tokens_tr["Route"].value_counts()
ROUTE_COUNT_MAP = np.log1p(_route_counts).to_dict()
FEATURE_COLS = FEATURE_COLS + ["RouteCount"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in BASE_CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=sorted(train[c].dropna().astype(str).unique()))

    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dt // 100).clip(0, 23)
    minute = (dt % 100).clip(0, 59)
    frac = (hour * 60 + minute) / 1440.0
    X["DepHour"] = hour
    X["DepMinute"] = minute
    X["DepMinSin"] = np.sin(2 * np.pi * frac)
    X["DepMinCos"] = np.cos(2 * np.pi * frac)
    tod = _tod_bin(hour)
    X["DepTimeBin"] = pd.Categorical(tod, categories=_TODS)

    def cnum(s):
        return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")

    month = cnum(df["Month"]).clip(1, 12)
    dom = cnum(df["DayofMonth"]).clip(1, 31)
    dow = cnum(df["DayOfWeek"]).clip(1, 7)
    X["MonthN"] = month
    X["DayOfMonthN"] = dom
    X["DayOfWeekN"] = dow
    X["MonthSin"] = np.sin(2 * np.pi * month / 12)
    X["MonthCos"] = np.cos(2 * np.pi * month / 12)
    X["DayOfWeekSin"] = np.sin(2 * np.pi * dow / 7)
    X["DayOfWeekCos"] = np.cos(2 * np.pi * dow / 7)

    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["LogDistance"] = np.log1p(dist)

    X["Weekend"] = (dow >= 6).astype(float)

    # target-encoded columns (maps fitted on train only)
    toks = _te_tokens(df)
    for col in TE_COLS:
        X[col] = toks[col].map(TE_MAPS[col]).fillna(TE_PRIOR).to_numpy()

    # route frequency
    X["RouteCount"] = toks["Route"].map(ROUTE_COUNT_MAP).fillna(0.0).to_numpy()
    return X[FEATURE_COLS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_all = prepare(train)
# replace TE cols with OOF values for training (no self-label leakage)
for _col in TE_COLS:
    X_all[_col] = OOF_TE[_col]

y_all = to_y(train)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_all, y_all, test_size=0.15, random_state=SEED, stratify=y_all
)

model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
best_it = int(model.best_iteration)
print(f"Training time: {time.time() - t0:.1f}s (best_iter={best_it})")

# refit on the full training set with the iteration count chosen by early stopping
final_model = xgb.XGBClassifier(
    n_estimators=best_it + 1,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
final_model.fit(X_all, y_all)
model = final_model


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
