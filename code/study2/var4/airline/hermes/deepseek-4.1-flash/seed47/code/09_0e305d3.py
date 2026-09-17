"""XGBoost binary classifier for the airline-delay task (see program.md for the contract).

  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     Every feature computation lives in `prepare()` and depends only on constants fitted from data/train.csv.

Notes on what transfers from 2005 (train) to 2006 (eval/holdout): clock-of-day, carrier and day-of-week
effects are stable across the year gap; per-month and per-route *target* rates are not.
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


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- raw columns ---------------------------------------------------------------
# Calendar identity (Month, DayofMonth) carries no transferable signal across the 2005->2006 gap: the
# per-month delay rate is flat in eval, so those columns give the trees nothing but overfitting room.
# DepTime is replaced by clean clock features derived inside prepare().
DROP = ["Month", "DayofMonth", "DepTime"]
RAW_FEATS = [
    c for c in train.columns
    if c not in ID_COLS + [TARGET] + DROP
    and (train[c].nunique() <= 1000 or not pd.api.types.is_string_dtype(train[c]))
]
CAT_COLS = [c for c in RAW_FEATS if pd.api.types.is_string_dtype(train[c])]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
CLOCK_COLS = ["hour", "minute_of_day", "tod_sin", "tod_cos"]


def _hour(df: pd.DataFrame) -> pd.Series:
    """DepTime is hhmm as an integer and contains junk above 2400 / with minutes >= 60."""
    t = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).to_numpy()
    return pd.Series(np.clip(t // 100, 0, 23), index=df.index)


def _clock(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).to_numpy()
    hh = np.clip(t // 100, 0, 23)
    mm = np.where(t % 100 < 60, t % 100, 0)
    minute_of_day = hh * 60 + mm
    ang = 2 * np.pi * minute_of_day / 1440.0
    return pd.DataFrame({
        "hour": hh.astype(np.float32),
        "minute_of_day": minute_of_day.astype(np.float32),
        "tod_sin": np.sin(ang).astype(np.float32),
        "tod_cos": np.cos(ang).astype(np.float32),
    }, index=df.index)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "hour":
        return _hour(df).astype(str)
    return df[name].astype(str)


def _key(df: pd.DataFrame, keyspec) -> pd.Series:
    out = _col(df, keyspec[0])
    for k in keyspec[1:]:
        out = out + "_" + _col(df, k)
    return out


# --- frequency (traffic-volume) encodings, fitted on training rows only --------
# How busy an airport / route / carrier is transfers across years far better than the target rate does.
COUNT_SPECS = [["Origin"], ["Dest"], ["UniqueCarrier"], ["Origin", "Dest"]]


def _fit_counts(keyspec):
    vc = _key(train, keyspec).value_counts()
    return {"name": "cnt_" + "_".join(keyspec), "keyspec": keyspec, "map": vc.to_dict(), "default": 1.0}


COUNT_FITS = [_fit_counts(sp) for sp in COUNT_SPECS]
COUNT_COLS = [f["name"] for f in COUNT_FITS]

# --- explicit crosses as categoricals (depth-4 trees cannot discover these) ----
CROSS_SPECS = [["UniqueCarrier", "hour"]]
CROSS_FITS = [
    {"name": "x_" + "_".join(sp), "keyspec": sp,
     "levels": pd.Index(sorted(_key(train, sp).unique()))}
    for sp in CROSS_SPECS
]
CROSS_COLS = [f["name"] for f in CROSS_FITS]

FEATURE_COLS = RAW_FEATS + CLOCK_COLS + COUNT_COLS + CROSS_COLS


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[RAW_FEATS].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    X = pd.concat([X, _clock(df)], axis=1)
    for f in COUNT_FITS:
        c = _key(df, f["keyspec"]).map(f["map"]).fillna(f["default"]).to_numpy(dtype=np.float32)
        X[f["name"]] = np.log1p(c)
    for f in CROSS_FITS:
        X[f["name"]] = pd.Categorical(_key(df, f["keyspec"]), categories=f["levels"])
    return X[FEATURE_COLS]


# --- model --------------------------------------------------------------------
# A small bag of XGBoost models; averaging decorrelated fits is a robust way to buy AUC without extra
# capacity, and it costs little because every member stays shallow and heavily subsampled.
ENSEMBLE = [
    dict(max_depth=6, colsample_bytree=0.60, subsample=0.90, random_state=SEED),
    dict(max_depth=6, colsample_bytree=0.50, subsample=0.85, random_state=SEED + 1),
    dict(max_depth=5, colsample_bytree=0.60, subsample=0.90, random_state=SEED + 2),
    dict(max_depth=7, colsample_bytree=0.60, subsample=0.90, random_state=SEED + 3),
    dict(max_depth=6, colsample_bytree=0.70, subsample=0.95, random_state=SEED + 4),
]

X_train = prepare(train)
y_train = to_y(train)
t0 = time.time()
MODELS = []
for cfg in ENSEMBLE:
    m = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        min_child_weight=20,
        reg_lambda=5.0,
        tree_method="hist",
        enable_categorical=True,
        n_jobs=N_JOBS,
        **cfg,
    )
    m.fit(X_train, y_train)
    MODELS.append(m)
print(f"Training time: {time.time() - t0:.1f}s for {len(MODELS)} models")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.zeros(len(X), dtype=float)
    for m in MODELS:
        p += m.predict_proba(X)[:, 1]
    return p / len(MODELS)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
