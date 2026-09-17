"""XGBoost binary classifier for the airline-delay task (see program.md for the contract).

  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     Every feature computation lives in `prepare()` and depends only on constants fitted from data/train.csv.
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


# --- features -----------------------------------------------------------------
# Calendar identity (Month, DayofMonth) carries no transferable signal across the 2005->2006 gap: the
# per-month delay rate is flat in eval, so those columns only give the trees something to overfit.
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

# --- frequency (traffic-volume) encodings, fitted on training rows only --------
# How busy an airport / route / carrier is transfers across years far better than the target rate does.
COUNT_SPECS = [["Origin"], ["Dest"], ["UniqueCarrier"], ["Origin", "Dest"]]


def _key(df, keyspec):
    if len(keyspec) == 1:
        return df[keyspec[0]].astype(str)
    out = df[keyspec[0]].astype(str)
    for k in keyspec[1:]:
        out = out + "_" + df[k].astype(str)
    return out


def _fit_counts(keyspec):
    vc = _key(train, keyspec).value_counts()
    return {"name": "cnt_" + "_".join(keyspec), "keyspec": keyspec, "map": vc.to_dict(), "default": 1.0}


COUNT_FITS = [_fit_counts(sp) for sp in COUNT_SPECS]
COUNT_COLS = [f["name"] for f in COUNT_FITS]

FEATURE_COLS = RAW_FEATS + CLOCK_COLS + COUNT_COLS


def _clock(df: pd.DataFrame) -> pd.DataFrame:
    """DepTime is hhmm as an integer and contains junk above 2400 / with minutes >= 60."""
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
    return X[FEATURE_COLS]


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=20,
    reg_lambda=5.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
