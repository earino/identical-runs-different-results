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
OBJ_COLS = [c for c in RAW_COLS if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in OBJ_COLS if train[c].nunique() <= 1000]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
NUM_COLS = [c for c in RAW_COLS if c not in OBJ_COLS]

# c-<n> encoded ordinal columns (Month, DayofMonth, DayOfWeek)
C_NUM = {c: c for c in cat_cols if train[c].str.startswith("c-", na=False).all()}


def _c_to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in NUM_COLS:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cat_cols:
        if c in C_NUM:
            v = _c_to_int(df[c]).astype("float64")
            if c == "Month":
                X["month"] = v
                X["month_sin"] = np.sin(2 * np.pi * v / 12.0)
                X["month_cos"] = np.cos(2 * np.pi * v / 12.0)
            elif c == "DayOfWeek":
                X["dow"] = v
                X["dow_sin"] = np.sin(2 * np.pi * v / 7.0)
                X["dow_cos"] = np.cos(2 * np.pi * v / 7.0)
            elif c == "DayofMonth":
                X["dom"] = v
                X["dom_sin"] = np.sin(2 * np.pi * v / 31.0)
                X["dom_cos"] = np.cos(2 * np.pi * v / 31.0)
        else:
            X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    # scheduled departure: hhmm -> minutes since midnight (+ cyclical position in the day)
    hh = np.floor(X["DepTime"] / 100.0)
    mm = X["DepTime"] - hh * 100.0
    dep = np.where((hh >= 0) & (hh <= 23) & (mm >= 0) & (mm <= 59), hh * 60.0 + mm, np.nan)
    X["dep_min"] = dep
    X["dep_hour"] = dep / 60.0
    X["dep_hour_sin"] = np.sin(2 * np.pi * dep / 1440.0)
    X["dep_hour_cos"] = np.cos(2 * np.pi * dep / 1440.0)
    X = X.drop(columns=["DepTime"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- target / frequency encodings (fit on TRAIN labels and keys only) ----------
Y = to_y(train)
PRIOR = Y.mean()
TE_SPEC = {"UniqueCarrier": 20.0, "Origin": 20.0, "Dest": 20.0, "route": 10.0}
TE_TRAIN_COLS = list(TE_SPEC)

_keys = pd.DataFrame(index=train.index)
_keys["UniqueCarrier"] = train["UniqueCarrier"].astype(str)
_keys["Origin"] = train["Origin"].astype(str)
_keys["Dest"] = train["Dest"].astype(str)
_keys["route"] = _keys["Origin"] + "_" + _keys["Dest"]


def _smooth_map(keys: pd.Series, y: np.ndarray, k: float) -> pd.Series:
    g = pd.DataFrame({"k": keys.values, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + PRIOR * k) / (g["count"] + k)


TE_MAP = {c: _smooth_map(_keys[c], Y, TE_SPEC[c]) for c in TE_TRAIN_COLS}
FREQ_MAP = {c: _keys[c].value_counts() for c in ("UniqueCarrier", "Origin", "Dest", "route")}

# out-of-fold encodings for the training rows, so the model does not see its own labels
from sklearn.model_selection import KFold

OOF = pd.DataFrame(index=train.index, columns=TE_TRAIN_COLS, dtype="float64")
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for tr_i, va_i in kf.split(train):
    for c in TE_TRAIN_COLS:
        m = _smooth_map(_keys[c].iloc[tr_i], Y[tr_i], TE_SPEC[c])
        OOF.loc[OOF.index[va_i], c] = _keys[c].iloc[va_i].map(m).to_numpy()


def _encode_keys(df: pd.DataFrame):
    key = pd.DataFrame(index=df.index)
    key["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    key["Origin"] = df["Origin"].astype(str)
    key["Dest"] = df["Dest"].astype(str)
    key["route"] = key["Origin"] + "_" + key["Dest"]
    return key


def _add_te(X: pd.DataFrame, key: pd.DataFrame, oof: bool = False) -> pd.DataFrame:
    for c in TE_TRAIN_COLS:
        if oof:
            X[f"te_{c}"] = OOF[c].reindex(X.index).to_numpy()
        else:
            X[f"te_{c}"] = key[c].map(TE_MAP[c]).to_numpy(dtype="float64")
    for c, fm in FREQ_MAP.items():
        X[f"freq_{c}"] = key[c].map(fm).to_numpy(dtype="float64")
    return X


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
SEEDS = [42, 7, 202, 1234, 99]
Xtr = _add_te(prepare(train), _encode_keys(train), oof=True)
ytr = to_y(train)

t0 = time.time()
models = []
for s in SEEDS:
    m = xgb.XGBClassifier(**PARAMS, random_state=s)
    m.fit(Xtr, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  models={len(models)}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = _add_te(prepare(df), _encode_keys(df), oof=False)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
