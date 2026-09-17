"""Airline delay XGBoost — bagged ensemble + OOF target encodings.

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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# --- encodings (fit on TRAIN only) ----------------------------------------------
y_tr = (train[TARGET] == POSITIVE).astype(float)
PRIOR = float(y_tr.mean())


def smoothed_te(by: pd.Series, m: float = 20) -> pd.Series:
    stats = y_tr.groupby(by).agg(["mean", "count"])
    return (stats["mean"] * stats["count"] + PRIOR * m) / (stats["count"] + m)


KEY_TIME = train["DepTime"] // 100 * 2  # 30-minute bins
KEY_TIME10 = train["DepTime"] // 100 * 6 + train["DepTime"] % 100 // 10  # 10-minute bins
TE_ORIGIN = smoothed_te(train["Origin"])
TE_H30 = smoothed_te(KEY_TIME)
TE_H10 = smoothed_te(KEY_TIME10)
CNT_O = train["Origin"].value_counts()
CNT_D = train["Dest"].value_counts()
CNT_OH = (train["Origin"].astype(str) + "_" + (train["DepTime"] // 100).astype(str)).value_counts()
# congestion: how many flights this origin saw in the same 30-min window / same dow
CNT_OH30 = (train["Origin"].astype(str) + "_" + ((train["DepTime"] % 2400) // 30).astype(str)).value_counts()
CNT_ODOW = (train["Origin"].astype(str) + "_" + train["DayOfWeek"].astype(str)).value_counts()
CNT_CH30 = (train["UniqueCarrier"].astype(str) + "_" + ((train["DepTime"] % 2400) // 30).astype(str)).value_counts()
CNT_OC = (train["Origin"].astype(str) + "_" + train["UniqueCarrier"].astype(str)).value_counts()

# OOF versions of the TEs for training rows (no self-leak in fit data).
# At predict time unseen rows use the full-train TE map (they are out-of-fold by definition).
OOF_TE_O = np.full(len(train), np.nan)
OOF_TE_TIME = np.full(len(train), np.nan)
OOF_TE_TIME10 = np.full(len(train), np.nan)
kf = KFold(5, shuffle=True, random_state=0)
for tr_idx, va_idx in kf.split(train):
    y_fold = y_tr.iloc[tr_idx]

    def te_fold(by: pd.Series, m: float = 20) -> pd.Series:
        stats = y_fold.groupby(by).agg(["mean", "count"])
        return (stats["mean"] * stats["count"] + PRIOR * m) / (stats["count"] + m)

    OOF_TE_O[va_idx] = train["Origin"].iloc[va_idx].map(te_fold(train["Origin"].iloc[tr_idx])).values
    OOF_TE_TIME[va_idx] = KEY_TIME.iloc[va_idx].map(te_fold(KEY_TIME.iloc[tr_idx])).values
    OOF_TE_TIME10[va_idx] = KEY_TIME10.iloc[va_idx].map(te_fold(KEY_TIME10.iloc[tr_idx])).values


def prepare(df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
    # ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows.
    X = df[feature_cols].copy()
    t = df["DepTime"].astype(int)
    mins = (t % 2400 // 100) * 60 + (t % 100)  # minutes since midnight
    X["mins"] = mins
    X["te_O"] = df["Origin"].map(TE_ORIGIN).astype(float).values
    X["hour_x_dist"] = (t // 100) * df["Distance"].astype(float)
    X["log_dist"] = np.log1p(df["Distance"].astype(float))
    X["sin_m"] = np.sin(2 * np.pi * mins / 1440.0)
    X["cos_m"] = np.cos(2 * np.pi * mins / 1440.0)
    # rich periodic bases: hour-of-day harmonics k=2..24, day-of-year harmonics k=1..4, dow harmonics k=1..3
    h = mins / 60.0
    for k in range(2, 25):
        X[f"sin{k}"] = np.sin(2 * np.pi * k * h / 24.0)
        X[f"cos{k}"] = np.cos(2 * np.pi * k * h / 24.0)
    mon = pd.to_numeric(df["Month"].astype("string").str.replace("c-", "", regex=False), errors="coerce").astype(float)
    dom = pd.to_numeric(df["DayofMonth"].astype("string").str.replace("c-", "", regex=False), errors="coerce").astype(float)
    doy = (mon - 1) * 31 + dom
    for k in range(1, 5):
        X[f"dsin{k}"] = np.sin(2 * np.pi * k * doy / 365.0)
        X[f"dcos{k}"] = np.cos(2 * np.pi * k * doy / 365.0)
    dow = pd.to_numeric(df["DayOfWeek"].astype("string").str.replace("c-", "", regex=False), errors="coerce").astype(float)
    for k in range(1, 4):
        X[f"wsin{k}"] = np.sin(2 * np.pi * k * dow / 7.0)
        X[f"wcos{k}"] = np.cos(2 * np.pi * k * dow / 7.0)
    X["te_time"] = (t // 100 * 2).map(TE_H30).astype(float).values
    X["te_time10"] = (t // 100 * 6 + t % 100 // 10).map(TE_H10).astype(float).values
    X["cnt_o"] = df["Origin"].map(CNT_O).astype(float).values
    X["cnt_d"] = df["Dest"].map(CNT_D).astype(float).values
    X["cnt_oh"] = (df["Origin"].astype(str) + "_" + (t // 100).astype(str)).map(CNT_OH).astype(float).values
    X["cnt_oh30"] = (df["Origin"].astype(str) + "_" + ((t % 2400) // 30).astype(str)).map(CNT_OH30).astype(float).values
    X["cnt_odow"] = (df["Origin"].astype(str) + "_" + df["DayOfWeek"].astype(str)).map(CNT_ODOW).astype(float).values
    X["cnt_ch30"] = (df["UniqueCarrier"].astype(str) + "_" + ((t % 2400) // 30).astype(str)).map(CNT_CH30).astype(float).values
    X["cnt_oc"] = (df["Origin"].astype(str) + "_" + df["UniqueCarrier"].astype(str)).map(CNT_OC).astype(float).values
    if is_train:
        X["te_O"] = OOF_TE_O
        X["te_time"] = OOF_TE_TIME
        X["te_time10"] = OOF_TE_TIME10
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- bagged ensemble -------------------------------------------------------------
N_MODELS = 16
BASE = dict(
    n_estimators=100,
    max_depth=14,
    min_child_weight=50,
    learning_rate=0.1,
    subsample=0.9,
    # colsample_bytree is set per model for feature-bag diversity
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_train_fit = prepare(train, is_train=True)
y_train = to_y(train)
models = []
for i in range(N_MODELS):
    cs = [0.5, 0.65, 0.8, 1.0][i % 4]  # feature-bag diversity
    m = xgb.XGBClassifier(random_state=100 + i, colsample_bytree=cs, **BASE)
    m.fit(X_train_fit, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)  # unseen rows: full-train TE maps
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
