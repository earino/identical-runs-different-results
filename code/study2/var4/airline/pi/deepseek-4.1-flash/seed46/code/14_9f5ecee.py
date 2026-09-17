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
from sklearn.model_selection import train_test_split

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
obj_cols = [c for c in RAW_COLS if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
RAW_COLS = [c for c in RAW_COLS if c not in obj_cols or c in cat_cols]
TIME_COLS = ["dep_hour", "dep_min", "dep_minutes"]
freq_extra = ["origin_hour_freq", "dest_hour_freq", "carrier_hour_freq",
              "origin_hour_frac", "dest_hour_frac", "carrier_hour_frac",
              "route_freq", "origin_dow_hour_freq", "dest_dow_hour_freq",
              "origin_30_freq", "dest_30_freq", "carrier_30_freq", "route_hour_freq",
              "origin_15_freq", "dest_15_freq", "carrier_origin_hour_freq",
              "origin_month_freq", "origin_dow_freq",
              "origin_max_hour", "dest_max_hour", "origin_hour_peak_ratio", "dest_hour_peak_ratio",
              "origin_30_win3", "dest_30_win3", "origin_15_win3", "dest_15_win3",
              "carrier_origin_freq", "carrier_dest_freq",
              "carrier_origin_share", "carrier_dest_share"]
FREQ_COLS = ["origin_freq", "dest_freq", "carrier_freq"] + freq_extra
feature_cols = RAW_COLS + TIME_COLS + FREQ_COLS
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
# frequency (airport/carrier size + congestion) encodings, fit on TRAIN ONLY
_freq = {c: train[c].value_counts() for c in ["Origin", "Dest", "UniqueCarrier"]}
_dep = pd.to_numeric(train["DepTime"], errors="coerce")
_dpm = (_dep // 100).clip(0, 23) * 60 + _dep % 100
_h = ("h" + (_dep // 100).clip(0, 23).astype(str))
_m30 = ("m" + _dpm.floordiv(30).astype(str))
_m15 = ("q" + _dpm.floordiv(15).astype(str))
_freq["origin_hour"] = (train["Origin"] + "_" + _h).value_counts()
_freq["dest_hour"] = (train["Dest"] + "_" + _h).value_counts()
_freq["carrier_hour"] = (train["UniqueCarrier"] + "_" + _h).value_counts()
_freq["carrier_origin"] = (train["UniqueCarrier"] + "_" + train["Origin"]).value_counts()
_freq["carrier_dest"] = (train["UniqueCarrier"] + "_" + train["Dest"]).value_counts()
_freq["carrier_origin_hour"] = (train["UniqueCarrier"] + "_" + train["Origin"] + "_" + _h).value_counts()
_freq["route"] = (train["Origin"] + "_" + train["Dest"]).value_counts()
_freq["route_hour"] = (train["Origin"] + "_" + train["Dest"] + "_" + _h).value_counts()
_freq["origin_dow_hour"] = (train["Origin"] + "_" + train["DayOfWeek"] + "_" + _h).value_counts()
_freq["dest_dow_hour"] = (train["Dest"] + "_" + train["DayOfWeek"] + "_" + _h).value_counts()
_freq["origin_30"] = (train["Origin"] + "_" + _m30).value_counts()
_freq["dest_30"] = (train["Dest"] + "_" + _m30).value_counts()
_freq["carrier_30"] = (train["UniqueCarrier"] + "_" + _m30).value_counts()
_freq["origin_15"] = (train["Origin"] + "_" + _m15).value_counts()
_freq["dest_15"] = (train["Dest"] + "_" + _m15).value_counts()
_freq["origin_month"] = (train["Origin"] + "_" + train["Month"]).value_counts()
_freq["origin_dow"] = (train["Origin"] + "_" + train["DayOfWeek"]).value_counts()
# peak-hour volume and current-hour share of that peak (scale-free congestion)
_freq["origin_max_hour"] = train.groupby(["Origin", _h]).size().groupby(level=0).max()
_freq["dest_max_hour"] = train.groupby(["Dest", _h]).size().groupby(level=0).max()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[RAW_COLS].copy()
    dep = pd.to_numeric(X["DepTime"], errors="coerce")
    X["dep_hour"] = (dep // 100).clip(0, 23)
    X["dep_min"] = dep % 100
    X["dep_minutes"] = X["dep_hour"] * 60 + X["dep_min"]
    X["origin_freq"] = df["Origin"].map(_freq["Origin"]).fillna(0)
    X["dest_freq"] = df["Dest"].map(_freq["Dest"]).fillna(0)
    X["carrier_freq"] = df["UniqueCarrier"].map(_freq["UniqueCarrier"]).fillna(0)
    hh = "h" + X["dep_hour"].astype(int).astype(str)
    m30 = "m" + X["dep_minutes"].floordiv(30).astype(int).astype(str)
    m15 = "q" + X["dep_minutes"].floordiv(15).astype(int).astype(str)
    X["origin_hour_freq"] = (df["Origin"] + "_" + hh).map(_freq["origin_hour"]).fillna(0)
    X["dest_hour_freq"] = (df["Dest"] + "_" + hh).map(_freq["dest_hour"]).fillna(0)
    X["carrier_hour_freq"] = (df["UniqueCarrier"] + "_" + hh).map(_freq["carrier_hour"]).fillna(0)
    X["carrier_origin_hour_freq"] = (df["UniqueCarrier"] + "_" + df["Origin"] + "_" + hh).map(
        _freq["carrier_origin_hour"]).fillna(0)
    X["origin_hour_frac"] = X["origin_hour_freq"] / X["origin_freq"].replace(0, np.nan)
    X["dest_hour_frac"] = X["dest_hour_freq"] / X["dest_freq"].replace(0, np.nan)
    X["carrier_hour_frac"] = X["carrier_hour_freq"] / X["carrier_freq"].replace(0, np.nan)
    X["route_freq"] = (df["Origin"] + "_" + df["Dest"]).map(_freq["route"]).fillna(0)
    X["route_hour_freq"] = (df["Origin"] + "_" + df["Dest"] + "_" + hh).map(_freq["route_hour"]).fillna(0)
    X["origin_dow_hour_freq"] = (df["Origin"] + "_" + df["DayOfWeek"] + "_" + hh).map(
        _freq["origin_dow_hour"]).fillna(0)
    X["dest_dow_hour_freq"] = (df["Dest"] + "_" + df["DayOfWeek"] + "_" + hh).map(
        _freq["dest_dow_hour"]).fillna(0)
    X["origin_30_freq"] = (df["Origin"] + "_" + m30).map(_freq["origin_30"]).fillna(0)
    X["dest_30_freq"] = (df["Dest"] + "_" + m30).map(_freq["dest_30"]).fillna(0)
    X["carrier_30_freq"] = (df["UniqueCarrier"] + "_" + m30).map(_freq["carrier_30"]).fillna(0)
    X["origin_15_freq"] = (df["Origin"] + "_" + m15).map(_freq["origin_15"]).fillna(0)
    X["dest_15_freq"] = (df["Dest"] + "_" + m15).map(_freq["dest_15"]).fillna(0)
    X["origin_month_freq"] = (df["Origin"] + "_" + df["Month"]).map(_freq["origin_month"]).fillna(0)
    X["origin_dow_freq"] = (df["Origin"] + "_" + df["DayOfWeek"]).map(_freq["origin_dow"]).fillna(0)
    X["origin_max_hour"] = df["Origin"].map(_freq["origin_max_hour"]).fillna(0)
    X["dest_max_hour"] = df["Dest"].map(_freq["dest_max_hour"]).fillna(0)
    X["origin_hour_peak_ratio"] = X["origin_hour_freq"] / X["origin_max_hour"].replace(0, np.nan)
    X["dest_hour_peak_ratio"] = X["dest_hour_freq"] / X["dest_max_hour"].replace(0, np.nan)
    # smoothed congestion: current + adjacent time bins
    b30 = X["dep_minutes"].floordiv(30).astype(int)
    b15 = X["dep_minutes"].floordiv(15).astype(int)

    def _win3(place, tag, freq_key, b, n):
        acc = np.zeros(len(X))
        for d in (-1, 0, 1):
            bb = (b + d).clip(0, n - 1).astype(int).astype(str)
            acc += (df[place] + "_" + tag + bb).map(_freq[freq_key]).fillna(0).to_numpy()
        return acc

    X["origin_30_win3"] = _win3("Origin", "m", "origin_30", b30, 48)
    X["dest_30_win3"] = _win3("Dest", "m", "dest_30", b30, 48)
    X["origin_15_win3"] = _win3("Origin", "q", "origin_15", b15, 96)
    X["dest_15_win3"] = _win3("Dest", "q", "dest_15", b15, 96)
    X["carrier_origin_freq"] = (df["UniqueCarrier"] + "_" + df["Origin"]).map(_freq["carrier_origin"]).fillna(0)
    X["carrier_dest_freq"] = (df["UniqueCarrier"] + "_" + df["Dest"]).map(_freq["carrier_dest"]).fillna(0)
    X["carrier_origin_share"] = X["carrier_origin_freq"] / X["origin_freq"].replace(0, np.nan)
    X["carrier_dest_share"] = X["carrier_dest_freq"] / X["dest_freq"].replace(0, np.nan)
    X = X[feature_cols]
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Diverse XGBoost members: different depths / feature subsampling / seeds. Each picks its own
# number of trees on an internal validation split (never eval.csv), then refits on all of train.
BASE = dict(
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
MEMBERS = [
    dict(max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
    dict(max_depth=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
    dict(max_depth=7, subsample=0.7, colsample_bytree=0.7, reg_lambda=2.0, min_child_weight=3),
    dict(max_depth=6, subsample=0.9, colsample_bytree=0.6, reg_lambda=1.0),
    dict(max_depth=6, subsample=0.7, colsample_bytree=0.9, reg_lambda=0.5),
    dict(max_depth=8, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, min_child_weight=5,
         learning_rate=0.03),
    dict(max_depth=4, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0),
    dict(max_depth=6, subsample=1.0, colsample_bytree=0.5, reg_lambda=2.0, learning_rate=0.04),
    dict(max_depth=7, subsample=0.6, colsample_bytree=1.0, reg_lambda=1.0, min_child_weight=2,
         learning_rate=0.04),
    dict(max_depth=6, subsample=0.85, colsample_bytree=0.75, reg_lambda=1.5, max_cat_to_onehot=1),
    dict(max_depth=5, subsample=0.9, colsample_bytree=0.6, reg_lambda=2.0),
    dict(max_depth=9, subsample=0.8, colsample_bytree=0.7, reg_lambda=4.0, min_child_weight=10,
         learning_rate=0.03),
    dict(max_depth=6, subsample=0.5, colsample_bytree=0.8, reg_lambda=1.0, learning_rate=0.06),
    dict(max_depth=7, subsample=0.9, colsample_bytree=0.5, reg_lambda=1.5, min_child_weight=4),
]


def _fit_member(params: dict, seed: int, n_rounds: int) -> xgb.XGBClassifier:
    p = dict(BASE, random_state=seed, **{k: v for k, v in params.items() if k != "seed"})
    m = xgb.XGBClassifier(n_estimators=n_rounds, **p)
    m.fit(X_all, y_all)
    return m


X_all, y_all = prepare(train), to_y(train)
# One early-stopping probe fixes the tree budget for every member (fast + consistent).
_p = dict(BASE, random_state=SEED, **MEMBERS[0])
X_tr, X_va, y_tr, y_va = train_test_split(X_all, y_all, test_size=0.15, random_state=SEED, stratify=y_all)
probe = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=50, **_p)
probe.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
N_ROUNDS = int((int(probe.best_iteration) + 1) * 1.15) + 1
print(f"Probe best_iteration={probe.best_iteration} -> n_rounds={N_ROUNDS}")

models = []
t0 = time.time()
for i, mp in enumerate(MEMBERS):
    models.append(_fit_member(mp, seed=SEED + i, n_rounds=N_ROUNDS))
print(f"Training {len(models)} members: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
