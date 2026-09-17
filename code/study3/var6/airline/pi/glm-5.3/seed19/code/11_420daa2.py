"""XGBoost binary classifier for airline departure delay (ensemble).

Families averaged with equal weights:
  - 3x "te" models (seeds): numeric time features + smoothed target encodings
    (TE) of categorical structures (Origin/Dest/Carrier/Route x hour, dow)
    + traffic-count features. TE maps fit on train only; train rows get
    out-of-fold (OOF) TE values.
  - 2x "raw" models: native categoricals (carrier/origin/dest/dow) + numeric
    features, depths 8 and 6.
Month/DayofMonth features are deliberately excluded: 2005 calendar noise that
does not generalize. Tree counts come from mean validation curves on train.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. Module-level `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives in prepare()/prepare_raw(), using only train-fit statistics.
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
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

K_SMOOTH = 30.0   # target-encoding smoothing count
N_FOLDS = 10      # OOF folds for TE on train rows
TE_SEEDS = [42, 1, 2]

TE_PARAMS = dict(
    max_depth=8,
    learning_rate=0.05,
    min_child_weight=5,
    reg_lambda=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=SEED,
    n_jobs=N_JOBS,
)
RAW_PARAMS = dict(
    max_depth=8,
    learning_rate=0.05,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
y_all = (train[TARGET] == POSITIVE).astype(int).to_numpy()
GM = float(y_all.mean())


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- feature engineering -------------------------------------------------------
def te_struct(df: pd.DataFrame) -> pd.DataFrame:
    """Categorical structures to target-encode (month/day-of-month excluded: 2005-only noise)."""
    d = pd.DataFrame(index=df.index)
    d["Origin"] = df["Origin"].astype(str)
    d["Dest"] = df["Dest"].astype(str)
    d["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    d["hour"] = (df["DepTime"].astype(int) // 100).astype(str)
    d["dow"] = df["DayOfWeek"].astype(str)
    d["Route"] = d["Origin"] + "_" + d["Dest"]
    d["ORoute"] = d["Origin"] + "_" + d["hour"]
    d["DRoute"] = d["Dest"] + "_" + d["hour"]
    d["CHour"] = d["UniqueCarrier"] + "_" + d["hour"]
    d["RHour"] = d["Route"] + "_" + d["hour"]
    d["OCarrier"] = d["Origin"] + "_" + d["UniqueCarrier"]
    d["OHourBin"] = d["Origin"] + "_" + pd.cut(df["DepTime"].astype(int) // 100, [-1, 6, 12, 18, 26]).astype(str)
    d["Min10"] = ((df["DepTime"].astype(int) % 100) // 10).astype(str)  # within-hour block (replicates year-over-year)
    d["Min5"] = ((df["DepTime"].astype(int) % 100) // 5).astype(str)
    return d


def te_maps(d: pd.DataFrame, y: np.ndarray) -> dict:
    """Smoothed target means per level; fit on the given (train) rows only."""
    y = pd.Series(np.asarray(y, dtype=float), index=d.index)
    maps = {}
    for c in d.columns:
        g = y.groupby(d[c]).agg(["mean", "count"])
        maps[c] = (g["count"] * g["mean"] + K_SMOOTH * GM) / (g["count"] + K_SMOOTH)
    return maps


def num_feats(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(int)
    tod = dep // 100 * 60 + dep % 100
    X["DepTime"] = dep
    X["tod"] = tod
    X["sin_tod"] = np.sin(2 * np.pi * tod / 1440.0)
    X["cos_tod"] = np.cos(2 * np.pi * tod / 1440.0)
    dist = df["Distance"].astype(float)
    X["Distance"] = dist
    X["log_dist"] = np.log1p(dist)
    return X


_STR_TRAIN = te_struct(train)
MAPS_FULL = te_maps(_STR_TRAIN, y_all)  # train-only fit
CNT_MAPS = {
    "Origin": _STR_TRAIN["Origin"].value_counts(),
    "Route": _STR_TRAIN["Route"].value_counts(),
    "OriginHour": (_STR_TRAIN["Origin"] + "_" + _STR_TRAIN["hour"]).value_counts(),
    "DestHour": (_STR_TRAIN["Dest"] + "_" + _STR_TRAIN["hour"]).value_counts(),
}
RAW_CATS = ["UniqueCarrier", "Origin", "Dest", "DayOfWeek"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in RAW_CATS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataframe -> TE-model feature matrix. Train-fit statistics only."""
    X = num_feats(df)
    d = te_struct(df)
    for c in d.columns:
        X["te_" + c] = d[c].map(MAPS_FULL[c]).fillna(GM).to_numpy()
    for k, s in {
        "Origin": d["Origin"],
        "Route": d["Route"],
        "OriginHour": d["Origin"] + "_" + d["hour"],
        "DestHour": d["Dest"] + "_" + d["hour"],
    }.items():
        X["cnt_" + k] = np.log1p(s.map(CNT_MAPS[k]).fillna(0).astype(float).to_numpy())
    return X


def prepare_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataframe -> rawcat-model feature matrix (native categoricals)."""
    X = pd.DataFrame(index=df.index)
    for c in RAW_CATS:
        X[c] = pd.Categorical(df[c].astype(str), categories=CAT_LEVELS[c])
    return pd.concat([X, num_feats(df)], axis=1)


# --- OOF target encoding of train rows ----------------------------------------
CNT_TRAIN = pd.DataFrame(
    {
        "cnt_Origin": np.log1p(_STR_TRAIN["Origin"].map(CNT_MAPS["Origin"]).fillna(0).astype(float).to_numpy()),
        "cnt_Route": np.log1p(_STR_TRAIN["Route"].map(CNT_MAPS["Route"]).fillna(0).astype(float).to_numpy()),
        "cnt_OriginHour": np.log1p((_STR_TRAIN["Origin"] + "_" + _STR_TRAIN["hour"]).map(CNT_MAPS["OriginHour"]).fillna(0).astype(float).to_numpy()),
        "cnt_DestHour": np.log1p((_STR_TRAIN["Dest"] + "_" + _STR_TRAIN["hour"]).map(CNT_MAPS["DestHour"]).fillna(0).astype(float).to_numpy()),
    },
    index=train.index,
)
oof_te = pd.DataFrame(index=train.index, columns=_STR_TRAIN.columns, dtype=float)
folds = list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(train))
for i_tr, i_va in folds:
    m = te_maps(_STR_TRAIN.iloc[i_tr], y_all[i_tr])
    dv = _STR_TRAIN.iloc[i_va]
    for c in oof_te.columns:
        oof_te.iloc[i_va, oof_te.columns.get_loc(c)] = dv[c].map(m[c]).fillna(GM)

X_train_all = pd.concat([num_feats(train), oof_te.rename(columns=lambda c: "te_" + c), CNT_TRAIN], axis=1)
X_train_raw = prepare_raw(train)

# --- tree counts: fixed at the plateau found by train-only CV curves (stable across runs) ---
t0 = time.time()
n_te = 800
n_raw = 800
n_raw_d6 = 800
print(f"tree counts: te={n_te}, raw={n_raw}, raw_d6={n_raw_d6}")

# --- fit final ensemble --------------------------------------------------------
members = []
for s in TE_SEEDS:
    m = xgb.XGBClassifier(n_estimators=n_te, **{**TE_PARAMS, "random_state": s})
    m.fit(X_train_all, y_all)
    members.append(("te", m))
for params in (RAW_PARAMS, {**RAW_PARAMS, "max_depth": 6}):
    m = xgb.XGBClassifier(n_estimators=n_raw if params["max_depth"] == 8 else n_raw_d6, **params)
    m.fit(X_train_raw, y_all)
    members.append(("raw", m))
print(f"Training time: {time.time() - t0:.1f}s ({len(members)} members)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    ps = []
    for kind, m in members:
        X = prepare(df) if kind == "te" else prepare_raw(df)
        ps.append(m.predict_proba(X)[:, 1])
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
