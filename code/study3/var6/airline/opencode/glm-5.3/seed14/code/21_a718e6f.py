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


def _num(col: pd.Series) -> pd.Series:
    return col.str.split("-").str[1].astype(int)


# --- features -----------------------------------------------------------------
RAW_NUM = ["DepTime", "Distance"]
CYC = {"Month": 12, "DayofMonth": 31, "DayOfWeek": 7}
CAT = ["UniqueCarrier", "Origin", "Dest"]

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT}

# --- label-free count features (hub/route volume, fit on train) ----------------
CNT = {
    "Origin": train["Origin"].value_counts().to_dict(),
    "Dest": train["Dest"].value_counts().to_dict(),
    "Route": (train["Origin"] + "_" + train["Dest"]).value_counts().to_dict(),
    "OriginHour": (train["Origin"] + "_" + (train["DepTime"] // 100).astype(str)).value_counts().to_dict(),
    "DestHour": (train["Dest"] + "_" + (train["DepTime"] // 100).astype(str)).value_counts().to_dict(),
    "CarrierHour": (train["UniqueCarrier"] + "_" + (train["DepTime"] // 100).astype(str)).value_counts().to_dict(),
    "OriginDow": (train["Origin"] + "_" + train["DayOfWeek"]).value_counts().to_dict(),
    "DestDow": (train["Dest"] + "_" + train["DayOfWeek"]).value_counts().to_dict(),
    "CarrierOrigin": (train["UniqueCarrier"] + "_" + train["Origin"]).value_counts().to_dict(),
    "CarrierDest": (train["UniqueCarrier"] + "_" + train["Dest"]).value_counts().to_dict(),
    "CarrierDow": (train["UniqueCarrier"] + "_" + train["DayOfWeek"]).value_counts().to_dict(),
    "OriginB30": (train["Origin"] + "_" + ((train["DepTime"] // 100) * 60 + (train["DepTime"] % 100)).mod(1440).floordiv(30).astype(str)).value_counts().to_dict(),
    "DestB30": (train["Dest"] + "_" + ((train["DepTime"] // 100) * 60 + (train["DepTime"] % 100)).mod(1440).floordiv(30).astype(str)).value_counts().to_dict(),
    "OriginB15": (train["Origin"] + "_" + ((train["DepTime"] // 100) * 60 + (train["DepTime"] % 100)).mod(1440).floordiv(15).astype(str)).value_counts().to_dict(),
    "DestB15": (train["Dest"] + "_" + ((train["DepTime"] // 100) * 60 + (train["DepTime"] % 100)).mod(1440).floordiv(15).astype(str)).value_counts().to_dict(),
    "CarrierB30": (train["UniqueCarrier"] + "_" + ((train["DepTime"] // 100) * 60 + (train["DepTime"] % 100)).mod(1440).floordiv(30).astype(str)).value_counts().to_dict(),
    "CarrierB15": (train["UniqueCarrier"] + "_" + ((train["DepTime"] // 100) * 60 + (train["DepTime"] % 100)).mod(1440).floordiv(15).astype(str)).value_counts().to_dict(),
    "OriginB10": (train["Origin"] + "_" + ((train["DepTime"] // 100) * 60 + (train["DepTime"] % 100)).mod(1440).floordiv(10).astype(str)).value_counts().to_dict(),
    "DestB10": (train["Dest"] + "_" + ((train["DepTime"] // 100) * 60 + (train["DepTime"] % 100)).mod(1440).floordiv(10).astype(str)).value_counts().to_dict(),
}


def _add_cnt(X: pd.DataFrame, name: str, keys: pd.Series) -> None:
    X["Cnt_" + name] = np.log1p(keys.map(CNT[name]).fillna(0.0).to_numpy())


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in RAW_NUM:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    X["Dist_log"] = np.log1p(X["Distance"])
    for c, period in CYC.items():
        n = _num(df[c])
        X[c + "_n"] = n
        X[c + "_sin"] = np.sin(2 * np.pi * n / period)
        X[c + "_cos"] = np.cos(2 * np.pi * n / period)
    dt = X["DepTime"]
    hour = dt // 100
    minute = dt % 100
    tod = hour * 60 + minute
    X["Hour"] = hour
    X["Tod_sin"] = np.sin(2 * np.pi * tod / 1440)
    X["Tod_cos"] = np.cos(2 * np.pi * tod / 1440)
    for c in CAT:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    _add_cnt(X, "Origin", df["Origin"])
    _add_cnt(X, "Dest", df["Dest"])
    _add_cnt(X, "Route", df["Origin"] + "_" + df["Dest"])
    hour_str = (X["DepTime"] // 100).astype(str)
    _add_cnt(X, "OriginHour", df["Origin"] + "_" + hour_str)
    _add_cnt(X, "DestHour", df["Dest"] + "_" + hour_str)
    _add_cnt(X, "CarrierHour", df["UniqueCarrier"] + "_" + hour_str)
    _add_cnt(X, "OriginDow", df["Origin"] + "_" + df["DayOfWeek"])
    _add_cnt(X, "DestDow", df["Dest"] + "_" + df["DayOfWeek"])
    _add_cnt(X, "CarrierOrigin", df["UniqueCarrier"] + "_" + df["Origin"])
    _add_cnt(X, "CarrierDest", df["UniqueCarrier"] + "_" + df["Dest"])
    _add_cnt(X, "CarrierDow", df["UniqueCarrier"] + "_" + df["DayOfWeek"])
    tod = ((X["DepTime"] // 100) * 60 + (X["DepTime"] % 100)).mod(1440)
    b30 = tod.floordiv(30).astype(str)
    b15 = tod.floordiv(15).astype(str)
    _add_cnt(X, "OriginB30", df["Origin"] + "_" + b30)
    _add_cnt(X, "DestB30", df["Dest"] + "_" + b30)
    _add_cnt(X, "OriginB15", df["Origin"] + "_" + b15)
    _add_cnt(X, "DestB15", df["Dest"] + "_" + b15)
    _add_cnt(X, "CarrierB30", df["UniqueCarrier"] + "_" + b30)
    _add_cnt(X, "CarrierB15", df["UniqueCarrier"] + "_" + b15)
    b10 = tod.floordiv(10).astype(str)
    _add_cnt(X, "OriginB10", df["Origin"] + "_" + b10)
    _add_cnt(X, "DestB10", df["Dest"] + "_" + b10)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: bagged ensemble of XGB models --------------------------------------
N_MODELS = 32
SUB = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85] * 2
DEP = [6, 7, 8, 9] * 6
MCW = [1, 3, 5] * 8
LAM = ([1, 5] * 12)
CS = [0.7, 0.8, 0.9] * 8
models = [
    xgb.XGBClassifier(
        n_estimators=220,
        max_depth=DEP[s % len(DEP)],
        learning_rate=0.07,
        subsample=SUB[s % len(SUB)],
        colsample_bytree=CS[s % len(CS)],
        min_child_weight=MCW[s % len(MCW)],
        reg_lambda=LAM[s % len(LAM)],
        tree_method="hist",
        enable_categorical=True,
        random_state=s,
        n_jobs=N_JOBS,
    )
    for s in range(N_MODELS)
]

SEED_SHIFT = 100
for i, s in enumerate(range(SEED_SHIFT, SEED_SHIFT + N_MODELS)):
    models[i].random_state = s

t0 = time.time()
X_tr = prepare(train)
y_tr = to_y(train)
for m in models:
    m.fit(X_tr, y_tr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
