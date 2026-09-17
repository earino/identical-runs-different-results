"""XGBoost airline delay classifier with engineered features.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
All statistics (frequency maps, target encodings, categorical levels) are fit on data/train.csv only.
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

y_all = (train[TARGET] == POSITIVE).astype(int)
GLOBAL_MEAN = float(y_all.mean())


def parse_base(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["month"] = pd.to_numeric(df["Month"].astype(str).str.slice(2), errors="coerce")
    out["day"] = pd.to_numeric(df["DayofMonth"].astype(str).str.slice(2), errors="coerce")
    out["dow"] = pd.to_numeric(df["DayOfWeek"].astype(str).str.slice(2), errors="coerce")
    out["deptime"] = pd.to_numeric(df["DepTime"], errors="coerce")
    out["distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    out["carrier"] = df["UniqueCarrier"].astype(str)
    out["origin"] = df["Origin"].astype(str)
    out["dest"] = df["Dest"].astype(str)
    out["route"] = out["origin"] + "_" + out["dest"]
    out["hour"] = out["deptime"] // 100
    out["minute"] = out["deptime"] % 100
    out["origin_hour"] = out["origin"] + "|" + out["hour"].astype(str)
    out["dest_hour"] = out["dest"] + "|" + out["hour"].astype(str)
    out["carrier_hour"] = out["carrier"] + "|" + out["hour"].astype(str)
    return out


def smooth_te(keys: pd.Series, values: pd.Series, m: float) -> dict:
    g = pd.DataFrame({"k": keys, "v": values}).groupby("k")["v"].agg(["sum", "count"])
    te = (g["sum"] + m * GLOBAL_MEAN) / (g["count"] + m)
    return te.to_dict()


base_tr = parse_base(train)

FREQ_COLS = ["carrier", "origin", "dest", "route", "hour", "dow", "day", "month"]
freq_maps = {c: base_tr[c].value_counts().to_dict() for c in FREQ_COLS}

TE_SMOOTH = {"carrier": 30, "origin": 20, "dest": 20, "route": 10, "hour": 30,
             "dow": 60, "day": 200, "month": 60,
             "origin_hour": 50, "dest_hour": 50, "carrier_hour": 50}
te_maps = {c: smooth_te(base_tr[c], y_all, m) for c, m in TE_SMOOTH.items()}

CAT_COLS = ["carrier", "origin", "dest", "route"]
cat_levels = {c: pd.Index(pd.unique(base_tr[c])) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here; maps above were fit on train only.
    b = parse_base(df)
    X = pd.DataFrame(index=df.index)
    X["month"] = b["month"]
    X["day"] = b["day"]
    X["dow"] = b["dow"]
    X["deptime"] = b["deptime"]
    X["hour"] = b["hour"]
    X["minute"] = b["minute"]
    tod = (b["hour"] * 60 + b["minute"]) % 1440
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440)
    X["dow_sin"] = np.sin(2 * np.pi * b["dow"] / 7)
    X["dow_cos"] = np.cos(2 * np.pi * b["dow"] / 7)
    X["day_sin"] = np.sin(2 * np.pi * b["day"] / 31)
    X["day_cos"] = np.cos(2 * np.pi * b["day"] / 31)
    X["month_sin"] = np.sin(2 * np.pi * b["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * b["month"] / 12)
    X["distance"] = b["distance"]
    X["log_distance"] = np.log1p(b["distance"])
    for c in FREQ_COLS:
        X["fr_" + c] = b[c].map(freq_maps[c]).astype(float)
    for c in TE_SMOOTH:
        X["te_" + c] = b[c].map(te_maps[c]).astype(float).fillna(GLOBAL_MEAN)
    for c in CAT_COLS:
        X[c] = pd.Categorical(b[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
N_EST = 2000
model = xgb.XGBClassifier(
    n_estimators=N_EST,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
    early_stopping_rounds=50,
)

t0 = time.time()
X_tr, X_va, y_tr, y_va = train_test_split(train, y_all, test_size=0.1, random_state=SEED, stratify=y_all)
model.fit(prepare(X_tr), y_tr, eval_set=[(prepare(X_va), y_va)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter: {getattr(model, 'best_iteration', N_EST)}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
