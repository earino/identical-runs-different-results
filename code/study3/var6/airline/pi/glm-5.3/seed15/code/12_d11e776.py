"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
Y_TRAIN = (train[TARGET] == POSITIVE).astype(int)
PRIOR = Y_TRAIN.mean()

# --- feature definitions (fit on TRAIN only; reused for any unseen dataframe) ------------------
RAW_CAT = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in RAW_CAT}
DATE_COLS = ["Month", "DayofMonth", "DayOfWeek"]


def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.slice(start=2), errors="coerce")


def te_map(kind, m):
    if kind == "origin":
        k = train["Origin"].astype(str)
    elif kind == "dest":
        k = train["Dest"].astype(str)
    elif kind == "carrier":
        k = train["UniqueCarrier"].astype(str)
    elif kind == "hour":
        k = (train["DepTime"] // 100).astype(str)
    elif kind == "dow":
        k = _cnum(train["DayOfWeek"]).astype(int).astype(str)
    else:
        raise ValueError(kind)
    g = pd.DataFrame({"k": k, "y": Y_TRAIN.to_numpy()}).groupby("k")["y"].agg(["sum", "count"])
    rate = (g["sum"] + m * PRIOR) / (g["count"] + m)
    return rate.to_dict()


TE_MAPS = {k: te_map(k, 100) for k in ["origin", "dest", "carrier", "hour", "dow"]}


def prepare(df: pd.DataFrame, group: str = "single") -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in DATE_COLS:
        X[c] = _cnum(df[c])
    X["hour"] = df["DepTime"] // 100
    X["minute"] = df["DepTime"] % 100
    X["DepTime"] = df["DepTime"]
    X["Distance"] = df["Distance"]
    for c in RAW_CAT:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    o = df["Origin"].astype(str)
    d = df["Dest"].astype(str)
    carr = df["UniqueCarrier"].astype(str)
    if group in ("single", "time"):
        X["te_origin"] = o.map(TE_MAPS["origin"]).fillna(PRIOR).to_numpy(dtype=float)
        X["te_dest"] = d.map(TE_MAPS["dest"]).fillna(PRIOR).to_numpy(dtype=float)
        X["te_carrier"] = carr.map(TE_MAPS["carrier"]).fillna(PRIOR).to_numpy(dtype=float)
    if group == "time":
        X["te_hour"] = (df["DepTime"] // 100).astype(str).map(TE_MAPS["hour"]).fillna(PRIOR).to_numpy(dtype=float)
        X["te_dow"] = _cnum(df["DayOfWeek"]).astype("Int64").astype(str).map(TE_MAPS["dow"]).fillna(PRIOR).to_numpy(dtype=float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------------------------
BASE = dict(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    learning_rate=0.05,
    min_child_weight=1,
    reg_lambda=1.0,
    nthread=N_JOBS,
)
D6 = dict(max_depth=6, subsample=0.7, colsample_bytree=0.6)
# (name, fe_group, params_over_base, num_rounds, seed)
MEMBERS = [
    ("d6_lr05_s1", "single", D6, 400, 1),
    ("d6_lr05_s2", "single", D6, 400, 2),
    ("d6_lr05_s3", "single", D6, 400, 3),
    ("d6_lr03", "single", {**D6, "learning_rate": 0.03}, 800, 4),
    ("d5_lr05_s5", "single", {**D6, "max_depth": 5}, 400, 5),
    ("d5_lr05_s6", "single", {**D6, "max_depth": 5}, 400, 6),
    ("d7_lr05_s7", "single", {**D6, "max_depth": 7}, 400, 7),
    ("d4_lr05_s8", "single", {**D6, "max_depth": 4, "subsample": 1.0, "colsample_bytree": 1.0}, 400, 8),
    ("d6_lr01_s9", "single", {**D6, "learning_rate": 0.1}, 150, 9),
    ("d6_lr01_s10", "single", {**D6, "learning_rate": 0.1}, 150, 10),
    ("time_lr05_s11", "time", D6, 400, 11),
    ("base_lr05_s12", "base", D6, 400, 12),
]

t0 = time.time()
y_all = to_y(train)
y_ev = to_y(evald)
_dm_cache = {}


def dmatrix(df, group, label=None):
    key = (id(df), group)
    if key not in _dm_cache:
        _dm_cache[key] = xgb.DMatrix(prepare(df, group), label=label, enable_categorical=True)
    return _dm_cache[key]


models = []
preds = []
for name, grp, over, n, seed in MEMBERS:
    b = xgb.train({**BASE, **over, "seed": seed}, dmatrix(train, grp, y_all), num_boost_round=n, verbose_eval=False)
    p = b.predict(dmatrix(evald, grp))
    models.append((b, grp))
    preds.append(p)
    print(f"member={name:14s} eval_auc={roc_auc_score(y_ev, p):.4f}")

print(f"ensemble_mean eval_auc={roc_auc_score(y_ev, np.mean(preds, axis=0)):.4f}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return np.mean([b.predict(xgb.DMatrix(prepare(df, grp), enable_categorical=True)) for b, grp in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
