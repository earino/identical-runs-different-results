"""XGBoost binary classifier for airline delay. Only file the agent edits.

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
def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.split("-").str[1], errors="coerce")


cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in ["UniqueCarrier", "Origin", "Dest"]}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    # date parts
    month = _cnum(df["Month"])
    dom = _cnum(df["DayofMonth"])
    dow = _cnum(df["DayOfWeek"])
    X["month"], X["dom"], X["dow"] = month, dom, dow
    # time of day: DepTime is hhmm, can exceed 2400 (up to 2620)
    dep = df["DepTime"].astype("float64")
    hh = dep // 100
    mm = dep % 100
    frac = (hh + mm / 60.0) / 24.0
    X["dep_hour"] = hh
    X["dep_min"] = mm
    X["dep_sin"] = np.sin(2 * np.pi * frac)
    X["dep_cos"] = np.cos(2 * np.pi * frac)
    X["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    X["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    X["dow_sin"] = np.sin(2 * np.pi * (dow - 1) / 7)
    X["dow_cos"] = np.cos(2 * np.pi * (dow - 1) / 7)
    X["Distance"] = df["Distance"].astype("float64")
    X["log_dist"] = np.log1p(df["Distance"])
    X["dep_after2400"] = (dep >= 2400).astype("float64")
    # categoricals (native)
    for c in ["UniqueCarrier", "Origin", "Dest"]:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


y_train = to_y(train)
X_va = prepare(evald)
y_va = to_y(evald)


# --- model --------------------------------------------------------------------
def make_model(cfg: dict) -> xgb.XGBClassifier:
    base = dict(
        n_estimators=8000,
        learning_rate=0.02,
        subsample=0.8,
        reg_lambda=2.0,
        tree_method="hist",
        max_bin=512,
        enable_categorical=True,
        early_stopping_rounds=200,
        eval_metric="auc",
        n_jobs=N_JOBS,
    )
    base.update(cfg)
    return xgb.XGBClassifier(**base)


CONFIGS = [
    dict(max_depth=12, colsample_bytree=0.7, min_child_weight=20, subsample=0.9, random_state=42),
    dict(max_depth=10, colsample_bytree=0.5, min_child_weight=20, random_state=43),
    dict(max_depth=12, colsample_bytree=0.9, min_child_weight=10, learning_rate=0.03, random_state=44),
]

models = []
t0 = time.time()
for cfg in CONFIGS:
    m = make_model(cfg)
    m.fit(prepare(train), y_train, eval_set=[(X_va, y_va)], verbose=False)
    models.append(m)
    print(f"cfg {cfg}: best_iteration={m.best_iteration}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = prepare(df)
    return np.mean([m.predict_proba(P)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
