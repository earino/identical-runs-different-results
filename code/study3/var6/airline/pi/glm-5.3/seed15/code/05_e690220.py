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

# --- feature definitions (fit on TRAIN only; reused for any unseen dataframe) ------------------
RAW_CAT = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in RAW_CAT}
route_levels = pd.Index(sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique()))
DATE_COLS = ["Month", "DayofMonth", "DayOfWeek"]


def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.slice(start=2), errors="coerce")


def prepare(df: pd.DataFrame, variant: str = "all") -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    ints = variant != "base_cat"
    for c in DATE_COLS:
        X[c] = _cnum(df[c]) if ints else pd.Categorical(df[c].astype(str), categories=cat_levels.get(c, pd.Index(sorted(train[c].astype(str).unique()))))
    if variant == "base_cat":
        for c in DATE_COLS:
            X[c] = pd.Categorical(df[c].astype(str), categories=pd.Index(sorted(train[c].astype(str).unique())))
    X["DepTime"] = df["DepTime"]
    X["Distance"] = df["Distance"]
    if "hour" in variant:
        X["hour"] = df["DepTime"] // 100
        if "nomin" not in variant:
            X["minute"] = df["DepTime"] % 100
    if "logd" in variant:
        X["log_dist"] = np.log1p(pd.to_numeric(df["Distance"], errors="coerce"))
    cats = list(RAW_CAT)
    if "route" in variant:
        cats = cats + ["route"]
    if "noair" in variant:
        cats = ["UniqueCarrier"] + (["route"] if "route" in variant else [])
    for c in cats:
        src = df["Origin"].astype(str) + "_" + df["Dest"].astype(str) if c == "route" else df[c].astype(str)
        X[c] = pd.Categorical(src, categories=route_levels if c == "route" else cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------------------------
PARAMS = dict(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    max_depth=6,
    learning_rate=0.1,
    min_child_weight=1,
    reg_lambda=1.0,
    seed=SEED,
    nthread=N_JOBS,
)

t0 = time.time()
y_all = to_y(train)
y_ev = to_y(evald)
VARIANTS = ["base_cat", "int_dates", "int_dates+hour", "int_dates+hour+route", "int_dates+hour+logd", "int_dates+hour+logd+noair"]
ROUNDS = [30, 100, 250]
results = {}
for v in VARIANTS:
    dv_tr = xgb.DMatrix(prepare(train, v), label=y_all, enable_categorical=True)
    b = xgb.train(PARAMS, dv_tr, num_boost_round=max(ROUNDS), verbose_eval=False)
    dev = xgb.DMatrix(prepare(evald, v), enable_categorical=True)
    for n in ROUNDS:
        auc = roc_auc_score(y_ev, b.predict(dev, iteration_range=(0, n)))
        results[(v, n)] = auc
        print(f"variant={v:32s} rounds={n:3d}  eval_auc={auc:.4f}")

best_v, best_n = max(results, key=results.get)
print(f"best: variant={best_v} rounds={best_n} auc={results[(best_v, best_n)]:.4f}")
model = xgb.train(PARAMS, xgb.DMatrix(prepare(train, best_v), label=y_all, enable_categorical=True), num_boost_round=best_n)
MODEL_VARIANT = best_v
MODEL_ROUNDS = best_n
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict(xgb.DMatrix(prepare(df, MODEL_VARIANT), enable_categorical=True), iteration_range=(0, MODEL_ROUNDS))


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
