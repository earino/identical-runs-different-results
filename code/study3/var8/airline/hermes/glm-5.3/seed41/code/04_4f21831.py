"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

This experiment: probe feature additions with the best (shallow) config.
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
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

BEST = dict(n_estimators=300, max_depth=3, learning_rate=0.05, reg_lambda=5.0)


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(r"^c-", "", regex=True), errors="coerce")


def prepare(df: pd.DataFrame, variant: str = "base") -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[feature_cols].copy()
    mon = _num(X["Month"])
    day = _num(X["DayofMonth"])
    dep = pd.to_numeric(X["DepTime"], errors="coerce").astype("float64")
    X["DepHour"] = np.floor(dep / 100.0)
    if variant in ("doy", "all"):
        X["DayOfYear"] = (mon - 1) * 31 + day
        X["DayOfWeekNum"] = _num(X["DayOfWeek"])
    if variant == "logdist":
        X["LogDist"] = np.log1p(pd.to_numeric(X["Distance"], errors="coerce").astype("float64"))
    if variant == "hourbin":
        X["HourBin"] = pd.cut(dep, bins=[0, 559, 1059, 1359, 1659, 1959, 2359],
                               labels=False).astype("float64")
    if variant in ("inter", "all"):
        X["HourxDow"] = X["DepHour"] * 10 + _num(X["DayOfWeek"])
        X["HourxMonth"] = X["DepHour"] * 13 + mon
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- probe ---------------------------------------------------------------------
ytr, yev = to_y(train), to_y(evald)

t0 = time.time()
variants = []


def probe(name, variant):
    Xtr, Xev = prepare(train, variant), prepare(evald, variant)
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED,
                          n_jobs=N_JOBS, **BEST)
    m.fit(Xtr, ytr)
    auc = roc_auc_score(yev, m.predict_proba(Xev)[:, 1])
    variants.append((auc, name, variant, m))
    print(f"[probe] {name}: {auc:.4f}  ({time.time() - t0:.1f}s)")


probe("base", "base")          # control: hour only (0.7160 last time, recheck with lam5)
probe("doy", "doy")            # day of year + dow numeric
probe("logdist", "logdist")    # log distance
probe("hourbin", "hourbin")    # binned hour
probe("inter", "inter")        # hour x dow, hour x month
probe("all", "all")            # everything

best_auc, best_name, best_variant, model = max(variants, key=lambda r: r[0])
print(f"[probe] selected {best_name} (variant={best_variant})")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, best_variant))[:, 1]


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
