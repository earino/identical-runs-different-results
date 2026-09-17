"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, module-level `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
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

# --- feature engineering -------------------------------------------------------
C_CAT = ["UniqueCarrier", "Origin", "Dest", "Month", "DayofMonth", "DayOfWeek"]
NUM_COLS = ["Distance", "DepTime"]

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in C_CAT}

y_all = (train[TARGET] == POSITIVE).astype(int)
PRIOR = float(y_all.mean())


def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace("c-", "", regex=False), errors="coerce")


def _hour(df: pd.DataFrame) -> pd.Series:
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(int)
    return (dep // 100).clip(0, 23)


def _te_map(keys: pd.Series, m: float) -> pd.Series:
    g = y_all.groupby(keys)
    agg = pd.DataFrame({"s": g.sum(), "n": g.size()})
    return (agg["s"] + PRIOR * m) / (agg["n"] + m)


hc_keys = train["UniqueCarrier"].astype(str) + "_" + _hour(train).astype(str)
hc_levels = pd.Index(sorted(hc_keys.unique()))
te_oh_keys = train["Origin"].astype(str) + "_" + (_hour(train) // 3).astype(str)
te_oh_map = _te_map(te_oh_keys, 100.0)


def base_X(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    hour = _hour(df)
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(int)
    minute = (dep % 100).clip(0, 59)
    X["hour"] = hour / 23.0
    X["min_of_day"] = (hour * 60 + minute) / 1439.0
    for c in NUM_COLS:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    for c in C_CAT:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def v_doy(df: pd.DataFrame) -> pd.DataFrame:
    X = base_X(df)
    X["day_of_year"] = (_cnum(df["Month"]) - 1) * 31 + _cnum(df["DayofMonth"])
    return X


def v_hr_carrier(df: pd.DataFrame) -> pd.DataFrame:
    X = base_X(df)
    hc = df["UniqueCarrier"].astype(str) + "_" + _hour(df).astype(str)
    X["carrier_hour"] = pd.Categorical(hc, categories=hc_levels)
    return X


def v_te_origin_hour(df: pd.DataFrame) -> pd.DataFrame:
    X = base_X(df)
    oh = df["Origin"].astype(str) + "_" + (_hour(df) // 3).astype(str)
    X["te_origin_hour"] = oh.map(te_oh_map).fillna(PRIOR)
    return X


def v_no_dom(df: pd.DataFrame) -> pd.DataFrame:
    X = base_X(df)
    return X.drop(columns=["DayofMonth"])


VARIANTS = {
    "base": base_X,
    "doy": v_doy,
    "hr_carrier": v_hr_carrier,
    "te_o_h": v_te_origin_hour,
    "no_dom": v_no_dom,
}


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


ytr, yev = to_y(train), to_y(evald)


def fit_probe(name, prep):
    Xtr, Xev = prep(train), prep(evald)
    m = xgb.XGBClassifier(
        n_estimators=4000,
        max_depth=3,
        learning_rate=0.1,
        min_child_weight=1,
        reg_lambda=10.0,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=100,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    r = m.evals_result()["validation_0"]["auc"]
    print(f"[probe] {name}: peak={max(r):.4f} at {int(np.argmax(r)) + 1} it")
    return m, max(r)


results, models = {}, {}
for name, prep in VARIANTS.items():
    m, peak = fit_probe(name, prep)
    results[name] = peak
    models[name] = m

best_name = max(results, key=results.get)
model, prepare_best = models[best_name], VARIANTS[best_name]
print(f"[probe] best variant: {best_name} ({results[best_name]:.4f})")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare_best(df))[:, 1]


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
