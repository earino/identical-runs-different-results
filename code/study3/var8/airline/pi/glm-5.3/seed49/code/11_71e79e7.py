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


def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace("c-", "", regex=False), errors="coerce")


def _hour(df: pd.DataFrame) -> pd.Series:
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(int)
    return (dep // 100).clip(0, 23)


def _cat_levels(series: pd.Series) -> pd.Index:
    return pd.Index(sorted(series.unique()))


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


# interaction categorical levels (fit on train only)
hc_levels = _cat_levels(train["UniqueCarrier"].astype(str) + "_" + _hour(train).astype(str))
hb = (_hour(train) // 3).astype(str)
oh_levels = _cat_levels(train["Origin"].astype(str) + "_" + hb)
dh_levels = _cat_levels(train["Dest"].astype(str) + "_" + hb)
dowh_levels = _cat_levels(train["DayOfWeek"].astype(str) + "_" + _hour(train).astype(str))


def add_hc(X, df):
    hc = df["UniqueCarrier"].astype(str) + "_" + _hour(df).astype(str)
    X["carrier_hour"] = pd.Categorical(hc, categories=hc_levels)
    return X


def add_doy(X, df):
    X["day_of_year"] = (_cnum(df["Month"]) - 1) * 31 + _cnum(df["DayofMonth"])
    return X


def add_oh(X, df):
    k = df["Origin"].astype(str) + "_" + (_hour(df) // 3).astype(str)
    X["origin_hb"] = pd.Categorical(k, categories=oh_levels)
    return X


def add_dh(X, df):
    k = df["Dest"].astype(str) + "_" + (_hour(df) // 3).astype(str)
    X["dest_hb"] = pd.Categorical(k, categories=dh_levels)
    return X


def add_dowh(X, df):
    k = df["DayOfWeek"].astype(str) + "_" + _hour(df).astype(str)
    X["dow_hour"] = pd.Categorical(k, categories=dowh_levels)
    return X


EXTRAS = {"hc": add_hc, "doy": add_doy, "oh": add_oh, "dh": add_dh, "dowh": add_dowh}

VARIANT_SETS = {
    "hc": ["hc"],
    "hc_doy": ["hc", "doy"],
    "hc_doy_oh": ["hc", "doy", "oh"],
    "hc_doy_dh": ["hc", "doy", "dh"],
    "hc_doy_dowh": ["hc", "doy", "dowh"],
}


def make_prep(extras):
    def prep(df):
        X = base_X(df)
        for e in extras:
            X = EXTRAS[e](X, df)
        return X
    return prep


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
    print(f"[probe] {name}: peak={max(r):.4f} at {int(np.argmax(r)) + 1} it, tail={r[-1]:.4f}")
    return m, max(r)


results, models, preps = {}, {}, {}
for name, extras in VARIANT_SETS.items():
    prep = make_prep(extras)
    m, peak = fit_probe(name, prep)
    results[name] = peak
    models[name], preps[name] = m, prep

best_name = max(results, key=results.get)
model, prepare_best = models[best_name], preps[best_name]
print(f"[probe] best variant: {best_name} ({results[best_name]:.4f})")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare_best(df))[:, 1]


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
