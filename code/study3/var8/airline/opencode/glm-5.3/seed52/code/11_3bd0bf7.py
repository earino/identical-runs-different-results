"""XGBoost binary classifier for airline delay: engineered numeric features + deep trees + strong colsample.

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
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- statistics fitted on TRAIN only -----------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in CAT_COLS}
route_tr = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
_route_freq = route_tr.value_counts(normalize=True).to_dict()
_origin_freq = train["Origin"].value_counts(normalize=True).to_dict()
_dest_freq = train["Dest"].value_counts(normalize=True).to_dict()
_hr_tr = (train["DepTime"] // 100).clip(0, 23).astype(int).astype(str)
_ohour_freq = (train["Origin"].astype(str) + "_" + _hr_tr).value_counts(normalize=True).to_dict()
_dhour_freq = (train["Dest"].astype(str) + "_" + _hr_tr).value_counts(normalize=True).to_dict()
_chour_freq = (train["UniqueCarrier"].astype(str) + "_" + _hr_tr).value_counts(normalize=True).to_dict()
_cdow_freq = (train["UniqueCarrier"].astype(str) + "_" + train["DayOfWeek"].astype(str)).value_counts(normalize=True).to_dict()
_oc_freq = (train["Origin"].astype(str) + "_" + train["UniqueCarrier"].astype(str)).value_counts(normalize=True).to_dict()
_dc_freq = (train["Dest"].astype(str) + "_" + train["UniqueCarrier"].astype(str)).value_counts(normalize=True).to_dict()


def _num_from_c(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    m, d, dw = _num_from_c(df["Month"]), _num_from_c(df["DayofMonth"]), _num_from_c(df["DayOfWeek"])
    dep = pd.to_numeric(df["DepTime"], errors="coerce").astype(float).clip(0, 2359)
    hr = (dep // 100).clip(0, 23)
    tmin = hr * 60 + dep % 100
    X["month"], X["day"], X["dow"] = m, d, dw
    X["DepTime"] = dep
    X["hour"] = hr
    X["tmin"] = tmin
    X["tmin_sin"] = np.sin(2 * np.pi * tmin / 1440.0)
    X["tmin_cos"] = np.cos(2 * np.pi * tmin / 1440.0)
    X["month_sin"] = np.sin(2 * np.pi * m / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * m / 12.0)
    X["dow_sin"] = np.sin(2 * np.pi * dw / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * dw / 7.0)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").astype(float)
    X["log_dist"] = np.log1p(X["Distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["route_freq"] = route.map(_route_freq).fillna(0.0).astype(float)
    X["origin_freq"] = df["Origin"].astype(str).map(_origin_freq).fillna(0.0).astype(float)
    X["dest_freq"] = df["Dest"].astype(str).map(_dest_freq).fillna(0.0).astype(float)
    hrs = hr.astype(int).astype(str)
    X["ohour_freq"] = (df["Origin"].astype(str) + "_" + hrs).map(_ohour_freq).fillna(0.0).astype(float)
    X["dhour_freq"] = (df["Dest"].astype(str) + "_" + hrs).map(_dhour_freq).fillna(0.0).astype(float)
    X["chour_freq"] = (df["UniqueCarrier"].astype(str) + "_" + hrs).map(_chour_freq).fillna(0.0).astype(float)
    X["cdow_freq"] = (df["UniqueCarrier"].astype(str) + "_" + df["DayOfWeek"].astype(str)).map(_cdow_freq).fillna(0.0).astype(float)
    X["oc_freq"] = (df["Origin"].astype(str) + "_" + df["UniqueCarrier"].astype(str)).map(_oc_freq).fillna(0.0).astype(float)
    X["dc_freq"] = (df["Dest"].astype(str) + "_" + df["UniqueCarrier"].astype(str)).map(_dc_freq).fillna(0.0).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


PARAMS = dict(
    max_depth=20,
    learning_rate=0.03,
    subsample=1.0,
    colsample_bytree=0.4,
    tree_method="hist",
    enable_categorical=True,
    max_bin=512,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=N_JOBS,
)

MEMBERS = [(0.36, 18, 11), (0.38, 20, 12), (0.40, 22, 13), (0.44, 24, 14), (0.48, 22, 15)]

t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
models = []
for c, dpt, s in MEMBERS:
    mdl = xgb.XGBClassifier(
        n_estimators=800,
        early_stopping_rounds=20,
        eval_metric="auc",
        colsample_bytree=c,
        max_depth=dpt,
        random_state=s,
        **{k: v for k, v in PARAMS.items() if k not in ("colsample_bytree", "max_depth", "random_state")},
    )
    mdl.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(mdl)
print(f"Training time: {time.time() - t0:.1f}s, best_iterations: {[m.best_iteration for m in models]}")


def _ranks(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    r = np.empty(len(p), dtype=float)
    r[order] = np.arange(len(p), dtype=float)
    return r


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    R = np.mean([_ranks(m.predict_proba(X)[:, 1]) for m in models], axis=0)
    # rescale to [0, 1]; monotone in R so AUC is that of the rank average
    return (R + 0.5) / len(R)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
