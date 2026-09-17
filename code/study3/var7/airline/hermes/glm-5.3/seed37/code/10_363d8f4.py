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

# --- feature engineering (all inside prepare: must apply to the hidden holdout) ---
CYC = {"Month": 12, "DayofMonth": 31, "DayOfWeek": 7}


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric/cyclic features. Called by prepare() on train, eval and hidden holdout."""
    X = pd.DataFrame(index=df.index)
    dt = df["DepTime"].astype(int)
    h, m = np.divmod(dt, 100)
    hour = (h + m / 60.0).astype(float)
    X["hour"] = hour
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["minofday"] = h * 60 + m
    X["dist_x_hour"] = X["minofday"] * df["Distance"].astype(float).to_numpy() / 1e5
    mon_v = df["Month"].str.extract(r"c-(\d+)", expand=False).astype(float)
    dom_v = df["DayofMonth"].str.extract(r"c-(\d+)", expand=False).astype(float)
    dow_v = df["DayOfWeek"].str.extract(r"c-(\d+)", expand=False).astype(float)
    for c, period in CYC.items():
        v = {"Month": mon_v, "DayofMonth": dom_v, "DayOfWeek": dow_v}[c]
        key = {"Month": "mon", "DayofMonth": "dom", "DayOfWeek": "dow"}[c]
        X[key] = v
        X[key + "_sin"] = np.sin(2 * np.pi * v / period)
        X[key + "_cos"] = np.cos(2 * np.pi * v / period)
    X["dist"] = df["Distance"].astype(float)
    X["dist_log"] = np.log1p(X["dist"])
    doy = (mon_v - 1) * 30.44 + dom_v
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X["is_weekend"] = (dow_v >= 6).astype(int)
    # schedule volume at (airport, hour): fit on TRAIN only (consistent scale on any frame)
    hb = (h % 24).astype(int)
    ko = df["Origin"].astype(str) + "_" + hb.astype(str)
    kd = df["Dest"].astype(str) + "_" + hb.astype(str)
    kc = df["UniqueCarrier"].astype(str) + "_" + hb.astype(str)
    X["o_hour_vol"] = ko.map(O_VOL).fillna(1).to_numpy().astype(float)
    X["d_hour_vol"] = kd.map(D_VOL).fillna(1).to_numpy().astype(float)
    X["c_hour_vol"] = kc.map(C_VOL).fillna(1).to_numpy().astype(float)
    return X


# volume lookups fit on TRAIN ONLY (schedule property, transfers across years)
_h_train = (train["DepTime"].astype(int) // 100) % 24
O_VOL = (train["Origin"].astype(str) + "_" + _h_train.astype(str)).value_counts()
D_VOL = (train["Dest"].astype(str) + "_" + _h_train.astype(str)).value_counts()
C_VOL = (train["UniqueCarrier"].astype(str) + "_" + _h_train.astype(str)).value_counts()


# native categorical features
CAT_COLS = list(CYC) + ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = _add_features(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Bagged ensemble of XGBoost seeds: variance reduction helps under time shift.
PARAMS = dict(
    n_estimators=1200,
    max_depth=4,
    learning_rate=0.02,
    min_child_weight=20,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=2.0,
    tree_method="hist",
    max_bin=1024,
    enable_categorical=True,
    n_jobs=N_JOBS,
)

X_train = prepare(train)
y_train = to_y(train)

t0 = time.time()
models = []
for seed in (42, 123, 2024):
    m = xgb.XGBClassifier(random_state=seed, **PARAMS)
    m.fit(X_train, y_train)
    models.append(m)
model = models[0]
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return ps


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
