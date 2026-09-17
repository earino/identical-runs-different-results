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
# time-of-day periods known to matter for delays (morning/evening banks, red-eye, etc.)
BLK_EDGES = [0, 600, 1200, 1800, 2400]
BLK_LABELS = ["night", "morning", "afternoon", "evening"]
HOLIDAYS_US = {  # US federal holidays 2005-2006 (month, day) approximate travel-surround windows
    "704": 1, "1225": 1, "1231": 1, "101": 1, "1111": 1, "1224": 1,
    "528": 1, "529": 1, "530": 1, "702": 1, "829": 1, "830": 1, "1122": 1, "1123": 1,
}
# month-day key for travel-rush proximity (Thanksgiving/christmas etc.) via (mon, dom)


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric/cyclic features. Called by prepare() on train, eval and hidden holdout."""
    X = pd.DataFrame(index=df.index)
    dt = df["DepTime"].astype(int)
    h, m = np.divmod(dt, 100)
    hour = (h + m / 60.0).astype(float)
    X["hour"] = hour
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    # continuous minute-of-day and its square (schedule smoothness)
    mod = (dt % 2400) + (dt // 2400) * 60  # handles 2400 as midnight
    X["minofday"] = h * 60 + m
    # distance interactions with time of day (morning flights delay differently)
    X["dist_x_hour"] = X["minofday"] * df["Distance"].astype(float).to_numpy() / 1e5
    for c, period in CYC.items():
        v = df[c].str.extract(r"c-(\d+)", expand=False).astype(float)
        key = {"Month": "mon", "DayofMonth": "dom", "DayOfWeek": "dow"}[c]
        X[key] = v
        X[key + "_sin"] = np.sin(2 * np.pi * v / period)
        X[key + "_cos"] = np.cos(2 * np.pi * v / period)
    X["dist"] = df["Distance"].astype(float)
    X["dist_log"] = np.log1p(X["dist"])
    # seasonality: day-of-year proxy + weekend flag (transfers across years)
    mon_v = df["Month"].str.extract(r"c-(\d+)", expand=False).astype(float)
    dom_v = df["DayofMonth"].str.extract(r"c-(\d+)", expand=False).astype(float)
    dow_v = df["DayOfWeek"].str.extract(r"c-(\d+)", expand=False).astype(float)
    doy = (mon_v - 1) * 30.44 + dom_v  # approximate day of year
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X["is_weekend"] = (dow_v >= 6).astype(int)
    return X


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
