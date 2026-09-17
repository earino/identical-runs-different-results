"""XGBoost airline delay: mixture of time-of-day experts + global ensemble.

Architecture: 48 30-minute departure-time windows each get 2 specialist XGB models
(fit only on that window's 2005 rows); 6 global models cover the whole day.
Final P = 0.35 * window-specialist mean + 0.65 * global mean. Fixed rounds, no early
stopping; eval.csv is used only for reporting.

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

# --- feature engineering (fit on train only) ------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: sorted(train[c].dropna().astype(str).unique()) for c in CAT_COLS}
_CUM = np.array([0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])  # day-of-year of month start

N_PART = 48          # 30-minute departure-time windows (fine specialists)
N_PART2 = 24         # 60-minute windows (coarse specialists)
W_FINE = 0.30        # blend weight of fine window specialists
W_COARSE = 0.10      # blend weight of coarse window specialists
ALPHA = 0.15         # recency weighting: later 2005 months matter more for 2006


def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    month = _cnum(df["Month"]).astype(int)
    dom = _cnum(df["DayofMonth"]).astype(int)
    X["month"] = month.astype(float)
    X["dom"] = dom.astype(float)
    X["dow"] = _cnum(df["DayOfWeek"]).astype(float)
    dep = df["DepTime"].astype(int)
    dep_min = ((dep // 100) * 60 + (dep % 100)) % 1440
    X["DepTime"] = dep.astype(float)
    X["dep_min"] = dep_min.astype(float)
    X["hour"] = (dep_min // 60).astype(float)
    X["Distance"] = df["Distance"].astype(float)
    # fixed-date holiday distances (day-of-year based; valid for both 2005/2006 non-leap)
    dy = (_CUM[month.to_numpy()] + dom.to_numpy()).astype(float)
    X["to_xmas"] = np.clip(dy - 359.0, -21.0, 35.0)
    X["to_ny"] = np.clip(dy - 1.0, -30.0, 30.0)
    X["to_jul4"] = np.clip(dy - 185.0, -30.0, 30.0)
    X["xmas_seas"] = np.where((dy >= 352.0) | (dy <= 5.0), 1.0, 0.0)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def _window(dep_min: np.ndarray, n_part: int) -> np.ndarray:
    return np.clip(dep_min.astype(int) // (1440 // n_part), 0, n_part - 1)


BASE = {"objective": "binary:logistic", "max_depth": 4, "tree_method": "hist",
        "eval_metric": "auc", "nthread": N_JOBS}
GLOBAL_CFGS = [
    {"learning_rate": 0.05, "min_child_weight": 20, "n_estimators": 150},
    {"learning_rate": 0.05, "min_child_weight": 10, "n_estimators": 155},
    {"learning_rate": 0.03, "min_child_weight": 20, "n_estimators": 240},
]
SPEC_CFGS = [
    {"learning_rate": 0.05, "min_child_weight": 5, "n_estimators": 180},
    {"learning_rate": 0.05, "min_child_weight": 10, "n_estimators": 155},
]

# --- train -----------------------------------------------------------------------
X_train = prepare(train)
y_train = to_y(train)
_train_month = _cnum(train["Month"]).astype(int).to_numpy()
w_train = np.exp(ALPHA * (_train_month - 12))

t0 = time.time()
dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, enable_categorical=True)
global_models = []
for cfg in GLOBAL_CFGS:
    for seed in (SEED, SEED + 1):
        params = {**BASE, "learning_rate": cfg["learning_rate"],
                  "min_child_weight": cfg["min_child_weight"], "seed": seed}
        global_models.append(xgb.train(params, dtrain, cfg["n_estimators"]))

win_train = _window(X_train["dep_min"].to_numpy(), N_PART)
spec_models = {}
for w in range(N_PART):
    idx = np.where(win_train == w)[0]
    if len(idx) == 0:
        continue
    dm = xgb.DMatrix(X_train.iloc[idx], label=y_train[idx], weight=w_train[idx], enable_categorical=True)
    spec_models[w] = [
        xgb.train({**BASE, **{k: v for k, v in c.items() if k != "n_estimators"}, "seed": SEED},
                  dm, c["n_estimators"])
        for c in SPEC_CFGS
    ]

win2_train = _window(X_train["dep_min"].to_numpy(), N_PART2)
spec2_models = {}
for w in range(N_PART2):
    idx = np.where(win2_train == w)[0]
    if len(idx) == 0:
        continue
    dm = xgb.DMatrix(X_train.iloc[idx], label=y_train[idx], weight=w_train[idx], enable_categorical=True)
    spec2_models[w] = [
        xgb.train({**BASE, **{k: v for k, v in c.items() if k != "n_estimators"}, "seed": SEED},
                  dm, c["n_estimators"])
        for c in SPEC_CFGS
    ]
print(f"Training time: {time.time() - t0:.1f}s ({len(global_models)} global + "
      f"{sum(len(v) for v in spec_models.values())} fine + {sum(len(v) for v in spec2_models.values())} coarse specialists)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    if len(X) == 0:
        return np.zeros(0)
    dm = xgb.DMatrix(X, enable_categorical=True)
    p_glob = np.mean([m.predict(dm) for m in global_models], axis=0)
    p_fine = np.zeros(len(X))
    p_coarse = np.zeros(len(X))
    have = np.zeros(len(X), dtype=bool)
    win = _window(X["dep_min"].to_numpy(), N_PART)
    win2 = _window(X["dep_min"].to_numpy(), N_PART2)
    for w, models in spec_models.items():
        rows = np.where(win == w)[0]
        if len(rows) == 0:
            continue
        sub = dm.slice(rows)
        p_fine[rows] = np.mean([m.predict(sub) for m in models], axis=0)
    for w, models in spec2_models.items():
        rows = np.where(win2 == w)[0]
        if len(rows) == 0:
            continue
        sub = dm.slice(rows)
        p_coarse[rows] = np.mean([m.predict(sub) for m in models], axis=0)
        have[rows] = True
    p = W_FINE * p_fine + W_COARSE * p_coarse + (1.0 - W_FINE - W_COARSE) * p_glob
    p[~have] = p_glob[~have]
    return p


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
