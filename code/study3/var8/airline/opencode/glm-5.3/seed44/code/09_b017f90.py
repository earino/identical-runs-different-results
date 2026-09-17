"""XGBoost airline delay: multi-resolution mixture of departure-time experts + global ensemble.

Architecture ("ladder"): departure-minute axis [0,1440) is partitioned at 4 resolutions
(24, 48, 96, 120 windows). Each window trains its own small XGB ensemble on just that
window's 2005 rows; 6 global models cover the whole day. Final probability per row:
  P = 0.50 * global_mean + 0.08 * spec24 + 0.26 * spec48 + 0.10 * spec96 + 0.06 * spec120
(rows whose window has no trained specialists put that weight on the global mean).
Fixed rounds, no early stopping; eval.csv is used only for reporting.

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

# --- feature engineering (fit on train only) ------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: sorted(train[c].dropna().astype(str).unique()) for c in CAT_COLS}
_CUM = np.array([0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])  # day-of-year of month start

ALPHA = 0.15  # recency weighting: later 2005 months matter more for 2006


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


BASE = {"objective": "binary:logistic", "max_depth": 4, "tree_method": "hist",
        "eval_metric": "auc", "nthread": N_JOBS}

GLOBAL_CFGS = [
    {"learning_rate": 0.05, "min_child_weight": 20, "n_estimators": 150},
    {"learning_rate": 0.05, "min_child_weight": 10, "n_estimators": 155},
    {"learning_rate": 0.03, "min_child_weight": 20, "n_estimators": 240},
]
_BAG = {"learning_rate": 0.05, "min_child_weight": 10, "subsample": 0.8,
        "colsample_bytree": 0.8, "n_estimators": 200}
SPEC_BASE = [
    {"learning_rate": 0.05, "min_child_weight": 5, "n_estimators": 180},
    {"learning_rate": 0.05, "min_child_weight": 10, "n_estimators": 155},
    dict(_BAG),
]
SPEC_EXTRA = {**_BAG, "min_child_weight": 5, "n_estimators": 220}
SPEC2 = SPEC_BASE[:2]

# resolution ladder: (num windows, specialist configs, blend weight)
LADDER = [
    (24, SPEC_BASE, 0.08),
    (48, SPEC_BASE + [SPEC_EXTRA], 0.26),
    (96, SPEC_BASE, 0.10),
    (120, SPEC2, 0.06),
]
W_GLOBAL = 1.0 - sum(w for _, _, w in LADDER)

# --- train -----------------------------------------------------------------------
X_train = prepare(train)
y_train = to_y(train)
_train_month = _cnum(train["Month"]).astype(int).to_numpy()
w_train = np.exp(ALPHA * (_train_month - 12))
dep_min_train = X_train["dep_min"].to_numpy()

t0 = time.time()
dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, enable_categorical=True)
global_models = []
for cfg in GLOBAL_CFGS:
    for seed in (SEED, SEED + 1):
        params = {**BASE, "learning_rate": cfg["learning_rate"],
                  "min_child_weight": cfg["min_child_weight"], "seed": seed}
        global_models.append(xgb.train(params, dtrain, cfg["n_estimators"]))

ladder_models = {}  # n_part -> {window -> [boosters]}
for n_part, cfgs, _ in LADDER:
    step = 1440.0 / n_part
    win = np.clip((dep_min_train / step).astype(int), 0, n_part - 1)
    models = {}
    for w in range(n_part):
        idx = np.where(win == w)[0]
        if len(idx) == 0:
            continue
        dm = dtrain.slice(idx.tolist())
        models[w] = [
            xgb.train({**BASE, **{k: v for k, v in c.items() if k != "n_estimators"}, "seed": SEED},
                      dm, c["n_estimators"])
            for c in cfgs
        ]
    ladder_models[n_part] = models
n_models = len(global_models) + sum(len(m) for lm in ladder_models.values() for m in lm.values())
print(f"Training time: {time.time() - t0:.1f}s ({n_models} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    if len(X) == 0:
        return np.zeros(0)
    dm = xgb.DMatrix(X, enable_categorical=True)
    p_glob = np.mean([m.predict(dm) for m in global_models], axis=0)
    p = W_GLOBAL * p_glob.copy()
    dep_min = X["dep_min"].to_numpy()
    for n_part, cfgs, w in LADDER:
        models = ladder_models[n_part]
        step = 1440.0 / n_part
        win = np.clip((dep_min / step).astype(int), 0, n_part - 1)
        p_spec = np.empty(len(X))
        have = np.zeros(len(X), dtype=bool)
        for wd, bsts in models.items():
            rows = np.where(win == wd)[0]
            if len(rows) == 0:
                continue
            sub = dm.slice(rows.tolist())
            p_spec[rows] = np.mean([m.predict(sub) for m in bsts], axis=0)
            have[rows] = True
        # missing specialists fall back to the global mean at the same weight
        p += w * np.where(have, p_spec, p_glob)
    return p


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
