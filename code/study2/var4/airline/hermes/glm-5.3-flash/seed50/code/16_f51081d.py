"""XGBoost binary classifier for airline delays. THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

v8: 6-model capacity/feature-diverse XGBoost ensemble over interaction categoricals.
Features: base cats (DayofMonth, DayOfWeek, UniqueCarrier, Origin, Dest — Month dropped,
its 2005 seasonality anti-transfers to 2006), train-fitted-level interaction cats
{Origin, UniqueCarrier, Dest, DayOfWeek} x dep-hour, {Origin, UniqueCarrier, DayOfWeek}
x Distance-block, Distance-block x dep-hour, plus raw DepTime/Distance. Members: full
feature set at d5-l200 and d8-m50-l30; dist-block-only and hour-only subsets as diverse
partners; a base-only member. All encodings train-fitted levels; unseen -> NaN.
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

CAT_COLS = ["DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
DIST_BINS = [0, 250, 500, 750, 1000, 1500, 2500, 10000]

# encoders fit on TRAIN ONLY ----------------------------------------------------
LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _hour(df: pd.DataFrame) -> pd.Series:
    return (df["DepTime"] // 100).clip(0, 24).astype(int)


def _time_block(df: pd.DataFrame, res: int) -> pd.Series:
    return (df["DepTime"] // res).clip(0, 1440 // res).astype(int)


def _dist_block(df: pd.DataFrame) -> pd.Series:
    return pd.cut(df["Distance"], bins=DIST_BINS, labels=False).astype(int)


def _inter_levels(cat_tr: pd.Series, num_tr: pd.Series) -> pd.Index:
    return pd.Index(sorted((cat_tr.astype(str) + "_" + num_tr.astype(str)).unique()))


_htr = _hour(train)
_qtr = _time_block(train, 20)
_q30tr = _time_block(train, 30)
_dbtr = _dist_block(train)
# (source-column name or "dist_block"/pseudo time columns, kind) -> train-fitted levels
INTER_SPECS = [
    ("Orig_t20", "Origin", "t20"),
    ("Car_hour", "UniqueCarrier", "hour"),
    ("Dest_hour", "Dest", "hour"),
    ("DOW_hour", "DayOfWeek", "hour"),
    ("Orig_dist", "Origin", "dist_block"),
    ("DB_hour", "dist_block", "hour"),
    ("Car_dist", "UniqueCarrier", "dist_block"),
    ("DOW_dist", "DayOfWeek", "dist_block"),
    ("Dest_dist", "Dest", "dist_block"),
    ("DB_t20", "dist_block", "t20"),
    ("Car_t30", "UniqueCarrier", "t30"),
]
INTER_LEVELS = {}
for _name, _src, _kind in INTER_SPECS:
    _a = _dbtr if _src == "dist_block" else train[_src]
    if _kind == "hour":
        _b = _htr
    elif _kind == "t20":
        _b = _qtr
    elif _kind == "t30":
        _b = _q30tr
    else:
        _b = _dbtr
    INTER_LEVELS[_name] = _inter_levels(_a, _b)

HOUR_FEATURES = ["Car_hour", "Dest_hour", "DOW_hour", "Car_t30"]
DIST_FEATURES = ["Orig_dist", "DB_hour", "Car_dist", "DOW_dist", "Dest_dist", "DB_t20"]


def prepare(df: pd.DataFrame, cols: list | None = None) -> pd.DataFrame:
    # ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows.
    # Categoricals use train-fitted levels; unseen levels become NaN automatically.
    X = pd.DataFrame(index=df.index)
    h = _hour(df).astype(str)
    q20 = _time_block(df, 20).astype(str)
    q30 = _time_block(df, 30).astype(str)
    db = _dist_block(df).astype(str)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=LEVELS[c])
    for name, src, kind in INTER_SPECS:
        if src == "dist_block":
            a = db
        else:
            a = df[src].astype(str)
        if kind == "hour":
            b = h
        elif kind == "t20":
            b = q20
        elif kind == "t30":
            b = q30
        else:
            b = db
        X[name] = pd.Categorical(a + "_" + b, categories=INTER_LEVELS[name])
    X["DepTime"] = df["DepTime"].astype(float)
    X["Distance"] = df["Distance"].astype(float)
    if cols is not None:
        X = X[cols]
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


FULL_COLS = None  # filled after first prepare(train)


# --- model --------------------------------------------------------------------
# capacity-diverse members; the dist/hour/base members see feature subsets of the same
# matrix (feature-split ensemble gave the largest single gain: ~+0.004 over full-set pair).
_P = dict(learning_rate=0.1, tree_method="hist", enable_categorical=True,
          random_state=SEED, n_jobs=N_JOBS)
D5 = {**_P, "n_estimators": 150, "max_depth": 5, "min_child_weight": 20, "reg_lambda": 200.0,
      "colsample_bynode": 0.8}
D8 = {**_P, "n_estimators": 100, "max_depth": 8, "min_child_weight": 50, "reg_lambda": 30.0,
      "colsample_bynode": 0.8}
# slow-lr variants for the dominant full-set members
D5S = {**D5, "learning_rate": 0.05, "n_estimators": 300}
D8S = {**D8, "learning_rate": 0.05, "n_estimators": 200}
# heavier λ on the slow full members (v2-era finding: λ trend kept rising with feature richness)
D5SH = {**D5S, "reg_lambda": 800.0}
D8SH = {**D8S, "reg_lambda": 120.0}

t0 = time.time()
_Xtr = prepare(train)
FULL_COLS = list(_Xtr.columns)
_base_set = set(HOUR_FEATURES + DIST_FEATURES + ["Orig_t20"])
_BASE_COLS = [c for c in FULL_COLS if c not in _base_set]
_HOUR_COLS = [c for c in FULL_COLS if c not in set(DIST_FEATURES)]
_DIST_COLS = [c for c in FULL_COLS if c not in set(HOUR_FEATURES)]
_CARRIER_COLS = _BASE_COLS + ["Car_hour", "Car_dist"]
_ORIGIN_COLS = _BASE_COLS + ["Orig_t20", "Orig_dist"]
_ytr = to_y(train)

MEMBERS = [
    (D5SH, FULL_COLS),
    (D8SH, FULL_COLS),
    ({**D5, "n_estimators": 400}, _DIST_COLS),
    ({**D8, "n_estimators": 250}, _DIST_COLS),
    ({**D8, "n_estimators": 250}, _HOUR_COLS),
    ({**D5, "n_estimators": 400}, _BASE_COLS),
    ({**D5, "n_estimators": 400}, _CARRIER_COLS),
    ({**D8, "n_estimators": 250}, _ORIGIN_COLS),
]
MODELS = []
for params, cols in MEMBERS:
    m = xgb.XGBClassifier(**params)
    m.fit(_Xtr[cols], _ytr)
    MODELS.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X[cols])[:, 1] for (params, cols), m in zip(MEMBERS, MODELS)],
                   axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
