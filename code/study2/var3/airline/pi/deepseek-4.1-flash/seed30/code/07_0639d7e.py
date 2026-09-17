"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Design notes (airline, train=2005 / eval=2006):
  * Only year-stable features are used. Calendar features (Month, DayofMonth) carried 2005-specific
    weather patterns that did NOT transfer to 2006; removing them raised eval AUC by ~0.007.
    DayOfWeek is stable across years and is kept. Departure time is kept because its effect on delay
    is structural (delays accumulate through the day) and transfers.
  * Explicit interactions that transfer: carrier x hour (some carriers degrade later in the day) and
    origin-hour traffic volume (airport congestion) help by ~0.007.
  * Carrier / Origin / Dest / carrier-hour are native XGBoost categoricals. Target encodings and
    origin-dest route encodings were tested and hurt (they do not transfer across the year boundary).
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
TWO_PI = 2.0 * np.pi

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")


def _to_int(s: pd.Series) -> pd.Series:
    """'c-12' -> 12, passthrough numeric."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _dep_min(df: pd.DataFrame) -> pd.Series:
    dt = df["DepTime"].astype(float).fillna(0.0)
    dm = (dt // 100.0) * 60.0 + (dt % 100.0)
    return dm.where(dm <= 1440.0, dm - 1440.0)  # fix rare hhmm typos > 2400


# --- feature plan (statistics fit on train only) ------------------------------
raw_cat = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in raw_cat}

# carrier x hour-of-day interaction (structural: delays accumulate during the day)
_train_hour = (_dep_min(train) // 60.0).astype(int).astype(str)
cat_levels["carrier_hour"] = pd.Index(
    sorted((train["UniqueCarrier"].astype(str) + "_" + _train_hour).unique())
)

# origin-hour and destination-hour traffic volume (congestion proxies), frequency stats
_oh_counts = (
    train.assign(_o=train["Origin"], _h=(_dep_min(train) // 60.0).astype(int))
    .groupby(["_o", "_h"])
    .size()
)
_oh_median = float(_oh_counts.median())
_dh_counts = (
    train.assign(_d=train["Dest"], _h=(_dep_min(train) // 60.0).astype(int))
    .groupby(["_d", "_h"])
    .size()
)
_dh_median = float(_dh_counts.median())


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    dow = _to_int(df["DayOfWeek"]).astype(float)
    dm = _dep_min(df)
    hour = (dm // 60.0).astype(int)

    X["dow"] = dow
    X["dep_min"] = dm
    X["dep_sin"] = np.sin(TWO_PI * dm / 1440.0)
    X["dep_cos"] = np.cos(TWO_PI * dm / 1440.0)
    X["is_red_eye"] = ((dm < 360.0) | (dm >= 1260.0)).astype(float)
    X["dist"] = df["Distance"].astype(float).fillna(0.0)

    X["oh_cnt"] = (
        pd.MultiIndex.from_arrays([df["Origin"], hour])
        .map(_oh_counts)
        .fillna(_oh_median)
        .astype(float)
    )
    X["dh_cnt"] = (
        pd.MultiIndex.from_arrays([df["Dest"], hour])
        .map(_dh_counts)
        .fillna(_dh_median)
        .astype(float)
    )
    # diagonal distance x time-of-day interaction (a single ratio split captures what
    # axis-aligned trees need many splits for)
    X["dist_h"] = X["dist"] / (dm + 1.0)
    for c in raw_cat:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    X["carrier_hour"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "_" + hour.astype(str),
        categories=cat_levels["carrier_hour"],
    )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Small ensemble of diverse, weakly-regularised deep trees. Averaging several configs
# reduces the run-to-run variance that dominates single-model comparisons on eval.
BASE = dict(
    learning_rate=0.03,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
    subsample=0.9,
    reg_lambda=1.0,
    reg_alpha=1.0,
)
CONFIGS = [
    dict(max_depth=16, colsample_bytree=0.4, n_estimators=250, min_child_weight=1, random_state=11),
    dict(max_depth=18, colsample_bytree=0.4, n_estimators=200, min_child_weight=2, random_state=22),
    dict(max_depth=14, colsample_bytree=0.4, n_estimators=250, min_child_weight=2, random_state=33),
    dict(max_depth=16, colsample_bytree=0.4, n_estimators=200, min_child_weight=2, random_state=44),
    dict(max_depth=0, grow_policy="lossguide", max_leaves=512, colsample_bytree=0.4,
         n_estimators=200, min_child_weight=2, random_state=55),
    dict(max_depth=20, colsample_bytree=0.35, n_estimators=180, min_child_weight=1, random_state=66),
]

t0 = time.time()
Xtr = prepare(train)
ytr = to_y(train)
models = []
for cfg in CONFIGS:
    m = xgb.XGBClassifier(**dict(BASE, **cfg))
    m.fit(Xtr, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
