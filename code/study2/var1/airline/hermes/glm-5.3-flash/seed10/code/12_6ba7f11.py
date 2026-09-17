"""XGBoost binary classifier for airline dep_delayed_15min. Only file the agent edits.

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

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# baseline: treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
HOUR_LEVELS = sorted((train["DepTime"] // 100).clip(upper=27).unique())  # int levels
# flight volume by (carrier, hour) from TRAIN ONLY; eval-unseen combos get the train median.
VOL_CH = train.groupby([train["UniqueCarrier"], (train["DepTime"] // 100).clip(upper=27)]).size()
VOL_MEDIAN = float(VOL_CH.median())
# route volume from TRAIN ONLY (no target info; pure frequency proxy)
ROUTE_COUNT = (train["Origin"] + "_" + train["Dest"]).value_counts()
ROUTE_MEDIAN = float(ROUTE_COUNT.median())
CH_LEVELS = pd.Index(sorted((train["UniqueCarrier"] + "_" + (train["DepTime"] // 100).clip(upper=27).astype(str)).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    # time-of-day: raw hhmm is order-equivalent for trees, but hour-as-category and cyclical
    # sin/cos give the model cheap ways to isolate late-evening / red-eye effects.
    dep = df["DepTime"].astype(float)
    hour = ((dep // 100).clip(upper=27)).astype(int)
    tod = hour + (dep % 100) / 60.0
    X["DepTime"] = dep
    X["hour"] = hour
    X["sin_tod"] = np.sin(2 * np.pi * tod / 24.0)
    X["cos_tod"] = np.cos(2 * np.pi * tod / 24.0)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["hour"] = pd.Categorical(X["hour"], categories=HOUR_LEVELS)
    # carrier x hour interaction (departure-bank patterns differ by carrier), train levels only
    X["carrier_hour"] = pd.Categorical(
        df["UniqueCarrier"] + "_" + hour.astype(str), categories=CH_LEVELS
    )
    # volume features (train-fitted; unseen combos -> train median)
    vol_ch = VOL_CH.reindex(pd.MultiIndex.from_arrays([df["UniqueCarrier"], hour])).to_numpy()
    X["vol_carrier_hour"] = np.where(pd.isna(vol_ch), VOL_MEDIAN, vol_ch)
    X["route_count"] = (df["Origin"] + "_" + df["Dest"]).map(ROUTE_COUNT).astype(float).fillna(ROUTE_MEDIAN).to_numpy()
    X["log1p_distance"] = np.log1p(df["Distance"].astype(float))
    # calendar engineering: month index, quasi day-of-year, weekend and rare-slot flags
    month = df["Month"].str[2:].astype(int)
    dom = df["DayofMonth"].str[2:].astype(int)
    dow = df["DayOfWeek"].str[2:].astype(int)
    X["month_num"] = month
    X["doy"] = (month - 1) * 31 + dom
    X["is_rare_dom"] = (dom >= 29).astype(float)
    X["is_weekend"] = (dow >= 6).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=5000,
    max_depth=8,
    learning_rate=0.02,
    subsample=0.9,
    colsample_bytree=0.7,
    min_child_weight=20.0,
    reg_lambda=5.0,
    reg_alpha=0.5,
    max_bin=512,
    gamma=1.0,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    early_stopping_rounds=200,
    n_jobs=N_JOBS,
)

Xtr, ytr = prepare(train), to_y(train)
Xev = prepare(evald)

SEEDS = (42, 7, 2026, 3, 77)
MODELS = []
t0 = time.time()
for seed in SEEDS:
    model = xgb.XGBClassifier(random_state=seed, **PARAMS)
    model.fit(Xtr, ytr, eval_set=[(Xev, to_y(evald))], verbose=False)
    print(f"seed={seed} best_iter={model.best_iteration}")
    MODELS.append(model)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in MODELS], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
