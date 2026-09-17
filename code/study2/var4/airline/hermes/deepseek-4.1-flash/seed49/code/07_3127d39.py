"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives in `prepare()`, which predict_proba() calls on unseen rows.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature schema (fitted on training data only) ------------------------------
CAL_COLS = ["Month", "DayofMonth", "DayOfWeek"]   # c-<n> levels that are really ordered integers
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000 and c not in CAL_COLS]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols or c in CAL_COLS]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
CUM_DAYS = np.array([0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=np.int64)
# carrier/airport x hour-of-day: operational patterns ("this airline at 6am") that are year-stable
_TRAIN_HOUR = (train["DepTime"].to_numpy() // 100) % 24
INTER_COLS = {"carrier_hour": "UniqueCarrier", "origin_hour": "Origin"}
INTER_LEVELS = {
    _n: pd.Index(sorted((train[_c].astype(str) + "_" + _TRAIN_HOUR.astype(str)).unique()))
    for _n, _c in INTER_COLS.items()
}


# --- feature engineering -------------------------------------------------------
def ord_cal(s: pd.Series) -> np.ndarray:
    """'c-<n>' encoded calendar levels -> their integer value (unparseable -> NaN)."""
    return pd.to_numeric(s.astype(str).str.slice(2), errors="coerce").to_numpy(dtype=np.float64)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything
    # computed outside this function (other than training set statistics) will NOT reach the hidden holdout.
    X = df[feature_cols].copy()

    # scheduled departure time: hhmm integer -> clock features (values >= 2400 mean past midnight)
    dep = X["DepTime"].to_numpy(dtype=np.int64)
    hour = dep // 100
    minute = dep % 100
    X["dep_hour"] = np.where(hour >= 24, hour - 24, hour)
    X["dep_minute"] = minute
    X["dep_tod"] = X["dep_hour"] * 60 + minute           # minutes after midnight
    X["dep_red_eye"] = (hour >= 24).astype(np.int8)

    # calendar: the c-<n> levels are ordered (month 1-12, day 1-31, weekday 1-7)
    month = ord_cal(X["Month"])
    dom = ord_cal(X["DayofMonth"])
    X["month"] = month
    X["day_of_month"] = dom
    X["day_of_week"] = ord_cal(X["DayOfWeek"])
    X["day_of_year"] = np.where(np.isfinite(month), CUM_DAYS[np.nan_to_num(month, nan=1).astype(np.int64).clip(1, 12)], np.nan) + dom
    X = X.drop(columns=CAL_COLS)

    for name, col in INTER_COLS.items():
        key = X[col].astype(str) + "_" + X["dep_hour"].astype(str)
        X[name] = pd.Categorical(key, categories=INTER_LEVELS[name])

    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


PARAMS = dict(
    n_estimators=700,
    max_depth=13,
    learning_rate=0.02,
    tree_method="hist",
    enable_categorical=True,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
Xtr = prepare(train)
ytr = to_y(train)
model = xgb.XGBClassifier(**PARAMS)
model.fit(Xtr, ytr, verbose=False)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
