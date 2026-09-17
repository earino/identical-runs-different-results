"""XGBoost binary classifier for airline delay. Agent-edited file.

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
from sklearn.model_selection import train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- encoders fitted on TRAIN only --------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c])]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in obj_cols}
MONTHS = {f"c-{i}": i for i in range(1, 13)}
DOM = {f"c-{i}": i for i in range(1, 32)}
CUMDAYS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])  # non-leap 2005

TE_SMOOTH = 20  # smoothing prior for target encoding


def _dep_hour(df: pd.DataFrame) -> pd.Series:
    v = df["DepTime"].astype("float64") % 2400
    return (v // 100).astype("int32")


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "->" + df["Dest"].astype(str)


def _dow(df: pd.DataFrame) -> pd.Series:
    return df["DayOfWeek"].astype(str).map(lambda s: int(str(s).split("-")[1]) if pd.notna(s) else np.nan)


def _te_map(keys: pd.Series, y: np.ndarray, prior: float = TE_SMOOTH) -> pd.Series:
    g = pd.DataFrame({"k": keys.astype(str).values, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + prior * y.mean()) / (g["count"] + prior)


def _freq_map(keys: pd.Series) -> pd.Series:
    return keys.astype(str).value_counts()


# build TE/freq maps from TRAIN ONLY
_y = (train[TARGET] == POSITIVE).astype(int).to_numpy()
GLOBAL_MEAN = _y.mean()
TE_KEYS = {
    "Origin": train["Origin"],
    "Dest": train["Dest"],
    "UniqueCarrier": train["UniqueCarrier"],
    "route": _route(train),
    "dep_hour": _dep_hour(train).astype(str),
    "Month": train["Month"],
    "DayOfWeek": train["DayOfWeek"],
    "carrier_hour": train["UniqueCarrier"].astype(str) + "_" + _dep_hour(train).astype(str),
}
FREQ_MAPS = {k: _freq_map(v) for k, v in TE_KEYS.items()}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here so predict_proba() reproduces it on unseen data."""
    X = df[feature_cols].copy()
    for c in obj_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # time of day: DepTime is hhmm, may exceed 2400 (next-day roll)
    v = X["DepTime"].astype("float64") % 2400
    hour = _dep_hour(df)
    minute = (v % 100).astype("int32")
    mins = hour * 60 + minute
    X["dep_hour"] = pd.Categorical(hour.astype(str), categories=[str(i) for i in range(24)])
    X["tod_sin"] = np.sin(2 * np.pi * mins / 1440)
    X["tod_cos"] = np.cos(2 * np.pi * mins / 1440)
    X["dep_minute"] = minute
    # calendar numerics + seasonality
    m = X["Month"].astype(str).map(MONTHS)
    d = X["DayofMonth"].astype(str).map(DOM)
    dow = _dow(df)
    X["doy"] = CUMDAYS[(m - 1).fillna(0).astype("int32").to_numpy()] + d
    X["doy_sin"] = np.sin(2 * np.pi * X["doy"] / 365)
    X["doy_cos"] = np.cos(2 * np.pi * X["doy"] / 365)
    X["month_num"] = m.astype("float64")
    X["dow_num"] = dow.astype("float64")
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    X["log_dist"] = np.log1p(X["Distance"].astype("float64"))
    # target encodings (maps fitted on train; unseen -> global mean)
    keys = {
        "Origin": df["Origin"],
        "Dest": df["Dest"],
        "UniqueCarrier": df["UniqueCarrier"],
        "route": _route(df),
        "dep_hour": hour.astype(str),
        "Month": df["Month"],
        "DayOfWeek": df["DayOfWeek"],
        "carrier_hour": df["UniqueCarrier"].astype(str) + "_" + hour.astype(str),
    }
    for k, kv in keys.items():
        X[f"freq_{k}"] = kv.astype(str).map(FREQ_MAPS[k]).fillna(0).astype("float64")
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# ensemble of diverse XGBoost members; diversity from seed/depth/subsample/colsample.
# One early-stopping fit (reference member) finds the round count R; every member is then
# trained once on the full data with rounds scaled by 0.05/lr.
MEMBER_PARAMS = [
    dict(learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42),
    dict(learning_rate=0.05, max_depth=7, subsample=0.7, colsample_bytree=0.7, random_state=7),
    dict(learning_rate=0.05, max_depth=5, subsample=0.9, colsample_bytree=0.9, random_state=2024),
    dict(learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=99,
         min_child_weight=10),
    dict(learning_rate=0.05, max_depth=8, subsample=0.75, colsample_bynode=0.6, random_state=31337),
    dict(learning_rate=0.05, max_depth=6, subsample=0.7, colsample_bytree=0.9, random_state=555,
         gamma=1.0),
    dict(learning_rate=0.05, max_depth=7, subsample=0.9, colsample_bytree=0.6, random_state=1234,
         min_child_weight=5),
    dict(learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.7, random_state=2025),
    dict(learning_rate=0.05, max_depth=6, subsample=0.85, colsample_bytree=0.85, random_state=31415,
         colsample_bynode=0.8),
]
BASE = dict(
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
    eval_metric="auc",
    early_stopping_rounds=100,
)

t0 = time.time()
X_all = prepare(train)
y_all = to_y(train)
X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.1, random_state=SEED, stratify=y_all)
ref = xgb.XGBClassifier(**{**BASE, **MEMBER_PARAMS[0], "n_estimators": 3000})
ref.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
R = int(ref.best_iteration) + 1
print(f"reference rounds R={R} val_auc={roc_auc_score(y_val, ref.predict_proba(X_val)[:, 1]):.4f}")
final_models = []
for cap in (0.25, 0.4, 0.7):
    for mp in MEMBER_PARAMS:
        n_est = max(50, round(R * 0.05 / mp["learning_rate"] * cap))
        m = xgb.XGBClassifier(**{**BASE, **mp, "n_estimators": n_est, "early_stopping_rounds": None})
        m.fit(X_all, y_all)
        final_models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Ps = [m.predict_proba(prepare(df))[:, 1] for m in final_models]
    return np.mean(Ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
