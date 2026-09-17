"""XGBoost binary classifier on the airline delay dataset.

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
from sklearn.model_selection import KFold

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
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
GLOBAL_MEAN = float(y_train.mean())


# --- target encoding (fit on train only) --------------------------------------
def k_hour(df):
    return pd.to_numeric(df["DepTime"], errors="coerce") // 100


def k_origin(df):
    return df["Origin"].astype(str)


def k_dest(df):
    return df["Dest"].astype(str)


def k_carrier(df):
    return df["UniqueCarrier"].astype(str)


def _pair(a, b):
    return a.astype(str) + "|" + b.astype(str)


TE_SPECS = {  # name -> (key function, smoothing m)
    "te_origin": (k_origin, 50.0),
    "te_dest": (k_dest, 50.0),
    "te_route": (lambda df: _pair(k_origin(df), k_dest(df)), 150.0),
    "te_carrier": (k_carrier, 30.0),
    "te_hour": (k_hour, 30.0),
    "te_dow": (lambda df: df["DayOfWeek"].astype(str), 20.0),
    "te_month": (lambda df: df["Month"].astype(str), 20.0),
    "te_origin_hour": (lambda df: _pair(k_origin(df), k_hour(df)), 200.0),
    "te_dest_hour": (lambda df: _pair(k_dest(df), k_hour(df)), 200.0),
}


def te_fit(keys: pd.Series, y: np.ndarray, m: float, prior: float) -> pd.Series:
    tmp = pd.DataFrame({"k": keys.values, "y": y})
    g = tmp.groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + m * prior) / (g["count"] + m)


TE_MAPS = {name: te_fit(keyfn(train), y_train, m, GLOBAL_MEAN) for name, (keyfn, m) in TE_SPECS.items()}
FREQ_MAPS = {c: train[c].astype(str).value_counts() for c in ["Origin", "Dest", "UniqueCarrier", "Route"] if c != "Route"}
FREQ_MAPS["Route"] = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here so predict_proba() reproduces it on unseen rows."""
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    for name, mapping in TE_MAPS.items():
        keyfn = TE_SPECS[name][0]
        X[name] = keyfn(df).map(mapping).fillna(GLOBAL_MEAN).astype(float)
    X["freq_route"] = np.log1p(
        (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).map(FREQ_MAPS["Route"]).fillna(0)
    )
    for c in ["Origin", "Dest", "UniqueCarrier"]:
        X[f"freq_{c}"] = np.log1p(df[c].astype(str).map(FREQ_MAPS[c]).fillna(0))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- training matrix with out-of-fold target encoding (anti-leakage) ----------
X_all = prepare(train)
y_all = to_y(train)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for tr_idx, va_idx in kf.split(train):
    y_tr_f = y_train[tr_idx]
    prior_f = float(y_tr_f.mean())
    for name, (keyfn, m) in TE_SPECS.items():
        map_f = te_fit(keyfn(train.iloc[tr_idx]), y_tr_f, m, prior_f)
        X_all.iloc[va_idx, X_all.columns.get_loc(name)] = (
            keyfn(train.iloc[va_idx]).map(map_f).fillna(prior_f).to_numpy()
        )

RAW_VIEW = [c for c in feature_cols if c not in ("DayofMonth", "Month", "DayOfWeek")] + ["freq_Origin", "freq_Dest", "freq_UniqueCarrier", "freq_route"]  # pruned raw + hubness
TE_VIEW = list(TE_SPECS.keys()) + ["freq_Origin", "freq_Dest", "freq_UniqueCarrier", "freq_route"]
NUM_VIEW = list(TE_SPECS.keys()) + ["freq_Origin", "freq_Dest", "freq_UniqueCarrier", "freq_route", "DepTime", "Distance"]
ALL_VIEW = RAW_VIEW + list(TE_SPECS.keys())

# --- model --------------------------------------------------------------------
# heterogeneous ensemble: members differ in hyperparams AND feature view
MEMBERS = [
    (dict(n_estimators=30, max_depth=6, colsample_bynode=1.0), "raw"),
    (dict(n_estimators=30, max_depth=6, colsample_bynode=0.7), "raw"),
    (dict(n_estimators=60, max_depth=6, colsample_bynode=0.7), "raw"),
    (dict(n_estimators=30, max_depth=8, colsample_bynode=0.7), "raw"),
    (dict(n_estimators=30, max_depth=6, colsample_bynode=1.0), "te"),
    (dict(n_estimators=30, max_depth=6, colsample_bynode=0.7), "te"),
    (dict(n_estimators=60, max_depth=6, colsample_bynode=0.7), "te"),
    (dict(n_estimators=30, max_depth=8, colsample_bynode=0.7), "te"),
    (dict(n_estimators=30, max_depth=6, colsample_bynode=1.0), "all"),
    (dict(n_estimators=30, max_depth=6, colsample_bynode=0.7), "all"),
    (dict(n_estimators=60, max_depth=8, colsample_bynode=0.7), "all"),
    (dict(n_estimators=100, max_depth=6, learning_rate=0.05, colsample_bynode=0.7), "all"),
    (dict(n_estimators=30, max_depth=6, colsample_bynode=1.0), "num"),
    (dict(n_estimators=60, max_depth=8, colsample_bynode=0.7), "num"),
    (dict(n_estimators=300, max_depth=8, learning_rate=0.05, colsample_bynode=0.7), "all"),
    (dict(n_estimators=300, max_depth=8, learning_rate=0.05, colsample_bynode=0.7), "te"),
    (dict(n_estimators=300, max_depth=10, learning_rate=0.05, colsample_bynode=0.7), "all"),
    (dict(n_estimators=300, max_depth=10, learning_rate=0.05, colsample_bynode=0.7), "te"),
    (dict(n_estimators=500, max_depth=8, learning_rate=0.03, colsample_bynode=0.7), "all"),
    (dict(n_estimators=500, max_depth=8, learning_rate=0.03, colsample_bynode=0.7), "te"),
    (dict(n_estimators=500, max_depth=10, learning_rate=0.03, colsample_bynode=0.7), "all"),
    (dict(n_estimators=500, max_depth=10, learning_rate=0.03, colsample_bynode=0.7), "te"),
    (dict(n_estimators=300, max_depth=8, learning_rate=0.05, colsample_bynode=0.7), "all"),
    (dict(n_estimators=300, max_depth=8, learning_rate=0.05, colsample_bynode=0.7), "te"),
    (dict(n_estimators=500, max_depth=8, learning_rate=0.03, colsample_bynode=0.7), "all"),
    (dict(n_estimators=500, max_depth=10, learning_rate=0.03, colsample_bynode=0.7), "te"),
]
VIEW_COLS = {"raw": RAW_VIEW, "te": TE_VIEW, "num": NUM_VIEW, "all": ALL_VIEW}
models = []
t0 = time.time()
# temporal drift adaptation: upweight late-2005 rows (closer to the 2006 eval period)
_month_num = pd.to_numeric(train["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce").to_numpy()
sample_w = 0.5 + 1.5 * (_month_num - 1) / 11.0  # Jan->Dec: 0.5 -> 2.0
for i, (cfg, view) in enumerate(MEMBERS):
    rng = np.random.RandomState(SEED + i)
    idx = rng.choice(len(X_all), size=int(0.8 * len(X_all)), replace=False)
    cols = VIEW_COLS[view]
    m = xgb.XGBClassifier(
        **{
            "learning_rate": 0.1,
            "tree_method": "hist",
            "enable_categorical": True,
            "max_bin": 512,
            "random_state": SEED + i,
            "n_jobs": N_JOBS,
            **cfg,
        }
    )
    m.fit(X_all.iloc[idx][cols], y_all[idx], sample_weight=sample_w[idx])
    models.append((m, cols))
print(f"Ensemble training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X[cols])[:, 1] for m, cols in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
