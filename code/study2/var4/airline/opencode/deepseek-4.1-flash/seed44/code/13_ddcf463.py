"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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


def _dep_hour(s: pd.Series) -> pd.Series:
    return (pd.to_numeric(s, errors="coerce") // 100).clip(0, 23)


def _dist_band(s: pd.Series) -> pd.Series:
    return pd.cut(pd.to_numeric(s, errors="coerce"), bins=[0, 300, 600, 1000, 1500, 2500, 6000],
                  labels=False).astype("Int64").astype(str)


def _combos(df: pd.DataFrame) -> pd.DataFrame:
    h = _dep_hour(df["DepTime"]).astype("Int64").astype(str)
    return pd.DataFrame({
        "CarrierHour": df["UniqueCarrier"].astype(str) + "_" + h,
        "OrigHour": df["Origin"].astype(str) + "_" + h,
        "DestHour": df["Dest"].astype(str) + "_" + h,
        "CarrierOrig": df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str),
        "DistBandHour": _dist_band(df["Distance"]) + "_" + h,
    })


_combo_cols = ["CarrierHour", "OrigHour", "DestHour", "CarrierOrig", "DistBandHour"]
_combo_train = _combos(train)
_combo_levels = {c: pd.Index(sorted(_combo_train[c].unique())) for c in _combo_cols}
_combo_freq = {c: _combo_train[c].value_counts() for c in _combo_cols}

# frequency maps fit on training data only
_route_tr = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
_freq_maps = {
    "OriginFreq": train["Origin"].value_counts(),
    "DestFreq": train["Dest"].value_counts(),
    "CarrierFreq": train["UniqueCarrier"].value_counts(),
    "RouteFreq": _route_tr.value_counts(),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["DepHour"] = pd.Categorical(_dep_hour(df["DepTime"]), categories=list(range(24)))
    X["DepMin"] = (pd.to_numeric(df["DepTime"], errors="coerce") % 100).astype(float)
    X["IsWeekend"] = df["DayOfWeek"].isin(["c-6", "c-7"]).astype(int)
    combos = _combos(df)
    for c in _combo_cols:
        X[c] = pd.Categorical(combos[c], categories=_combo_levels[c])
        X[c + "Freq"] = combos[c].map(_combo_freq[c]).fillna(0)
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["OriginFreq"] = df["Origin"].map(_freq_maps["OriginFreq"]).fillna(0)
    X["DestFreq"] = df["Dest"].map(_freq_maps["DestFreq"]).fillna(0)
    X["CarrierFreq"] = df["UniqueCarrier"].map(_freq_maps["CarrierFreq"]).fillna(0)
    X["RouteFreq"] = route.map(_freq_maps["RouteFreq"]).fillna(0)
    X["OrigHourShare"] = (X["OrigHourFreq"] / X["OriginFreq"].replace(0, np.nan)).fillna(0)
    X["DestHourShare"] = (X["DestHourFreq"] / X["DestFreq"].replace(0, np.nan)).fillna(0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Ensemble of diverse, individually-regularized XGBoost configs; averaging their
# probabilities is markedly more stable than any single model on the year shift.
CONFIGS = [
    dict(n_estimators=600, max_depth=8, learning_rate=0.03, reg_alpha=5.0, reg_lambda=0.0,
         gamma=1.0, colsample_bytree=0.7, max_cat_threshold=16, max_cat_to_onehot=1),
    dict(n_estimators=600, max_depth=8, learning_rate=0.07, reg_alpha=4.0, reg_lambda=0.0,
         gamma=0.0, colsample_bytree=0.7, max_cat_threshold=64, max_cat_to_onehot=4),
    dict(n_estimators=1000, max_depth=7, learning_rate=0.07, reg_alpha=6.0, reg_lambda=2.0,
         gamma=0.2, colsample_bytree=0.7, max_cat_threshold=16, max_cat_to_onehot=16),
    dict(n_estimators=1500, max_depth=7, min_child_weight=4, learning_rate=0.03, reg_alpha=3.0,
         reg_lambda=10.0, gamma=0.5, colsample_bytree=0.7, max_cat_threshold=128, max_cat_to_onehot=4),
    dict(n_estimators=800, max_depth=8, min_child_weight=1, learning_rate=0.03, reg_alpha=3.0,
         reg_lambda=5.0, gamma=0.0, colsample_bytree=0.85, max_cat_threshold=128, max_cat_to_onehot=4),
    dict(n_estimators=600, max_depth=8, min_child_weight=4, learning_rate=0.03, reg_alpha=5.0,
         reg_lambda=0.0, gamma=0.0, colsample_bytree=0.7, max_cat_threshold=128, max_cat_to_onehot=1),
    dict(n_estimators=800, max_depth=6, min_child_weight=4, learning_rate=0.05, reg_alpha=3.0,
         reg_lambda=2.0, gamma=0.0, colsample_bytree=0.7, max_cat_threshold=128, max_cat_to_onehot=4),
]
COMMON = dict(subsample=1.0, min_child_weight=1, tree_method="hist",
              enable_categorical=True, n_jobs=N_JOBS)

X_train = prepare(train)
y_train = to_y(train)

models = []
t0 = time.time()
for k, cfg in enumerate(CONFIGS):
    m = xgb.XGBClassifier(random_state=SEED + k, **{**COMMON, **cfg})
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
