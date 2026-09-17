"""XGBoost binary classifier for airline delay prediction. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives in prepare(), fit on training data only.
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

# --- feature bookkeeping ------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
# DepTime is replaced by decomposed time-of-day features
RAW_DROP = ["DepTime"]
feature_cols = [c for c in feature_cols if c not in RAW_DROP]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _dep_hour(df: pd.DataFrame) -> pd.Series:
    return (pd.to_numeric(df["DepTime"], errors="coerce") // 100).astype("Int64")


def _keys(df: pd.DataFrame, name: str) -> pd.Series:
    """Grouping key for a delay-rate (target) encoding."""
    if name == "CarrierHour":
        return df["UniqueCarrier"].astype(str) + "_" + _dep_hour(df).astype(str)
    if name == "OriginHour":
        return df["Origin"].astype(str) + "_" + _dep_hour(df).astype(str)
    if name == "DestHour":
        return df["Dest"].astype(str) + "_" + _dep_hour(df).astype(str)
    return df[name]


# --- smoothed delay-rate encodings, fit on the training slice only -------------
PRIOR = float((train[TARGET] == POSITIVE).mean())
_y = (train[TARGET] == POSITIVE).astype(float)
# only group cells whose delay propensity is plausibly stable across years get an encoding
TE_SPEC = {"CarrierHour": 300.0, "OriginHour": 500.0, "DestHour": 500.0}


def _te_map(keys: pd.Series, alpha: float) -> pd.Series:
    g = _y.groupby(keys).agg(["sum", "count"])
    return (g["sum"] + PRIOR * alpha) / (g["count"] + alpha)


te_maps = {name: _te_map(_keys(train, name), a) for name, a in TE_SPEC.items()}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN

    # scheduled departure time-of-day, decomposed (hhmm integer -> minutes since midnight)
    t = pd.to_numeric(df["DepTime"], errors="coerce")
    mins = (t // 100) * 60 + (t % 100)
    X["dep_min"] = mins
    X["dep_hour"] = t // 100
    X["dep_minute"] = t % 100
    X["dep_sin"] = np.sin(2 * np.pi * mins / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * mins / 1440.0)

    for name, m in te_maps.items():
        X[f"{name}_rate"] = _keys(df, name).map(m).fillna(PRIOR).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def make_model(n_estimators: int, early: bool = False, **cfg) -> xgb.XGBClassifier:
    params = dict(
        max_depth=6,
        learning_rate=0.02,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    params.update(cfg)
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        early_stopping_rounds=50 if early else None,
        **params,
    )


# --- model: an ensemble of depth-diverse XGBoost models, each early-stopped ----
t0 = time.time()
X_all = prepare(train)
y_all = to_y(train)
Xf, Xv, yf, yv = train_test_split(X_all, y_all, test_size=0.15, random_state=SEED, stratify=y_all)

RATE_COLS = [f"{name}_rate" for name in TE_SPEC]

CONFIGS = [
    dict(max_depth=2, min_child_weight=1),
    dict(max_depth=3, min_child_weight=2),
    dict(max_depth=4),
    dict(max_depth=5),
    dict(max_depth=6),
    dict(max_depth=7, min_child_weight=4),
    dict(max_depth=8, min_child_weight=8),
    # two extra members see a different feature view (no delay-rate encodings)
    dict(max_depth=4, drop_rates=True),
    dict(max_depth=6, drop_rates=True),
]

models = []
for cfg in CONFIGS:
    cfg = dict(cfg)
    drop = cfg.pop("drop_rates", False)
    if drop:
        Xa, Xf_i, Xv_i = X_all.drop(columns=RATE_COLS), Xf.drop(columns=RATE_COLS), Xv.drop(columns=RATE_COLS)
    else:
        Xa, Xf_i, Xv_i = X_all, Xf, Xv
    probe = make_model(1200, early=True, **cfg)
    probe.fit(Xf_i, yf, eval_set=[(Xv_i, yv)])
    n_rounds = int(probe.best_iteration) + 1
    m = make_model(n_rounds, **cfg)
    m.fit(Xa, y_all)
    models.append((m, drop))
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models, rounds={[m.n_estimators for m, _ in models]})")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    Xr = X.drop(columns=RATE_COLS)
    return np.mean([m.predict_proba(Xr if drop else X)[:, 1] for m, drop in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
