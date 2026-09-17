"""XGBoost binary classifier: diverse ensemble (shallow+deep), freq/hour features.

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
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

FREQ_COLS = [c for c in ("UniqueCarrier", "Origin", "Dest") if c in cat_cols]
freq_maps = {c: train[c].value_counts() for c in FREQ_COLS}
route_key = ("Origin", "Dest") if "Origin" in cat_cols and "Dest" in cat_cols else None
if route_key:
    freq_maps["route"] = train.groupby(list(route_key)).size().to_dict()
carorg_map = train.groupby(["UniqueCarrier", "Origin"]).size().to_dict() if "UniqueCarrier" in cat_cols and "Origin" in cat_cols else None

HOUR_LEVELS = pd.Index([f"h{h:02d}" for h in range(25)])


def _extras(df: pd.DataFrame) -> pd.DataFrame:
    E = pd.DataFrame(index=df.index)
    if "DepTime" in df.columns:
        dt = pd.to_numeric(df["DepTime"], errors="coerce")
        hour = (dt // 100).clip(0, 24)
        E["dep_hour_cat"] = pd.Categorical(("h" + hour.astype("Int64").astype(str)).astype(str), categories=HOUR_LEVELS)
    if "Distance" in df.columns:
        E["log_distance"] = np.log1p(pd.to_numeric(df["Distance"], errors="coerce"))
    if carorg_map is not None:
        E["carorg_freq"] = np.log1p(df[["UniqueCarrier", "Origin"]].apply(tuple, axis=1).map(carorg_map).astype("float64"))
    return E


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN

    for c in FREQ_COLS:
        X[c + "_freq"] = np.log1p(df[c].map(freq_maps[c]).astype("float64"))
    if route_key:
        X["route_freq"] = np.log1p(df[list(route_key)].apply(tuple, axis=1).map(freq_maps["route"]).astype("float64"))

    E = _extras(df)
    for c in ("dep_hour_cat", "log_distance", "carorg_freq"):
        X[c] = E[c]
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


BASE_PARAMS = dict(
    learning_rate=0.05,
    min_child_weight=1,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
    eval_metric="auc",
    max_bin=512,
)

ENSEMBLE = [
    dict(max_depth=3, n_estimators=250, random_state=42),
    dict(max_depth=3, n_estimators=300, random_state=7, colsample_bytree=0.8, subsample=0.8),
    dict(max_depth=3, n_estimators=350, random_state=13, colsample_bytree=0.9, subsample=0.85),
    dict(max_depth=3, n_estimators=400, random_state=99),
    dict(max_depth=5, n_estimators=300, random_state=97, min_child_weight=5),
    dict(max_depth=8, n_estimators=150, random_state=43, min_child_weight=20, colsample_bytree=0.7),
    dict(max_depth=10, n_estimators=120, random_state=44, min_child_weight=50, colsample_bytree=0.6),
]

# --- deep-regularized single-model probes -------------------------------------
DEEP_PROBES = [
    dict(max_depth=12, n_estimators=300, min_child_weight=100, colsample_bytree=0.5, reg_lambda=10.0, reg_alpha=10.0, random_state=42),
    dict(max_depth=12, n_estimators=300, min_child_weight=100, colsample_bytree=0.5, reg_lambda=10.0, reg_alpha=20.0, random_state=42),
    dict(max_depth=12, n_estimators=300, min_child_weight=100, colsample_bytree=0.5, reg_lambda=20.0, reg_alpha=5.0, random_state=42),
    dict(max_depth=12, n_estimators=300, min_child_weight=100, colsample_bytree=0.5, reg_lambda=20.0, reg_alpha=10.0, random_state=42),
    dict(max_depth=12, n_estimators=300, min_child_weight=100, colsample_bytree=0.5, reg_lambda=5.0, reg_alpha=5.0, random_state=42),
    dict(max_depth=12, n_estimators=400, min_child_weight=100, colsample_bytree=0.5, reg_lambda=10.0, reg_alpha=5.0, random_state=42),
]

# --- model --------------------------------------------------------------------
t0 = time.time()
X = prepare(train)
y = to_y(train)
Xe = prepare(evald)
ye = to_y(evald)

P = []
ENS_MODELS = []
for i, cfg in enumerate(ENSEMBLE):
    m = xgb.XGBClassifier(**{**BASE_PARAMS, **cfg})
    m.fit(X, y)
    p = m.predict_proba(Xe)[:, 1]
    P.append(p)
    ENS_MODELS.append(m)
    print(f"model {i} {cfg}: eval_auc={roc_auc_score(ye, p):.4f}")

P = np.array(P)
W_DICT = {
    "uniform7": np.ones(7) / 7,
    "shallow_5_deep2": np.array([0.15, 0.15, 0.15, 0.15, 0.1, 0.15, 0.15]),
    "shallow_6_deep1": np.array([0.18, 0.18, 0.18, 0.18, 0.13, 0.15, 0.0]),
    "shallow4_deep1": np.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0]),
    "s_halves": np.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.1, 0.1]),
}
best_w, best_auc, best_name = None, -1, None
for name, w in W_DICT.items():
    auc = roc_auc_score(ye, w @ P)
    print(f"BLEND {name}: eval_auc={auc:.4f}")
    if auc > best_auc:
        best_w, best_auc, best_name = w, auc, name
print(f"BEST blend {best_name} auc={best_auc:.4f}")

for cfg in DEEP_PROBES:
    m = xgb.XGBClassifier(**{**BASE_PARAMS, **cfg})
    m.fit(X, y)
    auc = roc_auc_score(ye, m.predict_proba(Xe)[:, 1])
    print(f"DEEP {cfg}: eval_auc={auc:.4f}")

print(f"Training time: {time.time() - t0:.1f}s")

WEIGHTS = best_w


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    ps = np.array([m.predict_proba(Xp)[:, 1] for m in ENS_MODELS])
    return WEIGHTS @ ps


eval_auc = roc_auc_score(ye, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
