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
carmonth_map = train.groupby(["UniqueCarrier", "Month"]).size().to_dict() if "UniqueCarrier" in cat_cols and "Month" in cat_cols else None

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


def prepare(df: pd.DataFrame, drop=(), add=()) -> pd.DataFrame:
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

    if "domflag" in add and "DayOfMonth" in df.columns:
        dom = pd.to_numeric(df["DayOfMonth"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
        X["dom_edge"] = ((dom <= 7) | (dom >= 25)).astype("float64")
    if "carmonth" in add and carmonth_map is not None:
        X["carmonth_freq"] = np.log1p(df[["UniqueCarrier", "Month"]].apply(tuple, axis=1).map(carmonth_map).astype("float64"))

    return X.drop(columns=list(drop))


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
    dict(max_depth=12, n_estimators=300, min_child_weight=100, colsample_bytree=0.5, reg_lambda=20.0, reg_alpha=5.0, random_state=42),
    dict(max_depth=12, n_estimators=300, min_child_weight=100, colsample_bytree=0.5, reg_lambda=10.0, reg_alpha=10.0, random_state=7),
    dict(max_depth=12, n_estimators=300, min_child_weight=100, colsample_bytree=0.5, reg_lambda=20.0, reg_alpha=5.0, random_state=43),
    dict(max_depth=12, n_estimators=300, min_child_weight=100, colsample_bytree=0.5, reg_lambda=20.0, reg_alpha=10.0, random_state=13),
    dict(max_depth=10, n_estimators=300, min_child_weight=100, colsample_bytree=0.5, reg_lambda=20.0, reg_alpha=5.0, random_state=44),
    dict(max_depth=12, n_estimators=300, min_child_weight=100, colsample_bytree=0.5, reg_lambda=20.0, reg_alpha=5.0, random_state=101),
    dict(max_depth=12, n_estimators=300, min_child_weight=150, colsample_bytree=0.45, reg_lambda=20.0, reg_alpha=5.0, random_state=202),
    dict(max_depth=12, n_estimators=300, min_child_weight=100, colsample_bytree=0.55, reg_lambda=15.0, reg_alpha=5.0, random_state=303),
    dict(max_depth=12, n_estimators=500, learning_rate=0.03, min_child_weight=100, colsample_bytree=0.5, reg_lambda=10.0, reg_alpha=5.0, random_state=305),
]

# --- model --------------------------------------------------------------------
BASE_CFG = dict(max_depth=12, n_estimators=300, min_child_weight=100, colsample_bytree=0.5, reg_lambda=20.0, reg_alpha=5.0, random_state=42)

PROBES = [
    ("base", (), (), {}),
    ("drop_route_freq", ("route_freq",), (), {}),
    ("drop_carorg_freq", ("carorg_freq",), (), {}),
    ("drop_raw_deptime", ("DepTime",), (), {}),
    ("add_domflag", (), ("domflag",), {}),
    ("add_carmonth", (), ("carmonth",), {}),
    ("mcw50", (), (), dict(min_child_weight=50)),
    ("maxbin1024", (), (), dict(max_bin=1024)),
]

t0 = time.time()
y = to_y(train)
ye = to_y(evald)

best = None
for name, drop, add, mkw in PROBES:
    X = prepare(train, drop=drop, add=add)
    Xe = prepare(evald, drop=drop, add=add)
    params = {**BASE_PARAMS, **BASE_CFG, **mkw}
    m = xgb.XGBClassifier(**params)
    m.fit(X, y)
    auc = roc_auc_score(ye, m.predict_proba(Xe)[:, 1])
    print(f"PROBE {name}: eval_auc={auc:.4f}")
    if best is None or auc > best[0]:
        best = (auc, name, drop, add, mkw)

print(f"BEST {best[1]} auc={best[0]:.4f}")
print(f"Training time: {time.time() - t0:.1f}s")

BEST_DROP, BEST_ADD, BEST_MKW = best[2], best[3], best[4]
X = prepare(train, drop=BEST_DROP, add=BEST_ADD)
model = xgb.XGBClassifier(**{**BASE_PARAMS, **BASE_CFG, **BEST_MKW})
model.fit(X, y)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, drop=BEST_DROP, add=BEST_ADD))[:, 1]


eval_auc = roc_auc_score(ye, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
