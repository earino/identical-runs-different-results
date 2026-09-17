"""XGBoost binary classifier: freq encodings; PROBE round 2 (feature variants + blend).

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

ROUTE_LEVELS = sorted({f"{o}|{d}" for o, d in train[list(route_key)].drop_duplicates().to_numpy().tolist()}) if route_key else None
HOUR_LEVELS = pd.Index([f"h{h:02d}" for h in range(25)])
MONTH_LEVELS = pd.Index([f"m{m:02d}" for m in range(1, 13)])


def _extras(df: pd.DataFrame) -> pd.DataFrame:
    E = pd.DataFrame(index=df.index)
    if "DepTime" in df.columns:
        dt = pd.to_numeric(df["DepTime"], errors="coerce")
        hour = (dt // 100).clip(0, 24)
        E["dep_hour_cat"] = pd.Categorical(("h" + hour.astype("Int64").astype(str)).astype(str), categories=HOUR_LEVELS)
    if "Distance" in df.columns:
        E["log_distance"] = np.log1p(pd.to_numeric(df["Distance"], errors="coerce"))
    if "Month" in df.columns:
        mnum = pd.to_numeric(df["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
        ang = 2 * np.pi * (mnum - 1) / 12.0
        E["month_sin"] = np.sin(ang)
        E["month_cos"] = np.cos(ang)
        E["month_cat"] = pd.Categorical(("m" + mnum.astype("Int64").astype(str)).astype(str), categories=MONTH_LEVELS)
    if route_key:
        rt = df[route_key[0]].astype(str) + "|" + df[route_key[1]].astype(str)
        E["route_cat"] = pd.Categorical(rt, categories=ROUTE_LEVELS)
    if carorg_map is not None:
        E["carorg_freq"] = np.log1p(df[["UniqueCarrier", "Origin"]].apply(tuple, axis=1).map(carorg_map).astype("float64"))
    return E


def prepare(df: pd.DataFrame, extra_cols=("log_distance", "dep_hour_cat")) -> pd.DataFrame:
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
    for c in extra_cols:
        X[c] = E[c]
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


BASE_PARAMS = dict(
    learning_rate=0.05,
    max_depth=3,
    n_estimators=300,
    min_child_weight=1,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
    max_bin=512,
)

# --- model --------------------------------------------------------------------
t0 = time.time()
y = to_y(train)
ye = to_y(evald)

PROBES = [
    ("base(hourcat)", ("log_distance", "dep_hour_cat"), {}),
    ("month_sincos", ("log_distance", "dep_hour_cat", "month_sin", "month_cos"), {}),
    ("month_cat", ("log_distance", "dep_hour_cat", "month_cat"), {}),
    ("route_cat", ("log_distance", "dep_hour_cat", "route_cat"), {}),
    ("carorg_freq", ("log_distance", "dep_hour_cat", "carorg_freq"), {}),
    ("all_new", ("log_distance", "dep_hour_cat", "month_sin", "month_cos", "route_cat", "carorg_freq"), {}),
]
best = None
for name, extra_cols, param_kw in PROBES:
    X = prepare(train, extra_cols=extra_cols)
    Xe = prepare(evald, extra_cols=extra_cols)
    m = xgb.XGBClassifier(**{**BASE_PARAMS, **param_kw})
    m.fit(X, y)
    auc = roc_auc_score(ye, m.predict_proba(Xe)[:, 1])
    print(f"PROBE {name}: eval_auc={auc:.4f}")
    if best is None or auc > best[0]:
        best = (auc, name, extra_cols, param_kw)

print(f"BEST {best[1]} auc={best[0]:.4f}")

# blend probe: best shallow + deep-regularized model
Xb = prepare(train, extra_cols=best[2])
Xeb = prepare(evald, extra_cols=best[2])
m_shallow = xgb.XGBClassifier(**{**BASE_PARAMS, "random_state": 42})
m_shallow.fit(Xb, y)
m_deep = xgb.XGBClassifier(**{**BASE_PARAMS, "max_depth": 8, "min_child_weight": 20, "colsample_bytree": 0.7, "n_estimators": 150, "random_state": 43})
m_deep.fit(Xb, y)
p = 0.7 * m_shallow.predict_proba(Xeb)[:, 1] + 0.3 * m_deep.predict_proba(Xeb)[:, 1]
print(f"PROBE blend70/30: eval_auc={roc_auc_score(ye, p):.4f}")

print(f"Training time: {time.time() - t0:.1f}s")

EXTRA_COLS = best[2]
model = m_shallow


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, extra_cols=EXTRA_COLS))[:, 1]


eval_auc = roc_auc_score(ye, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
