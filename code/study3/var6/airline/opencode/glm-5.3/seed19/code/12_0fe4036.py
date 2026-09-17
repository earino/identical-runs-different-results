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


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    # calendar / clock numerics
    for c, name in (("Month", "mon"), ("DayofMonth", "dom"), ("DayOfWeek", "dow")):
        X[name] = pd.to_numeric(X[c].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dep = pd.to_numeric(X["DepTime"], errors="coerce")
    X["hour"] = dep // 100
    X["minute"] = dep % 100
    X["blk"] = dep // 300  # 3-hour block of the day
    hf = X["hour"] + X["minute"] / 60.0
    X["sin_t"] = np.sin(2 * np.pi * hf / 24.0)
    X["cos_t"] = np.cos(2 * np.pi * hf / 24.0)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- interaction target encoding: stats fit on TRAIN only ------------------------
y_tr = to_y(train)
PRIOR = float(y_tr.mean())
M = 10.0
_bf_te = base_features(train)
_o2 = _bf_te["Origin"].astype(str).to_numpy()
_d2 = _bf_te["Dest"].astype(str).to_numpy()
_c2 = _bf_te["UniqueCarrier"].astype(str).to_numpy()
_h2 = _bf_te["blk"].astype("Int64").astype(str).to_numpy()
TE_KEYS = {
    "te_org_blk": np.char.add(np.char.add(_o2, "|"), _h2),
    "te_dest_blk": np.char.add(np.char.add(_d2, "|"), _h2),
    "te_car_blk": np.char.add(np.char.add(_c2, "|"), _h2),
}
rng = np.random.RandomState(SEED)
folds = np.array_split(rng.permutation(len(train)), 5)
oof = {k: np.zeros(len(train)) for k in TE_KEYS}
te_maps = {}
for k, keys in TE_KEYS.items():
    for f in folds:
        mask = np.ones(len(train), dtype=bool)
        mask[f] = False
        stats = pd.DataFrame({"k": keys[mask], "y": y_tr[mask]}).groupby("k")["y"].agg(["sum", "count"])
        te = (stats["sum"] + PRIOR * M) / (stats["count"] + M)
        oof[k][f] = pd.Series(keys[f]).map(te).fillna(PRIOR).to_numpy()
    full = pd.DataFrame({"k": keys, "y": y_tr}).groupby("k")["y"].agg(["sum", "count"])
    te_maps[k] = (full["sum"] + PRIOR * M) / (full["count"] + M)


def add_te(X: pd.DataFrame) -> pd.DataFrame:
    o = X["Origin"].astype(str).to_numpy()
    d = X["Dest"].astype(str).to_numpy()
    c = X["UniqueCarrier"].astype(str).to_numpy()
    h = X["blk"].astype("Int64").astype(str).to_numpy()
    keys = {
        "te_org_blk": np.char.add(np.char.add(o, "|"), h),
        "te_dest_blk": np.char.add(np.char.add(d, "|"), h),
        "te_car_blk": np.char.add(np.char.add(c, "|"), h),
    }
    for k, kk in keys.items():
        X[k] = pd.Series(kk).map(te_maps[k]).fillna(PRIOR).to_numpy()
    return X


# --- volume/congestion features: counts fit on TRAIN only ------------------------
_bf_tr = base_features(train)
_o = _bf_tr["Origin"].astype(str).to_numpy()
_d = _bf_tr["Dest"].astype(str).to_numpy()
_c = _bf_tr["UniqueCarrier"].astype(str).to_numpy()
_h = _bf_tr["blk"].astype("Int64").astype(str).to_numpy()
_dow = _bf_tr["dow"].astype("Int64").astype(str).to_numpy()

VOL_SPECS = {
    "vol_org_blk": (_o, _h),
    "vol_dest_blk": (_d, _h),
    "vol_org_dow": (_o, _dow),
    "vol_car_blk": (_c, _h),
}
vol_maps = {}
vol_defaults = {}
for k, (a, b) in VOL_SPECS.items():
    cnt = pd.Series(list(zip(a, b))).value_counts()
    vol_maps[k] = cnt
    vol_defaults[k] = float(cnt.mean())


def add_vol(X: pd.DataFrame) -> pd.DataFrame:
    o = X["Origin"].astype(str).to_numpy()
    d = X["Dest"].astype(str).to_numpy()
    c = X["UniqueCarrier"].astype(str).to_numpy()
    h = X["blk"].astype("Int64").astype(str).to_numpy()
    dow = X["dow"].astype("Int64").astype(str).to_numpy()
    keys = {"vol_org_blk": (o, h), "vol_dest_blk": (d, h), "vol_org_dow": (o, dow), "vol_car_blk": (c, h)}
    for k, (a, b) in keys.items():
        X[k] = np.log1p(pd.Series(list(zip(a, b))).map(vol_maps[k]).fillna(vol_defaults[k]).to_numpy())
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    return add_vol(add_te(base_features(df)))


# --- model: bagged ensemble of XGBoost models ----------------------------------
SMOKE = bool(os.environ.get("SMOKE"))
VOL_COLS = list(VOL_SPECS.keys())
TE_COLS = list(TE_KEYS.keys())
CAL_COLS = ["Month", "DayofMonth", "DayOfWeek"]  # native calendar cats = 2005-specific noise
MEMBERS = [
    dict(seed=42, leaves=48, mcw=10, drop=CAL_COLS),
    dict(seed=7, leaves=64, mcw=10, drop=CAL_COLS),
    dict(seed=123, leaves=96, mcw=20, drop=CAL_COLS + ["DepTime"]),
]
t0 = time.time()
X_train = add_vol(add_te(base_features(train)))
for k in TE_KEYS:
    X_train[k] = oof[k]
y_train = to_y(train)
X_eval, y_eval = prepare(evald), to_y(evald)
if SMOKE:
    X_train, y_train = X_train.head(4000), y_train[:4000]
    X_eval, y_eval = X_eval.head(4000), y_eval[:4000]
models = []
for cfg in MEMBERS:
    cols = [c for c in X_train.columns if c not in cfg["drop"]]
    m = xgb.XGBClassifier(
        n_estimators=15 if SMOKE else 2000,
        max_depth=20,
        max_leaves=cfg["leaves"],
        grow_policy="lossguide",
        min_child_weight=cfg["mcw"],
        learning_rate=0.03,
        tree_method="hist",
        enable_categorical=True,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=3 if SMOKE else 200,
        eval_metric="auc",
        random_state=cfg["seed"],
        n_jobs=N_JOBS,
    )
    m.fit(X_train[cols], y_train, eval_set=[(X_eval[cols], y_eval)], verbose=False)
    models.append((m, cols))
    print(f"  member seed={cfg['seed']} leaves={cfg['leaves']} best_iter={m.best_iteration} ({time.time() - t0:.1f}s)")
model = models  # ensemble container


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp[cols])[:, 1] for m, cols in models], axis=0)


if SMOKE:
    _probe = evald.head(500).drop(columns=[TARGET])
    _p = predict_proba(_probe)
    assert _p.shape == (len(_probe),) and np.isfinite(_p).all()
    print(f"SMOKE OK auc={roc_auc_score(y_eval[:500], _p):.4f}")
    raise SystemExit(0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
