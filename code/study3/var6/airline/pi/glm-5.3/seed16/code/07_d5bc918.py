"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- fitted-on-train statistics/levels (module constants, applied inside prepare) ---
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in ["UniqueCarrier", "Origin", "Dest"]}
ROUTE_LEVELS = pd.Index(sorted((train["Origin"] + "_" + train["Dest"]).dropna().unique()))

# --- hierarchical out-of-fold target encodings, fitted on TRAIN only --------------
RAW_COLS = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
PRIOR = float((train[TARGET] == POSITIVE).mean())
N_FOLDS = 5
_y_tr = (train[TARGET] == POSITIVE).astype(int)


def _hour_str(d):
    return ((d["DepTime"] % 2400) // 100).astype(str)


def _todq_str(d):
    return ((d["DepTime"] % 2400) // 15).astype(str)


def _route_str(d):
    return d["Origin"].astype(str) + "_" + d["Dest"].astype(str)


# (name, key_fn, parent_name_or_None, shrink_K)
TE_SPECS = [
    ("te_route", _route_str, None, 25.0),
    ("te_carrier", lambda d: d["UniqueCarrier"].astype(str), None, 25.0),
    ("te_origin", lambda d: d["Origin"].astype(str), None, 25.0),
    ("te_dest", lambda d: d["Dest"].astype(str), None, 25.0),
    ("te_carrier_hour", lambda d: d["UniqueCarrier"].astype(str) + "_" + _hour_str(d), "te_carrier", 10.0),
    ("te_route_hour", lambda d: _route_str(d) + "_" + _hour_str(d), "te_route", 10.0),
    ("te_carrier_todq", lambda d: d["UniqueCarrier"].astype(str) + "_" + _todq_str(d), "te_carrier", 5.0),
    ("te_origin_todq", lambda d: d["Origin"].astype(str) + "_" + _todq_str(d), "te_origin", 5.0),
    ("te_dest_todq", lambda d: d["Dest"].astype(str) + "_" + _todq_str(d), "te_dest", 5.0),
    ("te_dow_hour", lambda d: d["DayOfWeek"].astype(str) + "_" + _hour_str(d), None, 5.0),
    ("te_month_hour", lambda d: d["Month"].astype(str) + "_" + _hour_str(d), None, 10.0),
]


def _smoothed_te(keys: np.ndarray, y: np.ndarray, base_vals: np.ndarray, k: float):
    """keys -> smoothed mean, shrunk toward per-row parent base values."""
    g = pd.DataFrame({"k": keys, "y": y, "b": base_vals}).groupby("k").agg(
        {"y": ["sum", "count"], "b": "mean"})
    cnt = g[("y", "count")].to_numpy()
    sm = g[("y", "sum")].to_numpy()
    b = g[("b", "mean")].to_numpy()
    return pd.Series((sm + k * b) / (cnt + k), index=g.index)


def _row_base(d: pd.DataFrame, parent, maps: dict) -> np.ndarray:
    """Per-row base value: parent TE map applied to parent keys (PRIOR if unseen)."""
    if parent is None:
        return np.full(len(d), PRIOR)
    pk = pd.Series(kf_parent(parent, d)).map(maps[parent]).to_numpy()
    return np.where(pd.isna(pk), PRIOR, pk)


def _te_maps(src_idx: np.ndarray) -> dict:
    """Compute all TE maps from train rows in src_idx (parent features first)."""
    d = train.iloc[src_idx]
    y = _y_tr.to_numpy()[src_idx]
    maps = {}
    for name, kf, parent, k in TE_SPECS:
        keys = kf(d).to_numpy()
        base = _row_base(d, parent, maps)
        maps[name] = _smoothed_te(keys, y, base, k)
    return maps


def _apply_maps(dst_idx: np.ndarray, maps: dict) -> dict:
    """Evaluate TE maps on train rows in dst_idx; unseen keys fall back to parent, then PRIOR."""
    d = train.iloc[dst_idx]
    out = {}
    for name, kf, parent, _ in TE_SPECS:
        keys = kf(d).to_numpy()
        v = pd.Series(keys).map(maps[name]).to_numpy()
        if parent:
            pkeys = kf_parent(parent, d)
            v = np.where(pd.isna(v), pd.Series(pkeys).map(maps[parent]).to_numpy(), v)
        out[name] = np.where(pd.isna(v), PRIOR, v)
    return out


def kf_parent(parent: str, d: pd.DataFrame) -> np.ndarray:
    for name, kf, _, _ in TE_SPECS:
        if name == parent:
            return kf(d).to_numpy()
    raise KeyError(parent)


rng = np.random.RandomState(SEED)
_fold_id = rng.randint(0, N_FOLDS, len(train))
_tr_hash = pd.util.hash_pandas_object(train[RAW_COLS], index=False)
_oof = {name: np.full(len(train), PRIOR) for name, _, _, _ in TE_SPECS}
for f in range(N_FOLDS):
    dst = np.where(_fold_id == f)[0]
    src = np.where(_fold_id != f)[0]
    maps = _te_maps(src)
    vals = _apply_maps(dst, maps)
    for name in vals:
        _oof[name][dst] = vals[name]
_full_maps = _te_maps(np.arange(len(train)))
_oof_lookup = {n: pd.Series(v, index=_tr_hash).groupby(level=0).mean() for n, v in _oof.items()}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(np.float64)
    t = dep % 2400.0                      # 2400+ -> after-midnight times
    hour = (t // 100).astype(np.float64)
    minute = t - hour * 100
    tod = hour + minute / 60.0            # time of day in hours (float)
    X["DepTime"] = dep
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)
    month = df["Month"].astype(str).str.slice(start=2).astype(np.int16).astype(np.float64)
    X["month"] = month
    X["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    X["day"] = df["DayofMonth"].astype(str).str.slice(start=2).astype(np.int16).astype(np.float64)
    X["dow"] = df["DayOfWeek"].astype(str).str.slice(start=2).astype(np.int16).astype(np.float64)
    dist = df["Distance"].astype(np.float64)
    X["Distance"] = dist
    X["log_dist"] = np.log1p(dist)
    # native categoricals
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"], categories=cat_levels["UniqueCarrier"])
    X["Origin"] = pd.Categorical(df["Origin"], categories=cat_levels["Origin"])
    X["Dest"] = pd.Categorical(df["Dest"], categories=cat_levels["Dest"])
    X["route"] = pd.Categorical(_route_str(df), categories=ROUTE_LEVELS)
    # OOF target encodings (train rows: out-of-fold; unseen rows: full-train maps with parent fallback)
    h = pd.util.hash_pandas_object(df[RAW_COLS], index=False)
    for name, kf, parent, _ in TE_SPECS:
        v = h.map(_oof_lookup[name])
        if v.isna().any():
            keys = pd.Series(kf(df).to_numpy())
            fv = keys.map(_full_maps[name]).to_numpy()
            if parent:
                pk = pd.Series(kf_parent(parent, df)).map(_full_maps[parent]).to_numpy()
                fv = np.where(pd.isna(fv), pk, fv)
            fv = np.where(pd.isna(fv), PRIOR, fv)
            v = v.fillna(pd.Series(fv, index=df.index))
        X[name] = v.to_numpy(dtype=float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=3000,
    max_depth=8,
    learning_rate=0.05,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    early_stopping_rounds=300,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=100)
print(f"Training time: {time.time() - t0:.1f}s, best iters: {model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
imp = pd.Series(model.feature_importances_, index=prepare(train).columns).sort_values(ascending=False)
print("feature importance (gain):")
for k, v in imp.items():
    print(f"  {k}: {v:.4f}")
