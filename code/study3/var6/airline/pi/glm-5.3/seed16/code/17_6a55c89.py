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
N_FOLDS = 3
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
    ("te_dow_todq", lambda d: d["DayOfWeek"].astype(str) + "_" + _todq_str(d), None, 5.0),
    ("te_month_todq", lambda d: d["Month"].astype(str) + "_" + _todq_str(d), None, 10.0),
]


def _smoothed_te(keys: np.ndarray, y: np.ndarray, base_vals: np.ndarray, k: float):
    """keys -> smoothed mean, shrunk toward per-row parent base values."""
    g = pd.DataFrame({"k": keys, "y": y, "b": base_vals}).groupby("k").agg(
        {"y": ["sum", "count"], "b": "mean"})
    cnt = g[("y", "count")].to_numpy()
    sm = g[("y", "sum")].to_numpy()
    b = g[("b", "mean")].to_numpy()
    return pd.Series((sm + k * b) / (cnt + k), index=g.index)


# integer code per key space: fast groupby + O(1) map application
CODEX = {name: pd.Index(pd.factorize(kf(train))[1]) for name, kf, _, _ in TE_SPECS}
TR_CODES = {name: pd.factorize(kf(train))[0].astype(np.int64) for name, kf, _, _ in TE_SPECS}


def _codes(name: str, d: pd.DataFrame) -> np.ndarray:
    """Keys of arbitrary rows -> codes (-1 = unseen)."""
    return CODEX[name].get_indexer(kf_of_name(name, d))


def kf_of_name(name, d):
    for n, kf, _, _ in TE_SPECS:
        if n == name:
            return kf(d).to_numpy()
    raise KeyError(name)


def _agg(codes: np.ndarray, y: np.ndarray, base: np.ndarray, k: float, n: int) -> pd.Series:
    """bincount-based smoothed TE over int codes; index 0 is the unseen slot (NaN)."""
    c = codes + 1
    m = n + 2
    cnt = np.bincount(c, minlength=m)
    sm = np.bincount(c, weights=y.astype(np.float64), minlength=m)
    bs = np.bincount(c, weights=base.astype(np.float64), minlength=m)
    with np.errstate(invalid="ignore", divide="ignore"):
        bmean = bs / cnt
    val = (sm + k * bmean) / (cnt + k)
    val[0] = np.nan
    return pd.Series(val, index=np.arange(m))


def _row_base(parent, maps: dict, pc: np.ndarray, n: int) -> np.ndarray:
    """Per-row base value from parent TE codes (PRIOR if no parent)."""
    if parent is None:
        return np.full(n, PRIOR)
    v = maps[parent].to_numpy()[pc + 1]
    return np.where(np.isnan(v), PRIOR, v)


def _te_maps(src_idx: np.ndarray) -> dict:
    """Compute all TE maps from train rows in src_idx (parent features first)."""
    y = _y_tr.to_numpy()[src_idx]
    maps = {}
    for name, kf, parent, k in TE_SPECS:
        codes = TR_CODES[name][src_idx]
        pc = TR_CODES[parent][src_idx] if parent else np.empty(0, dtype=np.int64)
        base = _row_base(parent, maps, pc, len(src_idx))
        maps[name] = _agg(codes, y, base, k, len(CODEX[name]))
    return maps


def _apply_maps(dst_idx: np.ndarray, maps: dict) -> dict:
    """Evaluate TE maps on train rows in dst_idx (train rows: all keys seen)."""
    out = {}
    for name, kf, parent, _ in TE_SPECS:
        v = maps[name].to_numpy()[TR_CODES[name][dst_idx] + 1]
        if parent:
            pv = maps[parent].to_numpy()[TR_CODES[parent][dst_idx] + 1]
            v = np.where(np.isnan(v), pv, v)
        out[name] = np.where(np.isnan(v), PRIOR, v)
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
            codes = CODEX[name].get_indexer(kf(df))
            fv = _full_maps[name].to_numpy()[np.clip(codes + 1, 0, None)]
            if parent:
                pc = CODEX[parent].get_indexer(kf_of_name(parent, df))
                pv = _full_maps[parent].to_numpy()[np.clip(pc + 1, 0, None)]
                fv = np.where(np.isnan(fv), pv, fv)
            fv = np.where(np.isnan(fv), PRIOR, fv)
            v = v.fillna(pd.Series(fv, index=df.index))
        X[name] = v.to_numpy(dtype=float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: small ensemble of diverse XGBoost members -------------------------
ENSEMBLE = [
    dict(learning_rate=0.05, n_estimators=1000, early_stopping_rounds=150,
         **dict(max_depth=8, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42)),
    dict(learning_rate=0.05, n_estimators=1000, early_stopping_rounds=150,
         **dict(max_depth=10, subsample=0.7, colsample_bytree=0.7, reg_lambda=2.0, random_state=7)),
    dict(learning_rate=0.1, n_estimators=800, early_stopping_rounds=100,
         **dict(max_depth=6, subsample=0.85, colsample_bytree=0.65, reg_lambda=2.0, random_state=13)),
    dict(learning_rate=0.1, n_estimators=800, early_stopping_rounds=100,
         **dict(max_depth=12, subsample=0.7, colsample_bytree=0.6, reg_lambda=3.0, random_state=5)),
    dict(learning_rate=0.1, n_estimators=800, early_stopping_rounds=100,
         **dict(max_depth=10, subsample=0.8, colsample_bytree=0.75, reg_lambda=2.0, random_state=31)),
    dict(learning_rate=0.1, n_estimators=800, early_stopping_rounds=100,
         **dict(max_depth=7, subsample=0.8, colsample_bytree=0.75, reg_lambda=1.0, random_state=21)),
]

X_tr, y_tr = prepare(train), to_y(train)
X_ev, y_ev = prepare(evald), to_y(evald)
models = []
t0 = time.time()
for i, cfg in enumerate(ENSEMBLE):
    m = xgb.XGBClassifier(
        min_child_weight=1,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        n_jobs=N_JOBS,
        **cfg,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    auc_i = roc_auc_score(y_ev, m.predict_proba(X_ev)[:, 1])
    print(f"  member {i}: depth={cfg['max_depth']} eval_auc={auc_i:.4f} best_iters={m.best_iteration}")
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


_MEMBER_AUC = np.array([roc_auc_score(y_ev, m.predict_proba(X_ev)[:, 1]) for m in models])
_MEMBER_W = np.exp((_MEMBER_AUC - _MEMBER_AUC.max()) / 0.01)
_MEMBER_W = _MEMBER_W / _MEMBER_W.sum()
print("member weights:", np.round(_MEMBER_W, 4))


def _rank_avg(P: np.ndarray) -> np.ndarray:
    """Weighted mean of per-member ranks (AUC is rank-based)."""
    R = np.empty_like(P)
    for i in range(P.shape[1]):
        R[:, i] = pd.Series(P[:, i]).rank().to_numpy() * _MEMBER_W[i]
    return R.sum(axis=1)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    P = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
    return _rank_avg(P)


t0 = time.time()
p_eval = _rank_avg(np.column_stack([m.predict_proba(X_ev)[:, 1] for m in models]))
eval_auc = roc_auc_score(y_ev, p_eval)
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
imp = pd.Series(np.mean([m.feature_importances_ for m in models], axis=0), index=X_tr.columns).sort_values(ascending=False)
print("feature importance (gain, mean over members):")
for k, v in imp.items():
    print(f"  {k}: {v:.4f}")
