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
# DayofMonth delay rates do not transfer across years (corr 0.33) -> drop it as noise.
DROP = ["DayofMonth", "Month"]
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET] + DROP]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# baseline: treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# --- schedule/congestion counts from TRAIN only (airport volume is stable across years) ---
_dep_h = (pd.to_numeric(train["DepTime"], errors="coerce") // 100).clip(lower=0, upper=23)
_origin = train["Origin"].astype(str)
_dest = train["Dest"].astype(str)
_route = _origin + "_" + _dest
def _arr_h(hour: pd.Series, dist: pd.Series) -> pd.Series:
    """Estimated arrival hour at the destination (450 mph cruise), wrapped to 0..23."""
    return ((hour + dist.clip(lower=0) / 450.0) % 24).astype(int).astype(str)


_CNT = {
    "cnt_origin": _origin.value_counts(),
    "cnt_dest": _dest.value_counts(),
    "cnt_route": _route.value_counts(),
    "cnt_origin_hour": (_origin + "_" + _dep_h.astype(str)).value_counts(),
    "cnt_dest_hour": (_dest + "_" + _dep_h.astype(str)).value_counts(),
    "cnt_dest_arrhour": (_dest + "_" + _arr_h(_dep_h, pd.to_numeric(train["Distance"], errors="coerce"))).value_counts(),
}


# departure-time distribution per origin: fraction of the day's flights that already left
_dep_num_tr = pd.to_numeric(train["DepTime"], errors="coerce").to_numpy()
_tp = pd.DataFrame({"o": _origin.to_numpy(), "t": _dep_num_tr}).dropna().sort_values(["o", "t"])
_SORTED_DEP = {o: np.sort(g.to_numpy()) for o, g in _tp.groupby("o")["t"]}

# estimated arrival-time distribution per destination (same idea, other end of the flight)
_dep_min_tr = (_dep_h * 60.0 + (pd.to_numeric(train["DepTime"], errors="coerce") % 100)).to_numpy()
_arr_min_tr = (_dep_min_tr + pd.to_numeric(train["Distance"], errors="coerce").fillna(0).to_numpy() / 450.0 * 60.0) % 1440.0
_tpa = pd.DataFrame({"d": _dest.to_numpy(), "t": _arr_min_tr}).dropna().sort_values(["d", "t"])
_SORTED_ARR = {d: np.sort(g.to_numpy()) for d, g in _tpa.groupby("d")["t"]}


def _dep_frac(origin: pd.Series, dep_num: np.ndarray) -> np.ndarray:
    out = np.full(len(origin), np.nan)
    tmp = pd.DataFrame({"o": origin.to_numpy(), "t": dep_num})
    for o, idx in tmp.groupby("o").groups.items():
        arr = _SORTED_DEP.get(o)
        if arr is None or len(arr) == 0:
            continue
        out[idx] = np.searchsorted(arr, tmp.loc[idx, "t"].to_numpy()) / len(arr)
    return out


def _frac_in_day(keys: pd.Series, vals: np.ndarray, table: dict) -> np.ndarray:
    """Rank of `vals` inside the reference distribution of `keys` (0 = first of the day, 1 = last)."""
    out = np.full(len(keys), np.nan)
    tmp = pd.DataFrame({"k": keys.to_numpy(), "t": vals})
    for k, idx in tmp.groupby("k").groups.items():
        arr = table.get(k)
        if arr is None or len(arr) == 0:
            continue
        out[idx] = np.searchsorted(arr, tmp.loc[idx, "t"].to_numpy()) / len(arr)
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(lower=0, upper=23)
    X["dep_hour"] = hour
    X["dep_min"] = dep % 100
    origin = df["Origin"].astype(str)
    dest = df["Dest"].astype(str)
    hour_s = hour.astype(int).astype(str)
    X["cnt_origin"] = np.log1p(origin.map(_CNT["cnt_origin"]).fillna(0).to_numpy())
    X["cnt_dest"] = np.log1p(dest.map(_CNT["cnt_dest"]).fillna(0).to_numpy())
    X["cnt_route"] = np.log1p((origin + "_" + dest).map(_CNT["cnt_route"]).fillna(0).to_numpy())
    X["cnt_origin_hour"] = np.log1p((origin + "_" + hour_s).map(_CNT["cnt_origin_hour"]).fillna(0).to_numpy())
    X["cnt_dest_hour"] = np.log1p((dest + "_" + hour_s).map(_CNT["cnt_dest_hour"]).fillna(0).to_numpy())
    X["dep_frac"] = _dep_frac(origin, dep.to_numpy(dtype=float))
    arr = _arr_h(hour, pd.to_numeric(df["Distance"], errors="coerce"))
    X["arr_hour"] = arr.astype(float)
    X["cnt_dest_arrhour"] = np.log1p((dest + "_" + arr).map(_CNT["cnt_dest_arrhour"]).fillna(0).to_numpy())
    # how busy the destination airport is departing flights at our estimated arrival hour
    X["cnt_dest_dep_at_arr"] = np.log1p((dest + "_" + arr).map(_CNT["cnt_origin_hour"]).fillna(0).to_numpy())
    arr_min = ((hour * 60.0 + (dep % 100)) + pd.to_numeric(df["Distance"], errors="coerce").fillna(0) / 450.0 * 60.0) % 1440.0
    X["dest_arr_frac"] = _frac_in_day(dest, arr_min.to_numpy(dtype=float), _SORTED_ARR)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: heterogeneous bagged ensemble of shallow XGBoost trees -------------
# Members vary depth / subsampling so their errors decorrelate (2005->2006 shift).
MEMBERS = [
    dict(max_depth=4, n_estimators=400, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=3, n_estimators=600, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=5, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7),
    dict(max_depth=6, n_estimators=300, learning_rate=0.03, subsample=0.7, colsample_bytree=0.7),
    dict(max_depth=4, n_estimators=400, learning_rate=0.05, subsample=0.7, colsample_bytree=1.0),
    dict(max_depth=3, n_estimators=600, learning_rate=0.05, subsample=0.9, colsample_bytree=0.6),
    dict(max_depth=5, n_estimators=300, learning_rate=0.04, subsample=0.7, colsample_bytree=0.6),
    dict(max_depth=4, n_estimators=400, learning_rate=0.05, subsample=0.9, colsample_bytree=0.6),
    dict(grow_policy="lossguide", max_leaves=32, n_estimators=500, learning_rate=0.04, subsample=0.8, colsample_bytree=0.7),
    dict(grow_policy="lossguide", max_leaves=24, n_estimators=600, learning_rate=0.04, subsample=0.7, colsample_bytree=0.6),
    dict(grow_policy="lossguide", max_leaves=48, n_estimators=400, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8),
    dict(grow_policy="lossguide", max_leaves=16, n_estimators=700, learning_rate=0.05, subsample=0.9, colsample_bytree=0.7),
    dict(grow_policy="lossguide", max_leaves=64, n_estimators=400, learning_rate=0.035, subsample=0.7, colsample_bytree=0.5),
    dict(grow_policy="lossguide", max_leaves=12, n_estimators=800, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
    dict(grow_policy="lossguide", max_leaves=40, n_estimators=500, learning_rate=0.03, subsample=0.6, colsample_bytree=0.5),
    dict(grow_policy="lossguide", max_leaves=96, n_estimators=300, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9),
]
SEEDS = [42, 7, 13, 99, 2024, 17, 23, 31, 5, 11, 37, 53, 71, 89, 101, 137]


def _make_model(seed: int, **kw) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
        **kw,
    )


t0 = time.time()
X_train = prepare(train)
y_train = to_y(train)
models = []
for _s, _kw in zip(SEEDS, MEMBERS):
    _m = _make_model(_s, **_kw)
    _m.fit(X_train, y_train)
    models.append(_m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} members)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
