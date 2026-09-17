"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Design notes (see FINAL.md for the full story):
  * train.csv is 2005 and the scored data is 2006, so anything fitted to the *target* in 2005 decays.
    `Month` was dropped for that reason and target encodings were ablated entirely (equal AUC, less code).
  * What does transfer: the time-of-day curve and, from the *scheduled* (count-only) traffic structure, how
    busy an airport / carrier / route is and how its traffic is spread over the day. Those features are
    computed from feature counts, never from labels, so they carry no year-specific label noise.
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
DROP = ["Month", "DayofMonth"]                      # year-specific seasonality, does not transfer to 2006
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET] + DROP]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
num_cols = [c for c in feature_cols if c not in obj_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

TE_ROUTE_SEP = "_"
DENSITY_COLS = ["Origin", "Dest", "UniqueCarrier", "route"]


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + TE_ROUTE_SEP + df["Dest"].astype(str)


def _hour(df: pd.DataFrame) -> np.ndarray:
    return df["DepTime"].to_numpy() // 100


def _level(df: pd.DataFrame, name: str) -> np.ndarray:
    return {"Origin": df["Origin"], "Dest": df["Dest"], "UniqueCarrier": df["UniqueCarrier"],
            "route": _route(df)}[name].astype(str).to_numpy()


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def _density_maps(df: pd.DataFrame) -> dict:
    """Scheduled-traffic structure from feature counts only: share of an entity's flights leaving in each
    hour, and log1p entity size."""
    h = _hour(df)
    out = {}
    for name in DENSITY_COLS:
        lv = _level(df, name)
        cnt = pd.Series(1.0, index=pd.MultiIndex.from_arrays([lv, h])).groupby(level=[0, 1]).sum()
        tot = cnt.groupby(level=0).sum()
        share = cnt / tot.reindex(cnt.index.get_level_values(0)).to_numpy()
        out[("share", name)] = share.to_dict()
        out[("size", name)] = np.log1p(tot).to_dict()
    return out


def _density_features(df: pd.DataFrame, maps: dict) -> dict:
    h = _hour(df)
    feats = {}
    for name in DENSITY_COLS:
        idx = pd.MultiIndex.from_arrays([_level(df, name), h])
        for tag in ("share",):  # exp35: ablate size_* 
            m = maps[(tag, name)]
            key = idx if tag != "size" else _level(df, name)
            feats[f"{tag}_{name}"] = pd.Series(pd.Index(key).map(m), index=df.index).fillna(0.0).to_numpy()
    return feats


def _peak_maps(df: pd.DataFrame) -> dict:
    """Each entity's busiest hour of the day (argmax of its scheduled traffic). Used to express a flight's
    departure time relative to that entity's own daily rhythm - a scale-free congestion-phase feature."""
    h = _hour(df)
    out = {}
    for name in DENSITY_COLS:
        lv = _level(df, name)
        cnt = pd.Series(np.ones(len(df)), index=pd.MultiIndex.from_arrays([lv, h])).groupby(level=[0, 1]).sum()
        out[name] = {lvl: hour for lvl, (_lvl, hour) in cnt.groupby(level=0).idxmax().items()}
    return out


def _peak_features(df: pd.DataFrame, maps: dict) -> dict:
    h = _hour(df)
    feats = {}
    for name in DENSITY_COLS:
        peak = np.array([maps[name].get(k, np.nan) for k in _level(df, name)], dtype=float)
        feats[f"peak_off_{name}"] = np.where(np.isnan(peak), 0.0, h - peak)
    return feats


DENS_MAP = _density_maps(train)
PEAK_MAP = _peak_maps(train)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    # The density maps are fitted on train only and are target-free, so applying them to unseen rows is safe.
    X = df[cat_cols + num_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    for k, v in _density_features(df, DENS_MAP).items():
        X[k] = v
    for k, v in _peak_features(df, PEAK_MAP).items():
        X[k] = v
    return X


# --- model --------------------------------------------------------------------
# Seed-averaged ensemble: each member sees a different subsample/colsample draw; averaging cancels part
# of the variance that otherwise shows up as a year-specific fit.
N_MODELS = 32
COMMON = dict(
    n_estimators=30,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=0.7,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
X_train = prepare(train)

t0 = time.time()
models = []
for i in range(N_MODELS):
    m = xgb.XGBClassifier(random_state=SEED + i, **COMMON)
    m.fit(X_train, to_y(train))
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
