"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

exp10: schedule-density features. Target-rate encodings of Origin/Dest/Carrier decay across the year
boundary, but the *schedule* does not: how much of an airport's traffic is packed into a given hour is a
stable proxy for congestion, and hub size is a stable proxy for how much a carrier's delay propagates. Both
are computed from feature counts only (no target), so they carry no year-specific label noise.
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
DROP = ["Month"]                      # year-specific seasonality, does not transfer to 2006
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET] + DROP]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
num_cols = [c for c in feature_cols if c not in obj_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

TE_ROUTE_SEP = "_"
DENSITY_COLS = ["Origin", "Dest", "UniqueCarrier", "route"]


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + TE_ROUTE_SEP + df["Dest"].astype(str)


def _key(df: pd.DataFrame, name: str) -> pd.Series:
    return {"carrier": df["UniqueCarrier"].astype(str), "origin": df["Origin"].astype(str),
            "dest": df["Dest"].astype(str), "route": _route(df)}[name]


def _hour(df: pd.DataFrame) -> np.ndarray:
    return df["DepTime"].to_numpy() // 100


def _level(df: pd.DataFrame, name: str) -> np.ndarray:
    return {"Origin": df["Origin"], "Dest": df["Dest"], "UniqueCarrier": df["UniqueCarrier"],
            "route": _route(df)}[name].astype(str).to_numpy()


# (feature name, key column, smoothing weight on the prior); plain smoothed target encodings
TE_SPECS = []  # exp14: ablate target encodings - with the raw categoricals and the density/size features
# already present, the smoothed year-2005 target means may be redundant (and carry year-specific noise).


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


y_train = to_y(train)
PRIOR = float(y_train.mean())


def _encode(y: np.ndarray, keys: pd.Series, smooth: float) -> dict:
    """Smoothed mean target per level: (sum_y + prior * smooth) / (n + smooth)."""
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return ((g["sum"] + PRIOR * smooth) / (g["count"] + smooth)).to_dict()


def _density_maps(df: pd.DataFrame) -> dict:
    """Share of an entity's flights that leave in each hour + log1p entity size. Feature counts only."""
    h = _hour(df)
    out = {}
    for name in DENSITY_COLS:
        lv = _level(df, name)
        cnt = pd.Series(1.0, index=pd.MultiIndex.from_arrays([lv, h])).groupby(level=[0, 1]).sum()
        tot = cnt.groupby(level=0).sum()
        out[("share", name)] = (cnt / tot.reindex(cnt.index.get_level_values(0)).to_numpy()).to_dict()
        out[("size", name)] = np.log1p(tot).to_dict()
    return out


def _density_features(df: pd.DataFrame, maps: dict) -> dict:
    h = _hour(df)
    feats = {}
    for name in DENSITY_COLS:
        lv = _level(df, name)
        idx = pd.MultiIndex.from_arrays([lv, h])
        feats[f"dens_{name}"] = pd.Series(idx.map(maps[("share", name)]), index=df.index).fillna(0.0).to_numpy()
        feats[f"size_{name}"] = pd.Series(lv, index=df.index).map(maps[("size", name)]).fillna(0.0).to_numpy()
    return feats


# full-train maps -> used by the live prediction path
TE_MAP = {feat: _encode(y_train, _key(train, name), smooth) for feat, name, smooth in TE_SPECS}
DENS_MAP = _density_maps(train)

# out-of-fold target-encoding maps -> used only to build the training matrix
N_FOLD = 5
_fold = np.random.RandomState(SEED).randint(0, N_FOLD, size=len(train))
OOF = {}
for f in range(N_FOLD):
    m = _fold != f
    for feat, name, smooth in TE_SPECS:
        mp = _encode(y_train[m], _key(train[m], name), smooth)
        OOF.setdefault(feat, np.full(len(train), np.nan))[~m] = _key(train, name)[~m].map(mp).to_numpy()
OOF = {k: np.nan_to_num(v, nan=PRIOR) for k, v in OOF.items()}


def prepare(df: pd.DataFrame, oof: bool = False) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[cat_cols + num_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    for feat, name, _ in TE_SPECS:
        if oof:
            X[feat] = OOF[feat]
        else:
            X[feat] = _key(df, name).map(TE_MAP[feat]).fillna(PRIOR).to_numpy()
    for k, v in _density_features(df, DENS_MAP).items():
        X[k] = v
    return X


# --- model --------------------------------------------------------------------
# Seed-averaged ensemble: each member sees a different subsample/colsample draw; averaging cancels part
# of the variance that otherwise shows up as a year-specific fit.
N_MODELS = 8
COMMON = dict(
    n_estimators=30,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
X_train = prepare(train, oof=True)

t0 = time.time()
models = []
for i in range(N_MODELS):
    m = xgb.XGBClassifier(random_state=SEED + i, **COMMON)
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
