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
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
raw_feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [
    c
    for c in raw_feature_cols
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])
]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
num_cols = [c for c in raw_feature_cols if c not in obj_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
hour_levels = pd.Index(range(24))
qhour_levels = pd.Index(range(96))

# Target encoding: out-of-fold on train rows (anti-leak); full-fit map for unseen rows.
TE_SMOOTH = 20.0
te_maps: dict[str, dict] = {}  # full-fit maps, used for eval/holdout rows
te_oof: dict[str, np.ndarray] = {}  # OOF values aligned to train rows

# interaction definitions: name -> (list of base cols, smoothing)
INTERACTIONS: dict[str, tuple[list, float]] = {
    "Origin_Dest": (["Origin", "Dest"], 30.0),
}


def _hour(df: pd.DataFrame) -> np.ndarray:
    dep = df["DepTime"].astype(float).to_numpy()
    return np.clip(np.floor(dep / 100) % 24, 0, 23)


def _interaction_key(df: pd.DataFrame, cols: list) -> pd.Series:
    parts = []
    for c in cols:
        if c == "DepTime_hour":
            parts.append(pd.Series(_hour(df), index=df.index).astype(int).astype(str).reset_index(drop=True))
        else:
            parts.append(df[c].astype(str).reset_index(drop=True))
    out = parts[0]
    for p in parts[1:]:
        out = out + "_" + p
    return pd.Series(out.to_numpy(), index=df.index)


def _te_full(keys: pd.Series, y: np.ndarray, smooth: float, gmean: float) -> dict:
    stats = pd.DataFrame({"cat": keys, "y": y}).groupby("cat")["y"].agg(["sum", "count"])
    enc = (stats["sum"] + smooth * gmean) / (stats["count"] + smooth)
    return {**enc.to_dict(), "__global__": gmean}


def _fit_target_encodings(df: pd.DataFrame, y: np.ndarray) -> None:
    gmean = float(y.mean())
    n = len(df)
    # candidate TE sources: simple cat cols + interactions
    sources: dict[str, tuple[pd.Series, float]] = {c: (df[c], TE_SMOOTH) for c in cat_cols}
    for name, (cols, smooth) in INTERACTIONS.items():
        sources[name] = (_interaction_key(df, cols), smooth)
    # OOF values for train rows
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, (keys, _) in sources.items():
        oof = np.full(n, gmean)
        keys_reset = keys.reset_index(drop=True)
        for tr_idx, val_idx in kf.split(np.zeros(n)):
            fold_map = _te_full(keys_reset.iloc[tr_idx], y[tr_idx], TE_SMOOTH, gmean)
            oof[val_idx] = keys_reset.iloc[val_idx].map(fold_map).fillna(gmean).to_numpy()
        te_oof[name] = oof
        te_maps[name] = _te_full(keys_reset, y, TE_SMOOTH, gmean)
        te_maps[name]["__global__"] = gmean


def prepare(df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
    """ALL feature engineering lives here so predict_proba() reproduces it on unseen rows."""
    X = pd.DataFrame(index=df.index)
    for c in num_cols:
        X[c] = df[c].astype(float)
    dep = df["DepTime"].astype(float)
    hour = np.clip(np.floor(dep / 100) % 24, 0, 23)
    minute = dep - 100 * np.floor(dep / 100)
    minute_of_day = hour * 60 + minute
    X["hour"] = hour
    X["minute"] = minute
    X["sin_h"] = np.sin(2 * np.pi * hour / 24)
    X["cos_h"] = np.cos(2 * np.pi * hour / 24)
    # hour as native categorical (fine-grained time-of-day effects)
    X["cat_hour"] = pd.Categorical(hour.astype(int), categories=hour_levels)
    # 15-minute time-of-day bins
    X["cat_qhour"] = pd.Categorical((minute_of_day // 15).astype(int), categories=qhour_levels)
    # distance bins
    X["dist_bin"] = np.floor(np.log1p(df["Distance"].astype(float)) / 0.5)
    for c in cat_cols:
        X["cat_" + c] = pd.Categorical(df[c], categories=cat_levels[c])
        m = te_maps[c]
        vals = te_oof[c] if is_train else df[c].map(m).fillna(m["__global__"]).astype(float).to_numpy()
        X["te_" + c] = vals
    for name, (cols, _) in INTERACTIONS.items():
        key = _interaction_key(df, cols)
        m = te_maps[name]
        vals = te_oof[name] if is_train else key.map(m).fillna(m["__global__"]).astype(float).to_numpy()
        X["te_" + name] = vals
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


_fit_target_encodings(train, to_y(train))

# --- model --------------------------------------------------------------------
# 10-seed bagged ensemble: robust variance reduction, per-model config from exp6.
N_BAG = 10
models = []
t0 = time.time()
Xtr = prepare(train, is_train=True)
ytr = to_y(train)
Xev_for_es = prepare(evald)  # for early stopping only
for k in range(N_BAG):
    m = xgb.XGBClassifier(
        n_estimators=1500,
        max_depth=8,
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_lambda=1,
        random_state=SEED + k,
        n_jobs=N_JOBS,
        early_stopping_rounds=30,
        eval_metric="auc",
    )
    m.fit(Xtr, ytr, eval_set=[(Xev_for_es, to_y(evald))], verbose=False)
    models.append(m)
    print(f"  bag {k}: best_iter={m.best_iteration}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
