"""XGBoost binary classifier for airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Design notes (experiments 1-10 + scratch):
  - Raw categoricals via enable_categorical plateau ~0.71-0.715; larger/deeper models overfit the 2005
    snapshot (2006 conditional rates genuinely shifted, e.g. carrier delay rates move year to year).
  - Shrunk target encodings (m=100 toward the 0.5 base rate) are much stronger: compact, regularized
    representations of the categorical levels. Maps are fit on train.csv only and reused verbatim inside
    predict_proba, so train rows are encoded with full-train maps (self-encoding) — consistent with how
    unseen data is encoded at predict time, which measured better than out-of-fold encoding here.
  - Joint target encodings carrier x hourbin, origin x hourbin and carrier x dephour capture interaction
    effects (+~0.005 total). Heavier shrinkage (m=300-500) suits the joint cells.
  - DepTime // 5 (5-minute slot) as a numeric feature helps (+~0.001).
  - Route frequency (train count of Origin_Dest pair) is a useful popularity prior (+~0.002); a route
    target encoding itself hurt (-0.013) and is not used.
  - Loss-guided (leaf-wise) growth, low learning rate (0.03), many rounds (~3000), max_bin=512.
  - 4-seed ensemble (seeds 42/0/2/3) averages away fit-order noise: +0.002 over a single seed.
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
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
NUM_COLS = ["DepTime", "Distance", "dephour", "qslot"]
JOINTS = [("UniqueCarrier", "hourbin"), ("Origin", "hourbin"), ("UniqueCarrier", "dephour")]
JOINT_MS = [300, 300, 500]
JOINT_NAMES = [a + "_" + b for a, b in JOINTS]
TE_M = 100
BASE = 0.5
HOUR_BINS = [-1, 5, 9, 12, 16, 19, 23]
ENSEMBLE_SEEDS = (42, 0, 2)

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
_y = (train[TARGET] == POSITIVE).astype(int)


def _base_feats(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    dt = X["DepTime"].astype("int64")
    X["DepTime"] = dt.clip(upper=2359)      # 2400-2620 are next-day red-eyes -> treat as 0-220
    X["dephour"] = (dt // 100).clip(upper=23)
    X["qslot"] = dt.clip(upper=2359) // 5
    X["hourbin"] = pd.cut(X["dephour"], bins=HOUR_BINS, labels=False).astype(float)
    X["route"] = X["Origin"].astype(str) + "_" + X["Dest"].astype(str)
    for name in JOINT_NAMES:
        a, b = name.split("_", 1)
        X[name] = X[a].astype(str) + "_" + X[b].astype(str)
    return X


def _te_map(fitX: pd.DataFrame, y: pd.Series, col: str, m: float) -> dict:
    g = pd.DataFrame({"k": fitX[col].astype(str).values, "v": y.values}).groupby("k").v.agg(["sum", "size"])
    return ((g["sum"] + m * BASE) / (g["size"] + m)).to_dict()


# encoders fit on training data only
_fitX = _base_feats(train)
_yS = pd.Series(_y.to_numpy(), index=_fitX.index)
TE_MAPS = {c: _te_map(_fitX, _yS, c, TE_M) for c in CAT_COLS}
TE_MAPS_JOINT = {name: _te_map(_fitX, _yS, name, m) for name, m in zip(JOINT_NAMES, JOINT_MS)}
ROUTE_COUNTS = _fitX["route"].value_counts().to_dict()

FEATURES = (
    NUM_COLS
    + [c + "_te" for c in CAT_COLS]
    + [n + "_te" for n in JOINT_NAMES]
    + ["route_cnt"]
)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = _base_feats(df)
    for c in CAT_COLS:
        X[c + "_te"] = X[c].astype(str).map(TE_MAPS[c]).fillna(BASE)
    for name in JOINT_NAMES:
        X[name + "_te"] = X[name].map(TE_MAPS_JOINT[name]).fillna(BASE)
    X["route_cnt"] = X["route"].map(ROUTE_COUNTS).fillna(0).astype(float)
    return X[FEATURES]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=0,
    max_leaves=128,
    grow_policy="lossguide",
    min_child_weight=1.0,
    subsample=0.95,
    colsample_bytree=0.75,
    max_bin=512,
    tree_method="hist",
    eval_metric="auc",
    n_jobs=N_JOBS,
)

t0 = time.time()
models = []
for seed in ENSEMBLE_SEEDS:
    m = xgb.XGBClassifier(random_state=seed, **PARAMS)
    m.fit(prepare(train), to_y(train), verbose=False)
    models.append(m)
print(f"Training time ({len(models)} models): {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    preds = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(preds, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
