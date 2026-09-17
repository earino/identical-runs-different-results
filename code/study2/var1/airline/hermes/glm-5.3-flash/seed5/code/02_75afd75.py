"""XGBoost binary classifier for airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Design notes (experiments 1-3 + scratch):
  - Raw categoricals via enable_categorical plateau ~0.71-0.715; larger/deeper models overfit the 2005
    snapshot (2006 conditional rates genuinely shifted, e.g. carrier delay rates move year to year).
  - Shrunk target encodings (m=100 toward the 0.5 base rate) are much stronger: compact, regularized
    representations of the categorical levels. Maps are fit on train.csv only and reused verbatim inside
    predict_proba, so train rows are encoded with full-train maps (self-encoding) — consistent with how
    unseen data is encoded at predict time, which measured better than out-of-fold encoding here.
  - Loss-guided (leaf-wise) growth with ~128 leaves beats depth-6..10 hist trees.
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
NUM_COLS = ["DepTime", "Distance", "dephour"]
TE_M = 100
BASE = 0.5

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
_y = (train[TARGET] == POSITIVE).astype(int)


def _add_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    dt = X["DepTime"].astype("int64")
    X["DepTime"] = dt.clip(upper=2359)      # 2400-2620 are next-day red-eyes -> treat as 0-220
    X["dephour"] = (dt // 100).clip(upper=23)
    return X


# encoders fit on training data only
_train_x = _add_time_feats(train)
_yS = pd.Series(_y.to_numpy(), index=_train_x.index)
TE_MAPS = {}
for c in CAT_COLS:
    g = pd.DataFrame({"k": _train_x[c].astype(str).values, "v": _yS.values}).groupby("k").v.agg(["sum", "size"])
    TE_MAPS[c] = ((g["sum"] + TE_M * BASE) / (g["size"] + TE_M)).to_dict()

FEATURES = NUM_COLS + [c + "_te" for c in CAT_COLS]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = _add_time_feats(df)
    for c in CAT_COLS:
        X[c + "_te"] = X[c].astype(str).map(TE_MAPS[c]).fillna(BASE)
    return X[FEATURES]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=0,
    max_leaves=128,
    grow_policy="lossguide",
    min_child_weight=1.0,
    subsample=0.95,
    colsample_bytree=0.75,
    tree_method="hist",
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), verbose=False)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
