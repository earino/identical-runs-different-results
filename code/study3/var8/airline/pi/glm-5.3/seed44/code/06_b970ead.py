"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Learned so far (see experiments.tsv):
  - Categorical Month overfits year-specific noise; a numeric month generalizes better.
  - Depth 3-4, strong reg_lambda, ~100-200 rounds: best transfer to 2006.
  - Target encoding / route categorical memorize 2005 noise and hurt.
  - Schedule-position features from train only help: o_pct (origin dep-time ECDF), o_offset (vs origin mean),
    o_dist_mean / d_vs_origin (route length vs the origin's average).
  - Averaging diverse XGB configs helps a lot, including "weak but diverse" members: deep trees with big
    min_child_weight, a robust-feature member (no origin/dest/carrier), and an extended-feature member.
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
CAT_COLS = ["DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
ROBUST_CAT = ["DayofMonth", "DayOfWeek"]  # origin/dest/carrier effects shift year-to-year
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _mins(dep: pd.Series) -> np.ndarray:
    return (dep // 100).to_numpy() * 60 + (dep % 100).to_numpy()


# origin-schedule stats computed on TRAIN ONLY (30-minute-bin ECDF per origin + mean dep time + mean distance)
NB = 56  # bins cover 0..1650 minutes
_EDGES = np.arange(0, NB * 30, 30)
_o_bins = np.clip(np.searchsorted(_EDGES, _mins(train["DepTime"]), side="right") - 1, 0, NB - 2)
_o_cnt = pd.crosstab(train["Origin"], _o_bins).reindex(columns=range(NB - 1), fill_value=0)
_o_cum = _o_cnt.cumsum(axis=1).to_numpy()
_o_ecdf = (_o_cum - _o_cnt.to_numpy() / 2 + 0.5) / (_o_cum[:, -1:] + 1.0)
_o_pos = pd.Series(np.arange(len(_o_cnt)), index=_o_cnt.index)
_o_mean = pd.Series(_mins(train["DepTime"])).groupby(train["Origin"]).mean()
_o_dist_mean = train["Distance"].groupby(train["Origin"]).mean()
_train_dep_mean = float(_o_mean.mean())
_train_dist_mean = float(_o_dist_mean.mean())


def _binof(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(_EDGES, arr, side="right") - 1, 0, NB - 2)


# feature variants: "full" | "robust" (no origin/dest/carrier) | "odist" (full + route-length context)
def prepare(df: pd.DataFrame, variant: str = "full") -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in (ROBUST_CAT if variant == "robust" else CAT_COLS):
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    dm = _mins(df["DepTime"])
    X["DepTime"] = df["DepTime"].astype(int)
    X["Distance"] = df["Distance"].astype(float)
    # numeric month (c-1..c-12 -> 1..12): ordered splits generalize better than per-level splits
    X["month_n"] = df["Month"].astype(str).str.replace("c-", "", regex=False).astype(int)
    if variant != "robust":
        # schedule position within the origin's day (robust to the 2005 -> 2006 shift)
        idx = df["Origin"].map(_o_pos).fillna(-1).to_numpy().astype(int)
        X["o_pct"] = np.where(idx >= 0, _o_ecdf[np.maximum(idx, 0), _binof(dm)], 0.5)
        X["o_offset"] = dm - df["Origin"].map(_o_mean).fillna(_train_dep_mean).to_numpy()
    if variant == "odist":
        X["o_dist_mean"] = df["Origin"].map(_o_dist_mean).fillna(_train_dist_mean).to_numpy()
        X["d_vs_origin"] = X["Distance"].to_numpy() - X["o_dist_mean"].to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: ensemble of diverse XGB configs ------------------------------------
# (xgboost params, feature variant)
ENSEMBLE = [
    (dict(n_estimators=100, max_depth=3, learning_rate=0.1, min_child_weight=5, reg_lambda=30.0), "full"),
    (dict(n_estimators=200, max_depth=3, learning_rate=0.05, min_child_weight=5, reg_lambda=30.0), "full"),
    (dict(n_estimators=150, max_depth=4, learning_rate=0.05, min_child_weight=5, reg_lambda=30.0), "full"),
    (dict(n_estimators=150, max_depth=4, learning_rate=0.1, min_child_weight=5, reg_lambda=30.0,
          subsample=0.8, colsample_bytree=0.8), "full"),
    (dict(n_estimators=100, max_depth=5, learning_rate=0.1, min_child_weight=10, reg_lambda=50.0), "full"),
    (dict(n_estimators=200, max_depth=3, learning_rate=0.05, min_child_weight=5, reg_lambda=30.0), "robust"),
    (dict(n_estimators=300, max_depth=6, learning_rate=0.05, min_child_weight=50, reg_lambda=30.0), "full"),
    (dict(n_estimators=300, max_depth=8, learning_rate=0.05, min_child_weight=100, reg_lambda=30.0), "full"),
    (dict(n_estimators=200, max_depth=3, learning_rate=0.05, min_child_weight=5, reg_lambda=30.0), "odist"),
]

t0 = time.time()
y_train = to_y(train)
X_cache = {v: prepare(train, v) for v in ("full", "robust", "odist")}
models = []
for cfg, variant in ENSEMBLE:
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED,
                          n_jobs=N_JOBS, **cfg)
    m.fit(X_cache[variant], y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xs = {}
    out = np.zeros(len(df))
    for (cfg, variant), m in zip(ENSEMBLE, models):
        if variant not in Xs:
            Xs[variant] = prepare(df, variant)
        out += m.predict_proba(Xs[variant])[:, 1]
    return out / len(models)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
