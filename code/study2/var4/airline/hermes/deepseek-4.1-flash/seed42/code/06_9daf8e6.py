"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
  3. All feature engineering lives in prepare(); every statistic it uses is fit on data/train.csv only.
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
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

Y_TRAIN = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(Y_TRAIN.mean())


def add_keys(df: pd.DataFrame) -> dict:
    """Raw -> string interaction keys used for target encoding (no train statistics involved)."""
    dep = df["DepTime"].astype(float).to_numpy()
    dep = np.where(dep >= 2400, dep - 2400.0, dep)
    hour = np.floor(dep / 100.0).astype(int).astype(str)
    origin = df["Origin"].astype(str)
    carrier = df["UniqueCarrier"].astype(str)
    return {
        "Origin": origin,
        "Dest": df["Dest"].astype(str),
        "UniqueCarrier": carrier,
        "Origin_hour": origin + "|" + hour,
        "Carrier_hour": carrier + "|" + hour,
        "Route": origin + "|" + df["Dest"].astype(str),
    }


# --- target encoding: smoothed means, fit on TRAIN ONLY ------------------------
TE_SPEC = {"Origin": 50.0, "Dest": 50.0, "UniqueCarrier": 50.0,
           "Origin_hour": 100.0, "Carrier_hour": 100.0, "Route": 100.0}
_KEYS_TRAIN = add_keys(train)


def _te_stats(keys, y, k):
    s = pd.Series(y, index=keys.index).groupby(keys.to_numpy()).agg(["sum", "count"])
    return ((s["sum"] + PRIOR * k) / (s["count"] + k)).to_dict()


te_maps, oof_te = {}, {}
for _c, _k in TE_SPEC.items():
    te_maps[_c] = _te_stats(_KEYS_TRAIN[_c], Y_TRAIN, _k)
    _oof = np.full(len(train), PRIOR)
    for _tr, _va in KFold(n_splits=5, shuffle=True, random_state=SEED).split(train):
        _m = _te_stats(_KEYS_TRAIN[_c].iloc[_tr], Y_TRAIN[_tr], _k)
        _oof[_va] = _KEYS_TRAIN[_c].iloc[_va].map(_m).fillna(PRIOR).to_numpy()
    oof_te[_c] = _oof


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    keys = add_keys(df)
    for c in TE_SPEC:
        X[f"te_{c}"] = keys[c].map(te_maps[c]).fillna(PRIOR).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Xtr, ytr = prepare(train), to_y(train)
for c in TE_SPEC:  # train rows use out-of-fold encodings
    Xtr[f"te_{c}"] = oof_te[c]
Xev, yev = prepare(evald), to_y(evald)

# --- DIAGNOSTIC: which encoding block earns its place (3-seed averages) --------
PARAMS = dict(max_depth=4, learning_rate=0.03, n_estimators=300, tree_method="hist",
              enable_categorical=True, n_jobs=N_JOBS)
CANDIDATES = {
    "none (raw only)": [],
    "basic TE": ["Origin", "Dest", "UniqueCarrier"],
    "+Origin_hour": ["Origin", "Dest", "UniqueCarrier", "Origin_hour"],
    "+Carrier_hour": ["Origin", "Dest", "UniqueCarrier", "Carrier_hour"],
    "+both hours": ["Origin", "Dest", "UniqueCarrier", "Origin_hour", "Carrier_hour"],
    "Route TE": ["Route"],
}
for _label, _keep in CANDIDATES.items():
    _drop = [f"te_{c}" for c in TE_SPEC if c not in _keep]
    _A, _B = Xtr.drop(columns=_drop), Xev.drop(columns=_drop)
    _aucs = []
    for _s in (42, 43, 44):
        _m = xgb.XGBClassifier(random_state=_s, **PARAMS)
        _m.fit(_A, ytr)
        _aucs.append(roc_auc_score(yev, _m.predict_proba(_B)[:, 1]))
    print(f"diag {_label:16s} mean_auc={np.mean(_aucs):.4f}  per_seed={[round(a, 4) for a in _aucs]}")

# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(random_state=SEED, **PARAMS)
t0 = time.time()
model.fit(Xtr, ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
