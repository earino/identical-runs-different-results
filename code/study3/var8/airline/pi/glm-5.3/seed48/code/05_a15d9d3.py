"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

yall = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(yall.mean())
TE_M = 25.0
N_FOLDS = 5


def _keyfns(d: pd.DataFrame):
    dep = d["DepTime"]
    hour = dep // 100
    return {
        "te_carrier": d["UniqueCarrier"].astype(str),
        "te_origin": d["Origin"].astype(str),
        "te_dest": d["Dest"].astype(str),
        "te_route": d["Origin"].astype(str) + "_" + d["Dest"].astype(str),
        "te_hour": hour.astype(str),
        "te_hour_origin": hour.astype(str) + "_" + d["Origin"].astype(str),
        "te_hour_dest": hour.astype(str) + "_" + d["Dest"].astype(str),
        "te_hour_carrier": hour.astype(str) + "_" + d["UniqueCarrier"].astype(str),
        "te_dow_origin": d["DayOfWeek"].astype(str) + "_" + d["Origin"].astype(str),
    }


SIMPLE = ["te_carrier", "te_origin", "te_dest", "te_route", "te_hour"]
INTER = ["te_hour_origin", "te_hour_dest", "te_hour_carrier", "te_dow_origin"]

# target-encoding tables: fit on train ONLY; train rows get out-of-fold values
_te_maps = {}      # name -> dict for new data
_te_oof = {}       # name -> array aligned to train rows


def _te_map(keys: pd.Series, y: np.ndarray, m: float) -> dict:
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    te = (g["sum"] + m * PRIOR) / (g["count"] + m)
    return te.to_dict()


def _fit_target_encodings(specs):
    keys_train = _keyfns(train)
    rng = np.random.RandomState(SEED)
    fold_id = rng.randint(0, N_FOLDS, size=len(train))
    for name in specs:
        keys = keys_train[name]
        _te_maps[name] = _te_map(keys, yall, TE_M)
        oof = np.empty(len(train))
        for f in range(N_FOLDS):
            mask = fold_id == f
            fm = _te_map(keys[~mask], yall[~mask], TE_M)
            oof[mask] = keys[mask].map(fm).fillna(PRIOR).to_numpy()
        _te_oof[name] = oof


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dep = df["DepTime"]
    hour = dep // 100
    X["hour"] = hour
    X["minute"] = dep % 100
    X["tod"] = hour * 60 + dep % 100
    return X


def prepare(df: pd.DataFrame, te_specs=SIMPLE + INTER) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = base_features(df)
    keys = _keyfns(df)
    for name in te_specs:
        X[name] = keys[name].map(_te_maps[name]).fillna(PRIOR)
    return X


def prepare_train(te_specs=SIMPLE + INTER) -> pd.DataFrame:
    """Training matrix: same features but TE columns use out-of-fold values (no label leak)."""
    X = base_features(train)
    for name in te_specs:
        X[name] = _te_oof[name]
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
yev = to_y(evald)
results = []
t0 = time.time()
_fit_target_encodings(SIMPLE + INTER)
print(f"TE fit time: {time.time() - t0:.1f}s", flush=True)

for name, specs in [
    ("base", []),
    ("simple", SIMPLE),
    ("inter", INTER),
    ("simple+inter", SIMPLE + INTER),
]:
    Xtr, Xev = prepare_train(specs), prepare(evald, specs)
    m = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, yall)
    auc = roc_auc_score(yev, m.predict_proba(Xev)[:, 1])
    results.append((auc, m, specs))
    print(f"diag feat={name} auc={auc:.4f} t={time.time() - t0:.1f}s", flush=True)

best_auc, model, best_specs = max(results, key=lambda r: r[0])
print(f"diag BEST specs={best_specs} auc={best_auc:.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, best_specs))[:, 1]


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
