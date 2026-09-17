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
N_FOLDS = 5


def _keyfns(d: pd.DataFrame) -> dict:
    dep = d["DepTime"]
    hour = dep // 100
    bin3 = hour // 3
    route = d["Origin"].astype(str) + "_" + d["Dest"].astype(str)
    return {
        "te_hour_origin": hour.astype(str) + "_" + d["Origin"].astype(str),
        "te_hour_dest": hour.astype(str) + "_" + d["Dest"].astype(str),
        "te_hour_carrier": hour.astype(str) + "_" + d["UniqueCarrier"].astype(str),
        "te_dow_origin": d["DayOfWeek"].astype(str) + "_" + d["Origin"].astype(str),
        "te_hour_route": hour.astype(str) + "_" + route,
        "te_dow_hour": d["DayOfWeek"].astype(str) + "_" + hour.astype(str),
        "te_dow_dest": d["DayOfWeek"].astype(str) + "_" + d["Dest"].astype(str),
        "te_month_origin": d["Month"].astype(str) + "_" + d["Origin"].astype(str),
        "te_bin3_route": bin3.astype(str) + "_" + route,
        "te_dow_route": d["DayOfWeek"].astype(str) + "_" + route,
        "te_carrier_route": d["UniqueCarrier"].astype(str) + "_" + route,
        "te_bin3_origin": bin3.astype(str) + "_" + d["Origin"].astype(str),
        "te_bin3_dest": bin3.astype(str) + "_" + d["Dest"].astype(str),
        "te_bin3_carrier": bin3.astype(str) + "_" + d["UniqueCarrier"].astype(str),
        "te_month_hour": d["Month"].astype(str) + "_" + hour.astype(str),
    }


ALL_TE = list(_keyfns(train.head(1)).keys())
INTER = ["te_hour_origin", "te_hour_dest", "te_hour_carrier", "te_dow_origin"]

# target-encoding tables: fit on train ONLY; train rows get out-of-fold values
_te_maps = {}      # (name, m) -> dict for new data
_te_oof = {}       # (name, m) -> array aligned to train rows


def _te_map(keys: pd.Series, y: np.ndarray, m: float) -> dict:
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    te = (g["sum"] + m * PRIOR) / (g["count"] + m)
    return te.to_dict()


def _fit_target_encodings(ms=(25.0,)):
    keys_train = _keyfns(train)
    rng = np.random.RandomState(SEED)
    fold_id = rng.randint(0, N_FOLDS, size=len(train))
    for m in ms:
        for name in ALL_TE:
            keys = keys_train[name]
            _te_maps[(name, m)] = _te_map(keys, yall, m)
            oof = np.empty(len(train))
            for f in range(N_FOLDS):
                mask = fold_id == f
                fm = _te_map(keys[~mask], yall[~mask], m)
                oof[mask] = keys[mask].map(fm).fillna(PRIOR).to_numpy()
            _te_oof[(name, m)] = oof


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


def prepare(df: pd.DataFrame, te_specs=()) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = base_features(df)
    if te_specs:
        keys = _keyfns(df)
        for name, m in te_specs:
            X[name] = keys[name].map(_te_maps[(name, m)]).fillna(PRIOR)
    return X


def prepare_train(te_specs=()) -> pd.DataFrame:
    """Training matrix: same features but TE columns use out-of-fold values (no label leak)."""
    X = base_features(train)
    for name, m in te_specs:
        X[name] = _te_oof[(name, m)]
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
yev = to_y(evald)
results = []
t0 = time.time()
_fit_target_encodings(ms=(10.0, 25.0, 50.0))
print(f"TE fit time: {time.time() - t0:.1f}s", flush=True)


def inter(m):
    return tuple((n, m) for n in INTER)


def run(name, specs):
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


run("inter_m10", inter(10.0))
run("inter_m25", inter(25.0))
run("inter_m50", inter(50.0))
run("inter+hour_route", inter(25.0) + (("te_hour_route", 25.0),))
run("hr_m10", inter(25.0) + (("te_hour_route", 10.0),))
run("hr_m50", inter(25.0) + (("te_hour_route", 50.0),))
run("hr+bin3_route", inter(25.0) + (("te_hour_route", 25.0), ("te_bin3_route", 25.0)))
run("hr+dow_route", inter(25.0) + (("te_hour_route", 25.0), ("te_dow_route", 25.0)))
run("hr+carrier_route", inter(25.0) + (("te_hour_route", 25.0), ("te_carrier_route", 25.0)))
run("hr+month_hour", inter(25.0) + (("te_hour_route", 25.0), ("te_month_hour", 25.0)))
run("hr+bin3_origin", inter(25.0) + (("te_hour_route", 25.0), ("te_bin3_origin", 25.0)))
run("hr+bin3_dest", inter(25.0) + (("te_hour_route", 25.0), ("te_bin3_dest", 25.0)))
run("hr+bin3_carrier", inter(25.0) + (("te_hour_route", 25.0), ("te_bin3_carrier", 25.0)))

best_auc, model, best_specs = max(results, key=lambda r: r[0])
print(f"diag BEST specs={best_specs} auc={best_auc:.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, best_specs))[:, 1]


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
