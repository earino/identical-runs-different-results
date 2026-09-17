"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Approach (see FINAL.md):
  - Features: raw airline columns + hour/minute/tod numerics + smoothed target encodings (TE) of
    schedule-fingerprint interactions (hour x origin/dest/carrier, hour x route, carrier x route x DepTime,
    origin x DepTime, dest x DepTime, carrier x route x hour), all fit on train only with out-of-fold
    values for the training rows (no label leak).
  - Model: ensemble (uniform average) of 8 XGBoost classifiers over different feature "views"
    (full / no-categoricals / TEs-only / raw-only) and depths.
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
    route = d["Origin"].astype(str) + "_" + d["Dest"].astype(str)
    return {
        "te_hour_origin": hour.astype(str) + "_" + d["Origin"].astype(str),
        "te_hour_dest": hour.astype(str) + "_" + d["Dest"].astype(str),
        "te_hour_carrier": hour.astype(str) + "_" + d["UniqueCarrier"].astype(str),
        "te_dow_origin": d["DayOfWeek"].astype(str) + "_" + d["Origin"].astype(str),
        "te_hour_route": hour.astype(str) + "_" + route,
        "te_origin": d["Origin"].astype(str),
        "te_dest": d["Dest"].astype(str),
        "te_route": route,
        "te_carrier_route": d["UniqueCarrier"].astype(str) + "_" + route,
        "te_flight": d["UniqueCarrier"].astype(str) + "_" + route + "_" + dep.astype(str),
        "te_origdep": d["Origin"].astype(str) + "_" + dep.astype(str),
        "te_destdep": d["Dest"].astype(str) + "_" + dep.astype(str),
        "te_flight4": d["UniqueCarrier"].astype(str) + "_" + route + "_" + hour.astype(str),
    }


INTER = ["te_hour_origin", "te_hour_dest", "te_hour_carrier", "te_dow_origin"]
BASE_SPECS = tuple((n, 25.0) for n in INTER) + (("te_hour_route", 10.0),)
BEST_SPECS = BASE_SPECS + (
    ("hier", "te_flight", 5.0, "te_route", 25.0),
    ("hier", "te_origdep", 5.0, "te_origin", 25.0),
    ("hier", "te_destdep", 5.0, "te_dest", 25.0),
    ("hier", "te_flight4", 5.0, "te_carrier_route", 25.0),
)

# target-encoding tables: fit on train ONLY; train rows get out-of-fold values
_te_maps = {}      # (name, m) -> dict for new data (plain TE)
_hier_maps = {}    # (name, m, prior_name, prior_m) -> dict


def _te_map(keys: pd.Series, y: np.ndarray, m: float) -> dict:
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    te = (g["sum"] + m * PRIOR) / (g["count"] + m)
    return te.to_dict()


def _hier_te_map(keys: pd.Series, prior_keys: pd.Series, y: np.ndarray, m: float, prior_m: float) -> dict:
    prior_map = _te_map(prior_keys, y, prior_m)
    df = pd.DataFrame({"k": keys.to_numpy(), "pk": prior_keys.to_numpy(), "y": y})
    k2p = df.drop_duplicates("k").set_index("k")["pk"]
    g = df.groupby("k")["y"].agg(["sum", "count"])
    prior = k2p.map(prior_map).fillna(PRIOR)
    te = (g["sum"] + m * prior) / (g["count"] + m)
    return te.to_dict()


def _fit_oof(specs, fold_seed, n_folds):
    """Out-of-fold train encodings for plain + hier TE specs (no label leak into the model's own row)."""
    keys_train = _keyfns(train)
    rng = np.random.RandomState(fold_seed)
    fold_id = rng.randint(0, n_folds, size=len(train))
    oof = {}
    for s in specs:
        if s[0] == "hier":
            _, name, m, prior_name, prior_m = s
            keys, pkeys = keys_train[name], keys_train[prior_name]
        else:
            name, m = s
            keys, pkeys = keys_train[name], None
        col = np.empty(len(train))
        for f in range(n_folds):
            mask = fold_id == f
            if pkeys is not None:
                fm = _hier_te_map(keys[~mask], pkeys[~mask], yall[~mask], m, prior_m)
            else:
                fm = _te_map(keys[~mask], yall[~mask], m)
            col[mask] = keys[mask].map(fm).fillna(PRIOR).to_numpy()
        oof[s] = col
    return oof


def _fit_predict_maps(specs):
    keys_train = _keyfns(train)
    for s in specs:
        if s[0] == "hier":
            _, name, m, prior_name, prior_m = s
            _hier_maps[s] = _hier_te_map(keys_train[name], keys_train[prior_name], yall, m, prior_m)
        else:
            name, m = s
            _te_maps[s] = _te_map(keys_train[name], yall, m)


def base_features(df: pd.DataFrame, view="full") -> pd.DataFrame:
    drop = set()
    if view in ("nocat", "teonly"):
        drop |= set(cat_cols)
    if view == "teonly":
        drop |= {"DepTime", "Distance"}
    X = df[feature_cols].drop(columns=list(drop & set(df.columns))).copy()
    if view in ("full", "catsonly"):
        for c in cat_cols:
            X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dep = df["DepTime"]
    hour = dep // 100
    X["hour"] = hour
    X["minute"] = dep % 100
    X["tod"] = hour * 60 + dep % 100
    return X


def prepare(df: pd.DataFrame, specs=BEST_SPECS, view="full") -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = base_features(df, view)
    if view == "catsonly":
        return X
    keys = _keyfns(df)
    for s in specs:
        if s[0] == "hier":
            X[s[1]] = keys[s[1]].map(_hier_maps[s]).fillna(PRIOR)
        else:
            name, m = s
            X[name] = keys[name].map(_te_maps[s]).fillna(PRIOR)
    return X


def prepare_train(specs, oof, view="full") -> pd.DataFrame:
    """Training matrix: same features but TE columns use out-of-fold values (no label leak)."""
    X = base_features(train, view)
    if view == "catsonly":
        return X
    for s in specs:
        if s[0] == "hier":
            X[s[1]] = oof[s]
        else:
            X[s[0]] = oof[s]
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
yev = to_y(evald)
t0 = time.time()

PARAMS = dict(
    max_depth=3,
    learning_rate=0.025,
    n_estimators=2000,
    tree_method="hist",
    enable_categorical=True,
    colsample_bytree=0.7,
    random_state=SEED,
    n_jobs=N_JOBS,
)


def fit_one(view, **over):
    params = dict(PARAMS)
    params.update(over)
    m = xgb.XGBClassifier(**params)
    m.fit(prepare_train(BEST_SPECS, oof5, view), yall)
    return m


_fit_predict_maps(BEST_SPECS)
oof5 = _fit_oof(BEST_SPECS, SEED, N_FOLDS)
print(f"TE fit time: {time.time() - t0:.1f}s", flush=True)

MEMBERS = [
    (fit_one("full"), "full"),
    (fit_one("nocat"), "nocat"),
    (fit_one("teonly"), "teonly"),
    (fit_one("catsonly"), "catsonly"),
    (fit_one("catsonly", max_depth=6, learning_rate=0.1, n_estimators=200), "catsonly"),
    (fit_one("nocat", max_depth=4), "nocat"),
    (fit_one("full", max_depth=4), "full"),
    (fit_one("teonly", max_depth=4), "teonly"),
]
D5 = [
    (fit_one("full", max_depth=5), "full"),
    (fit_one("nocat", max_depth=5), "nocat"),
    (fit_one("teonly", max_depth=5), "teonly"),
    (fit_one("catsonly", max_depth=5), "catsonly"),
]
MEMBERS = MEMBERS + D5 + [
    (fit_one("full", max_depth=6), "full"),
    (fit_one("nocat", max_depth=6), "nocat"),
    (fit_one("teonly", max_depth=6), "teonly"),
    (fit_one("catsonly", max_depth=4), "catsonly"),
]
print(f"Ensemble fit time: {time.time() - t0:.1f}s", flush=True)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    ps = [m.predict_proba(prepare(df, BEST_SPECS, v))[:, 1] for m, v in MEMBERS]
    return np.mean(ps, axis=0)


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
