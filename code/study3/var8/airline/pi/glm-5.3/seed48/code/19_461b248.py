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
route_levels = pd.Index(sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique()))

yall = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(yall.mean())


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
        "te_carrier": d["UniqueCarrier"].astype(str),
        "te_hour": hour.astype(str),
    }


INTER = ["te_hour_origin", "te_hour_dest", "te_hour_carrier", "te_dow_origin"]
BASE_SPECS = tuple((n, 25.0) for n in INTER) + (("te_hour_route", 10.0),)
BEST_SPECS = BASE_SPECS + (
    ("hier", "te_flight", 5.0, "te_route", 25.0),
    ("hier", "te_origdep", 5.0, "te_origin", 25.0),
    ("hier", "te_destdep", 5.0, "te_dest", 25.0),
    ("hier", "te_flight4", 5.0, "te_carrier_route", 25.0),
)
SIMPLE_TE = (("te_carrier", 25.0), ("te_origin", 25.0), ("te_dest", 25.0), ("te_route", 25.0), ("te_hour", 25.0))
ESPECS = BEST_SPECS + SIMPLE_TE

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
    """Out-of-fold train encodings for plain + hier TE specs."""
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
    if view in ("full", "catsonly", "route", "noclock"):
        for c in cat_cols:
            X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dep = df["DepTime"]
    hour = dep // 100
    X["hour"] = hour
    X["minute"] = dep % 100
    X["tod"] = hour * 60 + dep % 100
    if view == "noclock":
        X = X.drop(columns=["hour", "minute", "tod"])
    if view == "route":
        X["route"] = pd.Categorical(
            df["Origin"].astype(str) + "_" + df["Dest"].astype(str), categories=route_levels
        )
    return X


def prepare(df: pd.DataFrame, specs=BEST_SPECS, view="full") -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = base_features(df, view)
    if view in ("catsonly", "route"):
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
    if view in ("catsonly", "route"):
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
results = []
t0 = time.time()

PARAMS = dict(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    colsample_bytree=0.7,
    random_state=SEED,
    n_jobs=N_JOBS,
)


def fit_one(specs, oof, view="full", frac=1.0, **over):
    params = dict(PARAMS)
    params.update(over)
    m = xgb.XGBClassifier(**params)
    X = prepare_train(specs, oof, view)
    if frac < 1.0:
        idx = np.random.RandomState(over.get("random_state", SEED) * 7 + 1).choice(len(X), int(len(X) * frac), replace=False)
        m.fit(X.iloc[idx], yall[idx])
    else:
        m.fit(X, yall)
    return m


def make_multi(members, w=None):
    """members: list of (model, view, specs). Prediction = (weighted) average of member probabilities."""
    def fn(df):
        ps = [m.predict_proba(prepare(df, specs, v))[:, 1] for m, v, specs in members]
        return np.average(ps, axis=0, weights=w) if w is not None else np.mean(ps, axis=0)
    return fn


def evaluate(name, members, w=None):
    fn = make_multi(members, w)
    auc = roc_auc_score(yev, fn(evald))
    results.append((auc, fn, name))
    print(f"diag feat={name} auc={auc:.4f} t={time.time() - t0:.1f}s", flush=True)
    return auc


_fit_predict_maps(BEST_SPECS + SIMPLE_TE)
oof5 = _fit_oof(BEST_SPECS + SIMPLE_TE, SEED, 5)
print(f"TE fit time: {time.time() - t0:.1f}s", flush=True)


def mv(m, view, specs=BEST_SPECS):
    return (m, view, specs)


A = mv(fit_one(BEST_SPECS, oof5), "full")
B = mv(fit_one(BEST_SPECS, oof5, "nocat"), "nocat")
B4 = mv(fit_one(BEST_SPECS, oof5, "nocat", max_depth=4), "nocat")
C = mv(fit_one(BEST_SPECS, oof5, "teonly"), "teonly")
C4 = mv(fit_one(BEST_SPECS, oof5, "teonly", max_depth=4), "teonly")
D = mv(fit_one(BEST_SPECS, oof5, "catsonly"), "catsonly")
D6 = mv(fit_one(BEST_SPECS, oof5, "catsonly", max_depth=6, learning_rate=0.1, n_estimators=200), "catsonly")
F4 = mv(fit_one(BEST_SPECS, oof5, "full", max_depth=4), "full")
E = mv(fit_one(ESPECS, oof5), "full", ESPECS)
S = mv(fit_one(BEST_SPECS, oof5, "full", frac=0.8, random_state=3), "full")

REF8 = [A, B, C, D, D6, B4, F4, C4]
FLIGHT_SPECS = tuple(s for s in BEST_SPECS if s not in BASE_SPECS)
M_base = mv(fit_one(BASE_SPECS, oof5), "full", BASE_SPECS)
M_flight = mv(fit_one(FLIGHT_SPECS, oof5), "full", FLIGHT_SPECS)
M_inter = mv(fit_one(tuple(s for s in BASE_SPECS if s[0] != "te_hour_route"), oof5), "full", tuple(s for s in BASE_SPECS if s[0] != "te_hour_route"))
CS04 = mv(fit_one(BEST_SPECS, oof5, "full", colsample_bytree=0.4, random_state=55), "full")
MCW10 = mv(fit_one(BEST_SPECS, oof5, "full", min_child_weight=10, random_state=56), "full")

evaluate("ref8", REF8)
evaluate("M_base", [M_base])
evaluate("M_flight", [M_flight])
evaluate("ref8+M_base", REF8 + [M_base])
evaluate("ref8+M_flight", REF8 + [M_flight])
evaluate("ref8+M_inter", REF8 + [M_inter])
evaluate("ref8+M_base+M_flight", REF8 + [M_base, M_flight])
evaluate("ref8+CS04", REF8 + [CS04])
evaluate("ref8+MCW10", REF8 + [MCW10])
evaluate("ref8+all5", REF8 + [M_base, M_flight, M_inter, CS04, MCW10])

best_auc, best_fn, best_name = max(results, key=lambda r: r[0])
print(f"diag BEST {best_name} auc={best_auc:.4f}")

predict_proba = best_fn
print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
