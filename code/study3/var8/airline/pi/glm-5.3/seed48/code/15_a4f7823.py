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


def fit_one(specs, oof, view="full", **over):
    params = dict(PARAMS)
    params.update(over)
    m = xgb.XGBClassifier(**params)
    m.fit(prepare_train(specs, oof, view), yall)
    return m


def make_multi(members, specs=BEST_SPECS, w=None):
    """members: list of (model, view). Prediction = (weighted) average of member probabilities."""
    def fn(df):
        ps = [m.predict_proba(prepare(df, specs, v))[:, 1] for m, v in members]
        return np.average(ps, axis=0, weights=w) if w is not None else np.mean(ps, axis=0)
    return fn


def evaluate(name, members, specs=BEST_SPECS, w=None):
    fn = make_multi(members, specs, w)
    auc = roc_auc_score(yev, fn(evald))
    results.append((auc, fn, name))
    print(f"diag feat={name} auc={auc:.4f} t={time.time() - t0:.1f}s", flush=True)
    return auc


_fit_predict_maps(BEST_SPECS)
oof5 = _fit_oof(BEST_SPECS, SEED, 5)
print(f"TE fit time: {time.time() - t0:.1f}s", flush=True)

A = (fit_one(BEST_SPECS, oof5, "full"), "full")
B = (fit_one(BEST_SPECS, oof5, "nocat"), "nocat")
B4 = (fit_one(BEST_SPECS, oof5, "nocat", max_depth=4), "nocat")
C = (fit_one(BEST_SPECS, oof5, "teonly"), "teonly")
D = (fit_one(BEST_SPECS, oof5, "catsonly"), "catsonly")
D6 = (fit_one(BEST_SPECS, oof5, "catsonly", max_depth=6, learning_rate=0.1, n_estimators=200), "catsonly")


def rank_avg(members, w=None):
    from scipy.stats import rankdata

    def fn(df):
        ps = [m.predict_proba(prepare(df, BEST_SPECS, v))[:, 1] for m, v in members]
        rs = [rankdata(p) for p in ps]
        return np.average(rs, axis=0, weights=w) if w is not None else np.mean(rs, axis=0)
    return fn


def evaluate_rank(name, members, w=None):
    fn = rank_avg(members, w)
    auc = roc_auc_score(yev, fn(evald))
    results.append((auc, fn, name))
    print(f"diag feat={name} auc={auc:.4f} t={time.time() - t0:.1f}s", flush=True)


ABCD = [A, B, C, D]
evaluate("ABCD", ABCD)
evaluate("ABCD w3221", ABCD, w=[3, 2, 2, 1])
evaluate("ABCD w2221", ABCD, w=[2, 2, 2, 1])
evaluate("ABCD w2112", ABCD, w=[2, 1, 1, 2])
evaluate("ABCD w4221", ABCD, w=[4, 2, 2, 1])
evaluate("ABCD+B4", ABCD + [B4])
evaluate("AB4CD", [A, B4, C, D])
evaluate("ABCD+D6", ABCD + [D6])
evaluate("ABCD+D6+B4", ABCD + [D6, B4])
evaluate_rank("ABCD_rank", ABCD)
A2 = (fit_one(BEST_SPECS, oof5, "full", random_state=11), "full")
B2 = (fit_one(BEST_SPECS, oof5, "nocat", random_state=11), "nocat")
C2 = (fit_one(BEST_SPECS, oof5, "teonly", random_state=11), "teonly")
D2 = (fit_one(BEST_SPECS, oof5, "catsonly", random_state=11), "catsonly")
evaluate("ABCDx2", [A, A2, B, B2, C, C2, D, D2])

best_auc, best_fn, best_name = max(results, key=lambda r: r[0])
print(f"diag BEST {best_name} auc={best_auc:.4f}")

predict_proba = best_fn
print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
