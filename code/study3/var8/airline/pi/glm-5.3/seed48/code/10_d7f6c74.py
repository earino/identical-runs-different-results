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
        "te_flight2": route + "_" + dep.astype(str),
        "te_flight3": d["UniqueCarrier"].astype(str) + "_" + d["Origin"].astype(str) + "_" + dep.astype(str),
        "te_flight4": d["UniqueCarrier"].astype(str) + "_" + route + "_" + hour.astype(str),
        "te_origdep": d["Origin"].astype(str) + "_" + dep.astype(str),
        "te_destdep": d["Dest"].astype(str) + "_" + dep.astype(str),
        "cnt_origin": d["Origin"].astype(str),
        "cnt_dest": d["Dest"].astype(str),
        "cnt_route": route,
        "cnt_route_hour": route + "_" + hour.astype(str),
    }


INTER = ["te_hour_origin", "te_hour_dest", "te_hour_carrier", "te_dow_origin"]
BASE_SPECS = tuple((n, 25.0) for n in INTER) + (("te_hour_route", 10.0),)

# target-encoding tables: fit on train ONLY; train rows get out-of-fold values
_te_maps = {}      # (name, m) -> dict for new data (plain TE)
_te_oof = {}       # (name, m) -> array aligned to train rows (plain TE)
_hier_maps = {}    # (name, m, prior_name, prior_m) -> dict
_hier_oof = {}     # (name, m, prior_name, prior_m) -> array
_cnt_maps = {}     # name -> dict key -> count in train


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


def _fit_target_encodings(specs):
    """specs: list of either (name, m) plain TE, ('hier', name, m, prior_name, prior_m), or ('cnt', name)."""
    keys_train = _keyfns(train)
    rng = np.random.RandomState(SEED)
    fold_id = rng.randint(0, N_FOLDS, size=len(train))
    names = [s[1] if s[0] in ("hier", "cnt") else s[0] for s in specs]
    for s in specs:
        if s[0] == "cnt":
            name = s[1]
            _cnt_maps[name] = keys_train[name].value_counts().to_dict()
            continue
        if s[0] == "hier":
            _, name, m, prior_name, prior_m = s
            keys, pkeys = keys_train[name], keys_train[prior_name]
            _hier_maps[s] = _hier_te_map(keys, pkeys, yall, m, prior_m)
            oof = np.empty(len(train))
            for f in range(N_FOLDS):
                mask = fold_id == f
                fm = _hier_te_map(keys[~mask], pkeys[~mask], yall[~mask], m, prior_m)
                oof[mask] = keys[mask].map(fm).fillna(PRIOR).to_numpy()
            _hier_oof[s] = oof
        else:
            name, m = s
            keys = keys_train[name]
            _te_maps[s] = _te_map(keys, yall, m)
            oof = np.empty(len(train))
            for f in range(N_FOLDS):
                mask = fold_id == f
                fm = _te_map(keys[~mask], yall[~mask], m)
                oof[mask] = keys[mask].map(fm).fillna(PRIOR).to_numpy()
            _te_oof[s] = oof
    return keys_train


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


def apply_specs(X: pd.DataFrame, keys: dict, specs) -> pd.DataFrame:
    for s in specs:
        if s[0] == "cnt":
            name = s[1]
            X[name] = np.log1p(keys[name].map(_cnt_maps[name]).fillna(0.0))
        elif s[0] == "hier":
            X[s[1]] = keys[s[1]].map(_hier_maps[s]).fillna(PRIOR)
        else:
            name, m = s
            X[name] = keys[name].map(_te_maps[s]).fillna(PRIOR)
    return X


def prepare(df: pd.DataFrame, specs=()) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = base_features(df)
    if specs:
        keys = _keyfns(df)
        X = apply_specs(X, keys, specs)
    return X


def prepare_train(specs) -> pd.DataFrame:
    """Training matrix: same features but TE columns use out-of-fold values (no label leak)."""
    X = base_features(train)
    for s in specs:
        if s[0] == "cnt":
            continue  # counts use full-train values for train too (no label involved)
        elif s[0] == "hier":
            X[s[1]] = _hier_oof[s]
        else:
            X[s[0]] = _te_oof[s]
    for s in specs:  # counts after: for train rows too
        if s[0] == "cnt":
            keys = _keyfns(train)
            X[s[1]] = np.log1p(keys[s[1]].map(_cnt_maps[s[1]]).fillna(0.0))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
yev = to_y(evald)
results = []
t0 = time.time()
ALL_SPECS = list(BASE_SPECS) + [
    ("hier", "te_flight", 5.0, "te_route", 25.0),
    ("hier", "te_flight", 3.0, "te_route", 25.0),
    ("hier", "te_flight", 8.0, "te_route", 25.0),
    ("hier", "te_flight", 5.0, "te_carrier_route", 25.0),
    ("hier", "te_flight2", 5.0, "te_route", 25.0),
    ("hier", "te_flight3", 5.0, "te_route", 25.0),
    ("hier", "te_flight4", 5.0, "te_route", 25.0),
    ("hier", "te_origdep", 5.0, "te_origin", 25.0),
    ("hier", "te_destdep", 5.0, "te_dest", 25.0),
    ("cnt", "cnt_route_hour"),
]
_fit_target_encodings(ALL_SPECS)
print(f"TE fit time: {time.time() - t0:.1f}s", flush=True)


def run(name, specs, **over):
    Xtr, Xev = prepare_train(specs), prepare(evald, specs)
    params = dict(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        colsample_bytree=0.7,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    params.update(over)
    m = xgb.XGBClassifier(**params)
    m.fit(Xtr, yall)
    auc = roc_auc_score(yev, m.predict_proba(Xev)[:, 1])
    results.append((auc, m, specs, over))
    print(f"diag feat={name} auc={auc:.4f} t={time.time() - t0:.1f}s", flush=True)
    return auc


HIER10 = ("hier", "te_hour_route", 10.0, "te_route", 25.0)
FLI5 = ("hier", "te_flight", 5.0, "te_route", 25.0)
FLI15 = ("hier", "te_flight", 15.0, "te_carrier_route", 25.0)

run("ref", BASE_SPECS)
run("f1_m5", BASE_SPECS + (FLI5,))
run("f1_m3", BASE_SPECS + (("hier", "te_flight", 3.0, "te_route", 25.0),))
run("f1_m8", BASE_SPECS + (("hier", "te_flight", 8.0, "te_route", 25.0),))
run("f1_m5_cr", BASE_SPECS + (("hier", "te_flight", 5.0, "te_carrier_route", 25.0),))
run("f2", BASE_SPECS + (("hier", "te_flight2", 5.0, "te_route", 25.0),))
run("f3", BASE_SPECS + (("hier", "te_flight3", 5.0, "te_route", 25.0),))
run("f4", BASE_SPECS + (("hier", "te_flight4", 5.0, "te_route", 25.0),))
run("origdep", BASE_SPECS + (("hier", "te_origdep", 5.0, "te_origin", 25.0),))
run("destdep", BASE_SPECS + (("hier", "te_destdep", 5.0, "te_dest", 25.0),))
run("f1+f2", BASE_SPECS + (FLI5, ("hier", "te_flight2", 5.0, "te_route", 25.0)))
run("f1+rhcnt", BASE_SPECS + (FLI5, ("cnt", "cnt_route_hour")))
run("f1+origdep+destdep", BASE_SPECS + (FLI5, ("hier", "te_origdep", 5.0, "te_origin", 25.0), ("hier", "te_destdep", 5.0, "te_dest", 25.0)))

best_auc, model, best_specs, best_over = max(results, key=lambda r: r[0])
print(f"diag BEST specs={best_specs} over={best_over} auc={best_auc:.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, best_specs))[:, 1]


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
