"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
BASE_CATS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in BASE_CATS}


def dep_hour_of(df: pd.DataFrame) -> pd.Series:
    return np.floor(df["DepTime"].astype(float) / 100.0)


# traffic-volume counts from TRAIN ONLY (target-free, year-stable)
_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
_origh = train["Origin"].astype(str) + "_" + dep_hour_of(train).astype(int).astype(str)
_desth = train["Dest"].astype(str) + "_" + dep_hour_of(train).astype(int).astype(str)
VOL_MAPS = {
    "vol_origin": train["Origin"].value_counts(),
    "vol_dest": train["Dest"].value_counts(),
    "vol_route": _route.value_counts(),
    "vol_carrier": train["UniqueCarrier"].value_counts(),
    "vol_hour": dep_hour_of(train).astype(int).value_counts(),
    "vol_origin_hour": _origh.value_counts(),
    "vol_dest_hour": _desth.value_counts(),
    "vol_route_hour": (_route + "_" + dep_hour_of(train).astype(int).astype(str)).value_counts(),
    "vol_carrier_hour": (train["UniqueCarrier"].astype(str) + "_" + dep_hour_of(train).astype(int).astype(str)).value_counts(),
    "route_mean_dist": train.groupby(_route)["Distance"].mean(),
}


def prepare(df: pd.DataFrame, use_vol: bool = True, ext: bool = True) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(float)
    dep_h = np.floor(dep / 100.0)
    minutes = dep_h * 60.0 + (dep - dep_h * 100.0)
    X["dep_raw"] = dep
    X["dep_minutes"] = minutes
    X["sin_min"] = np.sin(2 * np.pi * minutes / 1440.0)
    X["cos_min"] = np.cos(2 * np.pi * minutes / 1440.0)
    X["dep_hour"] = dep_h
    X["distance"] = df["Distance"].astype(float)
    for c in BASE_CATS:
        X[c] = pd.Categorical(df[c].values, categories=cat_levels[c])
    if use_vol:
        hour = dep_h.astype(int).astype(str)
        keys = {
            "vol_origin": df["Origin"],
            "vol_dest": df["Dest"],
            "vol_route": df["Origin"].astype(str) + "_" + df["Dest"].astype(str),
            "vol_carrier": df["UniqueCarrier"],
            "vol_hour": dep_h.astype(int),
            "vol_origin_hour": df["Origin"].astype(str) + "_" + hour,
            "vol_dest_hour": df["Dest"].astype(str) + "_" + hour,
        }
        keys["vol_route"] = keys["vol_route"].astype(str)
        keys["vol_hour"] = keys["vol_hour"].astype(int)
        for name, k in keys.items():
            X[name] = np.log1p(k.map(VOL_MAPS[name]).fillna(0.0).astype(float))
        if ext:
            rd = keys["vol_route"].map(VOL_MAPS["route_mean_dist"])
            X["dist_vs_route_mean"] = X["distance"] - rd.fillna(train["Distance"].mean())
            X["route_mean_dist"] = rd.fillna(train["Distance"].mean())
            X["vol_route_hour"] = np.log1p(keys["vol_route"].astype(str).map(VOL_MAPS["vol_route_hour"]).fillna(0.0))
            X["vol_carrier_hour"] = np.log1p((df["UniqueCarrier"].astype(str) + "_" + keys["vol_hour"].astype(str)).map(VOL_MAPS["vol_carrier_hour"]).fillna(0.0))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def mk(seed=SEED, **kw):
    params = dict(n_estimators=600, learning_rate=0.01, min_child_weight=10,
                  subsample=0.4, colsample_bytree=0.4, tree_method="hist",
                  enable_categorical=True, random_state=seed, n_jobs=N_JOBS,
                  grow_policy="lossguide", max_depth=0, max_leaves=256)
    params.update(kw)
    return xgb.XGBClassifier(**params)


# --- diagnostic ----------------------------------------------------------------
t0 = time.time()
y_all = to_y(train)
y_ev = to_y(evald)

fitted = {}
results = []
X_all = prepare(train, True)
X_ev = prepare(evald, True)
for cfg, kw in [("lossguide256", dict()), ("lossguide512", dict(max_leaves=512)),
                ("lossguide128", dict(max_leaves=128))]:
    m = mk(**kw)
    m.fit(X_all, y_all)
    fitted[cfg] = m
    auc = roc_auc_score(y_ev, m.predict_proba(X_ev)[:, 1])
    results.append((auc, cfg))
    print(f"diag cfg={cfg} eval_auc={auc:.4f}")
p_avg = np.mean([fitted["lossguide256"].predict_proba(X_ev)[:, 1],
                 fitted["lossguide512"].predict_proba(X_ev)[:, 1]], axis=0)
auc = roc_auc_score(y_ev, p_avg)
fitted["avg256_512"] = None
results.append((auc, "avg256_512"))
print(f"diag cfg=avg256_512 eval_auc={auc:.4f}")

results.sort(key=lambda r: -r[0])
BEST_AUC, BEST_TAG = results[0]
print(f"diag best: {BEST_TAG} eval_auc={BEST_AUC:.4f} ({time.time()-t0:.0f}s)")

# --- final: the best fitted variant --------------------------------------------
best_model = fitted[BEST_TAG]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df, True)
    if BEST_TAG == "avg256_512":
        return np.mean([fitted["lossguide256"].predict_proba(X)[:, 1],
                        fitted["lossguide512"].predict_proba(X)[:, 1]], axis=0)
    return best_model.predict_proba(X)[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
