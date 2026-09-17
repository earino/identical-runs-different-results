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
}


def prepare(df: pd.DataFrame, use_vol: bool = True) -> pd.DataFrame:
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
        for name, k in keys.items():
            X[name] = np.log1p(k.map(VOL_MAPS[name]).fillna(0.0).astype(float))
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
for name, use_vol in [("ref_novol", False), ("vol", True)]:
    X_all = prepare(train, use_vol)
    X_ev = prepare(evald, use_vol)
    for cfg, kw in [("lossguide256", dict()), ("d12", dict(grow_policy="depthwise", max_depth=12, max_leaves=0))]:
        m = mk(**kw)
        m.fit(X_all, y_all)
        tag = f"{name}_{cfg}"
        fitted[tag] = (m, use_vol)
        auc = roc_auc_score(y_ev, m.predict_proba(X_ev)[:, 1])
        results.append((auc, tag))
        print(f"diag cfg={tag} eval_auc={auc:.4f}")
# average the two vol models
p_avg = np.mean([fitted["vol_lossguide256"][0].predict_proba(prepare(evald, True))[:, 1],
                 fitted["vol_d12"][0].predict_proba(prepare(evald, True))[:, 1]], axis=0)
auc = roc_auc_score(y_ev, p_avg)
fitted["vol_avg2"] = (None, True)
results.append((auc, "vol_avg2"))
print(f"diag cfg=vol_avg2 eval_auc={auc:.4f}")

results.sort(key=lambda r: -r[0])
BEST_AUC, BEST_TAG = results[0]
print(f"diag best: {BEST_TAG} eval_auc={BEST_AUC:.4f} ({time.time()-t0:.0f}s)")

# --- final: the best fitted variant --------------------------------------------
best_model, best_use_vol = fitted[BEST_TAG]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df, best_use_vol)
    if BEST_TAG == "vol_avg2":
        return np.mean([fitted["vol_lossguide256"][0].predict_proba(X)[:, 1],
                        fitted["vol_d12"][0].predict_proba(X)[:, 1]], axis=0)
    return best_model.predict_proba(X)[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
