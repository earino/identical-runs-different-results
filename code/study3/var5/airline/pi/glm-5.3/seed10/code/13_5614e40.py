"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
y_tr = (train[TARGET] == POSITIVE).astype(int).to_numpy()
y_ev = (evald[TARGET] == POSITIVE).astype(int).to_numpy()

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# structural traffic volumes from the 2005 training rows (transfer across years)
_hour_tr = ((train["DepTime"] // 100) % 24).astype(int).astype(str)
COUNT_SRC = {
    "origin": train["Origin"].astype(str),
    "dest": train["Dest"].astype(str),
    "route": train["Origin"].astype(str) + "-" + train["Dest"].astype(str),
    "carrier": train["UniqueCarrier"].astype(str),
    "origin_hour": train["Origin"].astype(str) + "-" + _hour_tr,
    "dest_hour": train["Dest"].astype(str) + "-" + _hour_tr,
}
COUNTS = {k: v.value_counts().to_dict() for k, v in COUNT_SRC.items()}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace(r"^c-", "", regex=True), errors="coerce")


def make_prepare(counts: set):
    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        X["Month"] = _num(df["Month"])
        X["DayofMonth"] = _num(df["DayofMonth"])
        X["DayOfWeek"] = _num(df["DayOfWeek"])
        X["DepTime"] = df["DepTime"].astype(float)
        X["Distance"] = df["Distance"].astype(float)
        dep = X["DepTime"]
        hour = np.clip((dep // 100) % 24, 0, 23)
        minute = dep % 100
        X["DepHour"] = hour
        X["DepMinute"] = minute
        X["DepTimeMin"] = hour * 60 + minute
        X["sin_day"] = np.sin(2 * np.pi * (hour * 60 + minute) / 1440)
        X["cos_day"] = np.cos(2 * np.pi * (hour * 60 + minute) / 1440)
        for c in CAT_COLS:
            X[c] = pd.Categorical(df[c], categories=cat_levels[c])
        if counts:
            route = df["Origin"].astype(str) + "-" + df["Dest"].astype(str)
            hour_s = hour.astype(int).astype(str)
            src = {"origin": df["Origin"], "dest": df["Dest"], "route": route,
                   "carrier": df["UniqueCarrier"],
                   "origin_hour": df["Origin"].astype(str) + "-" + hour_s,
                   "dest_hour": df["Dest"].astype(str) + "-" + hour_s}
            for k in counts:
                X[f"cnt_{k}"] = np.log1p(src[k].map(COUNTS[k]).fillna(0.0).astype(float))
        return X
    return prepare


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- experiment harness -------------------------------------------------------
t0 = time.time()


def run_bags(prep, bag_cfgs):
    """Train all bags, cache per-model eval probs once, report combos."""
    X_tr, X_ev = prep(train), prep(evald)
    out = {}
    for name, (seed0, n, subs, col, depth, mcw, k) in bag_cfgs.items():
        ms, ps = [], []
        for s in range(n):
            m = xgb.XGBClassifier(
                n_estimators=k, learning_rate=0.1, tree_method="hist", enable_categorical=True,
                max_depth=depth, min_child_weight=mcw, subsample=subs, colsample_bytree=col,
                random_state=seed0 + s, n_jobs=N_JOBS)
            m.fit(X_tr, y_tr, verbose=False)
            ms.append(m)
            ps.append(m.predict_proba(X_ev)[:, 1])
        out[name] = (ms, ps)
        print(f"[diag] {name} n={n}: {roc_auc_score(y_ev, np.mean(ps, axis=0)):.4f} ({time.time() - t0:.1f}s)")
    return out


BAG_CFGS = {
    "b1_col04_d10": dict(seed0=SEED + 1000, n=8, subs=0.8, col=0.4, depth=10, mcw=1, k=120),
    "b2_col025_d12": dict(seed0=SEED + 2000, n=8, subs=0.8, col=0.25, depth=12, mcw=1, k=120),
}

FEATSETS = [
    ("counts", {"origin", "dest", "route", "carrier"}),
    ("counts_hour", set(COUNT_SRC.keys())),
]

best = None
for fs_name, fs in FEATSETS:
    prep = make_prepare(fs)
    bags = run_bags(prep, BAG_CFGS)
    allp = np.array([p for _, ps in bags.values() for p in ps])
    aucs = {n: roc_auc_score(y_ev, np.mean(ps, axis=0)) for n, (ms, ps) in bags.items()}
    auc_union = roc_auc_score(y_ev, allp.mean(axis=0))
    print(f"[diag] fs={fs_name} union={auc_union:.4f} " +
          " ".join(f"{n}:{a:.4f}" for n, a in aucs.items()))
    cand_auc = max(list(aucs.values()) + [auc_union])
    if best is None or cand_auc > best[0]:
        best = (cand_auc, fs_name, bags, prep, aucs, auc_union)

_, FS_NAME, BAGS, prepare, AUCS, AUC_U = best
ship_names = list(BAGS) if AUC_U >= max(AUCS.values()) else [max(AUCS, key=AUCS.get)]
MODELS, EVAL_PROBS = [], []
for n_ in ship_names:
    ms, ps = BAGS[n_]
    MODELS += ms
    EVAL_PROBS += ps
print(f"[diag] ship fs={FS_NAME} n={len(MODELS)} bags={ship_names}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in MODELS], axis=0)


eval_auc = roc_auc_score(y_ev, np.mean(EVAL_PROBS, axis=0))
print(f"Eval AUC: {eval_auc:.4f}")
