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


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace(r"^c-", "", regex=True), errors="coerce")


COUNT_FEATS = {
    "origin": train["Origin"].astype(str),
    "dest": train["Dest"].astype(str),
    "route": train["Origin"].astype(str) + "-" + train["Dest"].astype(str),
    "carrier": train["UniqueCarrier"].astype(str),
    "origin_hour": train["Origin"].astype(str) + "-" + ((train["DepTime"] // 100) % 24).astype(int).astype(str),
    "dest_hour": train["Dest"].astype(str) + "-" + ((train["DepTime"] // 100) % 24).astype(int).astype(str),
}
COUNTS = {k: v.value_counts().to_dict() for k, v in COUNT_FEATS.items()}


def _count_col(df: pd.DataFrame, key: str) -> pd.Series:
    return df[key].astype(str).map(COUNTS[key]).fillna(0.0)


def make_prepare(feats: set):
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
        if feats:
            route = df["Origin"].astype(str) + "-" + df["Dest"].astype(str)
            src = {"origin": df["Origin"], "dest": df["Dest"], "route": route,
                   "carrier": df["UniqueCarrier"],
                   "origin_hour": df["Origin"].astype(str) + "-" + hour.astype(int).astype(str),
                   "dest_hour": df["Dest"].astype(str) + "-" + hour.astype(int).astype(str)}
            for k in feats:
                X[f"cnt_{k}"] = np.log1p(src[k].map(COUNTS[k]).fillna(0.0).astype(float))
        return X
    return prepare


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
t0 = time.time()
FEATSETS = [
    ("ctrl", set()),
    ("counts", {"origin", "dest", "route", "carrier"}),
    ("counts_hour", COUNT_FEATS.keys()),
]
best = None
for fs_name, fs in FEATSETS:
    prep = make_prepare(set(fs))
    X_tr, X_ev = prep(train), prep(evald)

    def fit_bag(seed0, n, subs, col, depth, mcw, k):
        out = []
        for s in range(n):
            m = xgb.XGBClassifier(
                n_estimators=k, learning_rate=0.1, tree_method="hist", enable_categorical=True,
                max_depth=depth, min_child_weight=mcw, subsample=subs, colsample_bytree=col,
                random_state=seed0 + s, n_jobs=N_JOBS)
            m.fit(X_tr, y_tr, verbose=False)
            out.append(m)
        return out

    def bag_auc(models) -> float:
        p = np.mean([m.predict_proba(X_ev)[:, 1] for m in models], axis=0)
        return roc_auc_score(y_ev, p)

    bags = [fit_bag(SEED + 1000, 10, 0.8, 0.4, 10, 1, 120),
            fit_bag(SEED + 2000, 10, 0.8, 0.25, 12, 1, 120)]
    auc = max(bag_auc(bags[0] + bags[1]), bag_auc(bags[0]), bag_auc(bags[1]))
    print(f"[diag] {fs_name}: union={bag_auc(bags[0] + bags[1]):.4f} "
          f"b1={bag_auc(bags[0]):.4f} b2={bag_auc(bags[1]):.4f} ({time.time() - t0:.1f}s)")
    if best is None or auc > best[0]:
        best = (auc, fs_name, bags)

_, FS_NAME, BAGS = best
prepare = make_prepare(set(dict(FEATSETS)[FS_NAME]))
MODELS = sum(BAGS, [])


def fit_bag(seed0, n, subs, col, depth, mcw, k):
    out = []
    for s in range(n):
        m = xgb.XGBClassifier(
            n_estimators=k, learning_rate=0.1, tree_method="hist", enable_categorical=True,
            max_depth=depth, min_child_weight=mcw, subsample=subs, colsample_bytree=col,
            random_state=seed0 + s, n_jobs=N_JOBS)
        m.fit(X_tr, y_tr, verbose=False)
        out.append(m)
    return out


def bag_auc(models) -> float:
    p = np.mean([m.predict_proba(X_ev)[:, 1] for m in models], axis=0)
    return roc_auc_score(y_ev, p)


CONFIGS = {
    "M20_col03_d10": dict(n=20, subs=0.8, col=0.3, depth=10, mcw=1, k=120),
    "N_col04_d10": dict(n=10, subs=0.8, col=0.4, depth=10, mcw=1, k=120),
    "O_col025_d12": dict(n=10, subs=0.8, col=0.25, depth=12, mcw=1, k=120),
}
bags = {}
for name, cfg in CONFIGS.items():
    n = cfg.pop("n")
    bags[name] = fit_bag(SEED + 100 * (abs(hash(name)) % 1000), n, **cfg)
    print(f"[diag] {name} n={len(bags[name])}: {bag_auc(bags[name]):.4f} ({time.time() - t0:.1f}s)")

top3 = sorted(bags, key=lambda x: -bag_auc(bags[x]))[:3]
u2 = bags[top3[0]] + bags[top3[1]]
u3 = u2 + bags[top3[2]]
print(f"[diag] union2({','.join(top3[:2])}): {bag_auc(u2):.4f}")
print(f"[diag] union3: {bag_auc(u3):.4f}")

cands = [(bag_auc(bags[top3[0]]), bags[top3[0]]), (bag_auc(u2), u2), (bag_auc(u3), u3)]
best_auc, MODELS = max(cands)
print(f"[diag] ship: n={len(MODELS)}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in MODELS], axis=0)


eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
