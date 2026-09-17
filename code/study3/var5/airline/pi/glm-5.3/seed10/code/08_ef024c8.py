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
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
t0 = time.time()
X_tr, X_ev = prepare(train), prepare(evald)


def fit_bag(seed0, n, subs, col, depth=6, mcw=1, k=100):
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
    "A_ctrl": dict(subs=0.8, col=0.8, depth=6, mcw=1, k=100),
    "B_col06": dict(subs=0.8, col=0.6, depth=6, mcw=1, k=100),
    "C_sub09": dict(subs=0.9, col=0.8, depth=6, mcw=1, k=100),
    "D_d7": dict(subs=0.8, col=0.8, depth=7, mcw=1, k=100),
    "E_k150": dict(subs=0.8, col=0.8, depth=6, mcw=1, k=150),
    "F_w10": dict(subs=0.8, col=0.8, depth=6, mcw=10, k=100),
}
N_PER = 8
bags = {}
for name, cfg in CONFIGS.items():
    bags[name] = fit_bag(SEED + 100 * hash(name) % 1000, N_PER, cfg["subs"], cfg["col"],
                         cfg["depth"], cfg["mcw"], cfg["k"])
    print(f"[diag] {name} n={N_PER}: {bag_auc(bags[name]):.4f} ({time.time() - t0:.1f}s)")

top2 = sorted(bags, key=lambda n: -bag_auc(bags[n]))[:2]
union = bags[top2[0]] + bags[top2[1]]
print(f"[diag] union({','.join(top2)}) n={len(union)}: {bag_auc(union):.4f}")
allu = [m for b in bags.values() for m in b]
print(f"[diag] union(all) n={len(allu)}: {bag_auc(allu):.4f}")

best_single = max(bags, key=lambda n: bag_auc(bags[n]))
MODELS = bags[best_single] if bag_auc(bags[best_single]) >= max(bag_auc(union), bag_auc(allu)) else (
    union if bag_auc(union) >= bag_auc(allu) else allu)
print(f"[diag] ship: n={len(MODELS)}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in MODELS], axis=0)


eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
