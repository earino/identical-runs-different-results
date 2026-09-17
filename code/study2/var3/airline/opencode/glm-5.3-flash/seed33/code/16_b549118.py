"""XGBoost binary classifier for airline delays. THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives in prepare(); encoders/statistics are fit on train only.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# Month is DROPPED: month-of-year delay patterns do not transfer from 2005 to 2006.
# DayofMonth is DROPPED: day-of-month carries no transferable signal.
# CarQ(q) = carrier x departure-hour x q-minute slot. Members use different slot
# sizes (15/20/30 min): unregularized members do best with coarse slots, L1-
# regularized members with fine slots, and mixing both axes adds up.
CAT_COLS = ["DayOfWeek", "UniqueCarrier", "Origin", "Dest", "DepHour"]
CARQ_BINS = (15, 20, 30, 45)
ORIGQ_BINS = (15, 30, 45)
DESTQ_BINS = (30, 45)
ROUTEQ_BINS = (30,)
FEATS = (
    CAT_COLS
    + ["DepTime", "DepMin", "Distance"]
    + [f"CarQ{q}" for q in CARQ_BINS]
    + [f"OrigQ{q}" for q in ORIGQ_BINS]
    + [f"DestQ{q}" for q in DESTQ_BINS]
    + [f"RouteQ{q}" for q in ROUTEQ_BINS]
    + ["CarD", "Route"]
)
INTERACT_PREFIX = ("CarQ", "OrigQ", "DestQ", "RouteQ", "CarD", "Route")


def build_levels() -> dict:
    lv = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS if c != "DepHour"}
    h = ((train["DepTime"] // 100).clip(0, 29)).astype(str)
    lv["DepHour"] = pd.Index(sorted(h.unique()))
    o = train["Origin"].astype(str)
    d = train["Dest"].astype(str)
    u = train["UniqueCarrier"].astype(str)
    for q in CARQ_BINS:
        s = (train["DepTime"] % 100) // q
        lv[f"CarQ{q}"] = pd.Index(sorted((u + "_" + h + "_" + s.astype(str)).unique()))
    for q in ORIGQ_BINS:
        s = (train["DepTime"] % 100) // q
        lv[f"OrigQ{q}"] = pd.Index(sorted((o + "_" + h + "_" + s.astype(str)).unique()))
    for q in DESTQ_BINS:
        s = (train["DepTime"] % 100) // q
        lv[f"DestQ{q}"] = pd.Index(sorted((d + "_" + h + "_" + s.astype(str)).unique()))
    lv["CarD"] = pd.Index(sorted((u + "_" + (train["Distance"] // 250).astype(str)).unique()))
    lv["Route"] = pd.Index(sorted((o + "_" + d).unique()))
    for q in ROUTEQ_BINS:
        s = (train["DepTime"] % 100) // q
        lv[f"RouteQ{q}"] = pd.Index(sorted((o + "_" + d + "_" + h + "_" + s.astype(str)).unique()))
    return lv


CAT_LEVELS = build_levels()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    h = ((df["DepTime"] // 100).clip(0, 29)).astype(str)
    dt = df["DepTime"].astype(int)
    for c in CAT_COLS:
        vals = h if c == "DepHour" else df[c].astype(str)
        X[c] = pd.Categorical(vals, categories=CAT_LEVELS[c])  # unseen levels -> NaN
    X["DepTime"] = dt
    X["DepMin"] = dt % 100
    X["Distance"] = df["Distance"].astype(int)
    o = df["Origin"].astype(str)
    for q in CARQ_BINS:
        s = (dt % 100) // q
        vals = df["UniqueCarrier"].astype(str) + "_" + h + "_" + s.astype(str)
        X[f"CarQ{q}"] = pd.Categorical(vals, categories=CAT_LEVELS[f"CarQ{q}"])
    for q in ORIGQ_BINS:
        s = (dt % 100) // q
        vals = o + "_" + h + "_" + s.astype(str)
        X[f"OrigQ{q}"] = pd.Categorical(vals, categories=CAT_LEVELS[f"OrigQ{q}"])
    for q in DESTQ_BINS:
        s = (dt % 100) // q
        vals = df["Dest"].astype(str) + "_" + h + "_" + s.astype(str)
        X[f"DestQ{q}"] = pd.Categorical(vals, categories=CAT_LEVELS[f"DestQ{q}"])
    X["CarD"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "_" + (df["Distance"] // 250).astype(str),
        categories=CAT_LEVELS["CarD"],
    )
    X["Route"] = pd.Categorical(
        df["Origin"].astype(str) + "_" + df["Dest"].astype(str),
        categories=CAT_LEVELS["Route"],
    )
    r = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    for q in ROUTEQ_BINS:
        s = (dt % 100) // q
        X[f"RouteQ{q}"] = pd.Categorical(r + "_" + h + "_" + s.astype(str), categories=CAT_LEVELS[f"RouteQ{q}"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Xtr, Xev = prepare(train), prepare(evald)
ytr = to_y(train)

# Lean 6-member ensemble (greedy forward/backward-selected from a 45+-member
# diverse candidate pool, LOO-verified: every member contributes >= 0.0002).
# Diversity comes from: max_bin=1024 vs default histograms, depth 6 vs 8,
# L1 regularization on/off, max_cat_threshold=32, carrier x distance-bucket,
# route x hour x slot interaction, and different seeds.
_A = dict(n_estimators=300, max_depth=6, learning_rate=0.10)
_B = dict(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.7)
_A1 = dict(_A, reg_alpha=1)
MEMBERS = [
    (dict(_A1, max_bin=1024), 45, "OrigQ", 42),
    (dict(_A1, max_depth=8), 20, None, 42),
    (_A, 30, "RouteQ", 42),
    (_A1, 30, "CarD", 42),
    (dict(_A1, max_cat_threshold=32), 45, "OrigQ", 42),
    (_B, 30, None, 44),
]


def member_cols(qb, xfeat):
    want = {f"CarQ{qb}"}
    if xfeat:
        want.add(xfeat if xfeat == "CarD" else f"{xfeat}{qb}")
    cols = [c for c in FEATS if not c.startswith(INTERACT_PREFIX) or c in want]
    return cols


def make_model(kw, seed):
    return xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
        **kw,
    )


t0 = time.time()
models = []
for kw, qb, xfeat, seed in MEMBERS:
    cols = member_cols(qb, xfeat)
    m = make_model(kw, seed)
    m.fit(Xtr[cols], ytr)
    models.append((m, cols))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.mean([m.predict_proba(X[cols])[:, 1] for m, cols in models], axis=0)
    return p


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
