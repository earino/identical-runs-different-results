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
FEATS = CAT_COLS + ["DepTime", "DepMin", "Distance"] + [f"CarQ{q}" for q in CARQ_BINS] + [f"OrigQ{q}" for q in ORIGQ_BINS]


def build_levels() -> dict:
    lv = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS if c != "DepHour"}
    h = ((train["DepTime"] // 100).clip(0, 29)).astype(str)
    lv["DepHour"] = pd.Index(sorted(h.unique()))
    o = train["Origin"].astype(str)
    for q in CARQ_BINS:
        s = (train["DepTime"] % 100) // q
        lv[f"CarQ{q}"] = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_" + h + "_" + s.astype(str)).unique()))
    for q in ORIGQ_BINS:
        s = (train["DepTime"] % 100) // q
        lv[f"OrigQ{q}"] = pd.Index(sorted((o + "_" + h + "_" + s.astype(str)).unique()))
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
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Xtr, Xev = prepare(train), prepare(evald)
ytr = to_y(train)

BASE = [
    dict(n_estimators=300, max_depth=6, learning_rate=0.10),
    dict(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.7),
    dict(n_estimators=600, max_depth=5, learning_rate=0.05, colsample_bytree=0.8),
    dict(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=300, max_depth=6, learning_rate=0.10, subsample=0.8),
]
# 5 unregularized members on CarQ30, 10 L1 members on CarQ15 / CarQ20, plus
# 5 Origin-slot-interaction members (OrigQ) for view diversity.
MEMBERS = (
    [(kw, 30, None) for kw in BASE]
    + [(dict(kw, reg_alpha=1), 15, None) for kw in BASE]
    + [(dict(kw, reg_alpha=1), 20, None) for kw in BASE]
    + [
        (dict(BASE[0], reg_alpha=1), 30, "OrigQ"),
        (dict(BASE[3], reg_alpha=1), 30, "OrigQ"),
        (dict(BASE[1], reg_alpha=1), 15, "OrigQ"),
        (dict(BASE[2], reg_alpha=1), 15, "OrigQ"),
        (dict(BASE[0], reg_alpha=1), 45, "OrigQ"),
    ]
)


def member_cols(qb, xfeat):
    cols = [
        c
        for c in FEATS
        if (not c.startswith("CarQ") or int(c[4:]) == qb)
        and (not c.startswith("OrigQ") or (xfeat == "OrigQ" and int(c[5:]) == qb))
    ]
    return cols


def make_model(kw):
    return xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
        **kw,
    )


t0 = time.time()
models = []
for kw, qb, xfeat in MEMBERS:
    cols = member_cols(qb, xfeat)
    m = make_model(kw)
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
