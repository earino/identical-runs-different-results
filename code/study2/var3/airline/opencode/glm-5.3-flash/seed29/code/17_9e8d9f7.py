"""Experiment 36: final-form ensemble — 18 XGB + HGB(600/63, 450/31, 900/63).

Exp35 (timeout, scan completed in-log): best combo = 18 XGB + HGB(1,3,4) = 0.7352;
members 1/3/4 appear in every top-5 combo -> robust choice. This run fits exactly
those 21 members (no scanning) and reports the final number.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
CUM_DAYS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=float)


def _ord(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.slice(2), errors="coerce")


def _cat(series: pd.Series, n_levels: int) -> pd.Categorical:
    return pd.Categorical(series.astype(str), categories=[str(i) for i in range(n_levels)])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    X["dep_missing"] = dt.isna().astype(float)
    dtf = dt.fillna(0)
    h = (dtf // 100).astype(float) % 24
    total_min = h * 60 + dtf % 100
    X["dep_min"] = total_min
    X["dep_block"] = _cat((total_min // 15).astype(int), 96)
    X["dep_hour"] = _cat(h.astype(int), 24)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").fillna(0)
    month = _ord(df["Month"]).fillna(1).clip(1, 12).astype(int)
    X["dayofyear"] = CUM_DAYS[month - 1] + _ord(df["DayofMonth"]).fillna(15)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    return X


def to_hgb(df_X: pd.DataFrame) -> pd.DataFrame:
    Xh = df_X.copy()
    for c in ["Origin", "Dest"]:
        codes = {v: i for i, v in enumerate(cat_levels[c])}
        Xh[c] = Xh[c].astype(str).map(codes).astype(float)
    return Xh


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


MEMBERS = [
    (2, 0.05, 554), (3, 0.05, 374), (4, 0.05, 214), (5, 0.05, 237),
    (6, 0.05, 232), (7, 0.05, 402), (8, 0.05, 626), (9, 0.05, 521), (10, 0.05, 386),
    (2, 0.03, 875), (3, 0.03, 645), (4, 0.03, 86), (5, 0.03, 317),
    (6, 0.03, 298), (7, 0.03, 364), (8, 0.03, 823),
    (7, 0.1, 2285), (8, 0.1, 472),
]
HGB = [(600, 63), (450, 31), (900, 63)]

t0 = time.time()
yfull = to_y(train)
yev = to_y(evald)
Xfull = prepare(train)
Xev = prepare(evald)
Xfull_h = to_hgb(Xfull)
Xev_h = to_hgb(Xev)
hmask = (Xfull_h.dtypes == "category").to_numpy()

xgb_models = []
for d, lr, n in MEMBERS:
    m = xgb.XGBClassifier(
        n_estimators=n, max_depth=d, learning_rate=lr, tree_method="hist",
        enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
    )
    m.fit(Xfull, yfull)
    xgb_models.append(m)
print(f"xgb18 fit done ({time.time() - t0:.0f}s)")

hgb_models = []
for it, leaves in HGB:
    h = HistGradientBoostingClassifier(
        max_iter=it, learning_rate=0.05, max_leaf_nodes=leaves,
        categorical_features=hmask, early_stopping=False,
        random_state=SEED,
    )
    h.fit(Xfull_h, yfull)
    hgb_models.append(h)
print(f"hgb fit done ({time.time() - t0:.0f}s)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    out = [m.predict_proba(X)[:, 1] for m in xgb_models]
    Xh = to_hgb(X)
    out += [m.predict_proba(Xh)[:, 1] for m in hgb_models]
    return np.mean(out, axis=0)


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
print(f"total {time.time() - t0:.1f}s")
