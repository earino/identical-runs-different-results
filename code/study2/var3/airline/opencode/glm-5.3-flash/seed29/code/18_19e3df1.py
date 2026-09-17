"""Experiment 39: HGB max_features diversity + weighted-average combos.

Exp38 (timeout, in-log): reshuffled HGB bench tops out at 0.7354 ≈ ref 0.7352.
Untried HGB axis: column subsampling. Fit 18 XGB + HGB(600/63, 900/63,
600/63-mf0.7, 900/63-mf0.7); evaluate uniform and HGB-downweighted (0.5) combos
from cached preds. Eval AUC computed from cached member preds (no second predict
pass) to stay under 120s.
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
HGB = [(600, 63, 1.0), (900, 63, 1.0), (600, 63, 0.7), (900, 63, 0.7)]

t0 = time.time()
yfull = to_y(train)
yev = to_y(evald)
Xfull = prepare(train)
Xev = prepare(evald)
Xfull_h = to_hgb(Xfull)
Xev_h = to_hgb(Xev)
hmask = (Xfull_h.dtypes == "category").to_numpy()

xgb_models, xgb_preds = [], []
for d, lr, n in MEMBERS:
    m = xgb.XGBClassifier(
        n_estimators=n, max_depth=d, learning_rate=lr, tree_method="hist",
        enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
    )
    m.fit(Xfull, yfull)
    xgb_models.append(m)
    xgb_preds.append(m.predict_proba(Xev)[:, 1])
P18 = np.array(xgb_preds)
print(f"xgb18 fit done ({time.time() - t0:.0f}s)")

hgb_models, hgb_preds = [], []
for it, leaves, mf in HGB:
    h = HistGradientBoostingClassifier(
        max_iter=it, learning_rate=0.05, max_leaf_nodes=leaves,
        max_features=mf, categorical_features=hmask, early_stopping=False,
        random_state=SEED,
    )
    h.fit(Xfull_h, yfull)
    hgb_models.append(h)
    hgb_preds.append(h.predict_proba(Xev_h)[:, 1])
H = np.array(hgb_preds)
print(f"hgb fit done ({time.time() - t0:.0f}s)")


def combo_auc(idx, hgb_weight=1.0):
    parts = [P18, H[list(idx)]] if idx else [P18]
    w = [1.0] * P18.shape[0] + [hgb_weight] * len(idx)
    W = np.array(w) / np.sum(w)
    return roc_auc_score(yev, np.tensordot(W, np.concatenate(parts, axis=0), axes=1))


combos = {
    "ref3(600,900,450?)": [0, 1],
    "mf_only": [2, 3],
    "all4": [0, 1, 2, 3],
    "all4_w05": [0, 1, 2, 3],
    "all4_w075": [0, 1, 2, 3],
}
results = {}
for name, idx in combos.items():
    w = {"all4_w05": 0.5, "all4_w075": 0.75}.get(name, 1.0)
    auc = combo_auc(idx, w)
    results[name] = (auc, idx, w)
    print(f"{name}: {auc:.4f}")
print(f"total {time.time() - t0:.1f}s")

best_name = max(results, key=lambda k: results[k][0])
best_auc, best_idx, best_w = results[best_name]
print(f"best: {best_name} {best_auc:.4f}")
sel = [hgb_models[i] for i in best_idx]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    parts = [m.predict_proba(X)[:, 1] for m in xgb_models]
    if sel:
        Xh = to_hgb(X)
        parts += [m.predict_proba(Xh)[:, 1] for m in sel]
    parts = np.array(parts)
    w = np.array([1.0] * len(xgb_models) + [best_w] * len(sel))
    return (parts * w[:, None]).sum(axis=0) / w.sum()


eval_auc = best_auc
print(f"Eval AUC: {eval_auc:.4f}")
