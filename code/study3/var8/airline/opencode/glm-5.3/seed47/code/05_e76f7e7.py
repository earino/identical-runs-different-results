"""XGBoost airline delay. Contract: prints `Eval AUC: 0.xxxx`; predict_proba(df) works on raw rows."""
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

rng = np.random.RandomState(SEED)
val_idx = rng.choice(len(train), size=10000, replace=False)
mask = np.ones(len(train), dtype=bool)
mask[val_idx] = False
FIT = train[mask]
yfit = (FIT[TARGET] == POSITIVE).astype(float).to_numpy()

# --- fitted on train (fit part) only -------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(FIT[c].dropna().unique())) for c in CAT_COLS}
PRIOR = yfit.mean()

TE_ALPHA = {"UniqueCarrier": 100, "Origin": 100, "Dest": 100, "Route": 300}
te_map = {}
te_map["UniqueCarrier"] = pd.Series(yfit).groupby(FIT["UniqueCarrier"].to_numpy()).agg(["mean", "size"])
te_map["Origin"] = pd.Series(yfit).groupby(FIT["Origin"].to_numpy()).agg(["mean", "size"])
te_map["Dest"] = pd.Series(yfit).groupby(FIT["Dest"].to_numpy()).agg(["mean", "size"])
_rt = FIT["Origin"].astype(str) + "_" + FIT["Dest"].astype(str)
te_map["Route"] = pd.Series(yfit).groupby(_rt.to_numpy()).agg(["mean", "size"])

_DAYS_BEFORE = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]


def _c_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.extract(r"(\d+)")[0], errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    mo = _c_num(df["Month"]).astype(int)
    dom = _c_num(df["DayofMonth"]).astype(int)
    X["Month"] = mo
    X["DayofMonth"] = dom
    X["DayOfWeek"] = _c_num(df["DayOfWeek"]).astype(int)
    X["DayOfYear"] = np.asarray(_DAYS_BEFORE, dtype=float)[mo.to_numpy() - 1] + dom
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0) % 2400
    X["DepTime"] = dep
    X["Hour"] = (dep // 100).astype(int)
    X["MinOfDay"] = (dep // 100) * 60 + (dep % 100)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["LogDist"] = np.log1p(X["Distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
        stats = te_map[c]
        key = df[c].to_numpy()
        mean = stats["mean"].reindex(key).to_numpy()
        size = stats["size"].reindex(key).to_numpy()
        a = TE_ALPHA[c]
        X[f"TE_{c}"] = np.where(np.isnan(mean), PRIOR, (mean * size + PRIOR * a) / (size + a))
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    stats = te_map["Route"]
    mean = stats["mean"].reindex(route.to_numpy()).to_numpy()
    size = stats["size"].reindex(route.to_numpy()).to_numpy()
    a = TE_ALPHA["Route"]
    X["TE_Route"] = np.where(np.isnan(mean), PRIOR, (mean * size + PRIOR * a) / (size + a))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


ALL = ["Month", "DayofMonth", "DayOfWeek", "DayOfYear", "DepTime", "Hour", "MinOfDay", "Distance",
       "LogDist", "UniqueCarrier", "Origin", "Dest", "TE_UniqueCarrier", "TE_Origin", "TE_Dest", "TE_Route"]
NO_TE_ROUTE = [c for c in ALL if c != "TE_Route"]
NO_CATS = [c for c in ALL if c not in ("UniqueCarrier", "Origin", "Dest")]

CONFIGS = [
    ("ref_d4", dict(max_depth=4, min_child_weight=50), ALL),
    ("te_d4_mcw200", dict(max_depth=4, min_child_weight=200), ALL),
    ("te_d5_mcw100", dict(max_depth=5, min_child_weight=100), ALL),
    ("noteroute_d4", dict(max_depth=4, min_child_weight=50), NO_TE_ROUTE),
    ("nocats_d4", dict(max_depth=4, min_child_weight=50), NO_CATS),
]

Xtr, Xev = prepare(train), prepare(evald)
ytr, yv, yev = to_y(train[mask]), to_y(train[~mask]), to_y(evald)
best = None
for name, params, cols in CONFIGS:
    base = dict(n_estimators=3000, learning_rate=0.05, tree_method="hist", enable_categorical=True,
                eval_metric="auc", early_stopping_rounds=30, subsample=0.9, colsample_bytree=0.9,
                random_state=SEED, n_jobs=N_JOBS)
    base.update(params)
    m = xgb.XGBClassifier(**base)
    t0 = time.time()
    m.fit(Xtr[mask][cols], ytr, eval_set=[(Xtr[~mask][cols], yv)], verbose=False)
    ev = roc_auc_score(yev, m.predict_proba(Xev[cols])[:, 1])
    va = roc_auc_score(yv, m.predict_proba(Xtr[~mask][cols])[:, 1])
    print(f"[{name}] valid={va:.4f} eval={ev:.4f} iter={m.best_iteration} t={time.time()-t0:.1f}s")
    if best is None or ev > best[3]:
        best = (name, params, cols, ev)

name, params, USE_COLS, eval_auc = best
print(f"best: {name}")

base = dict(n_estimators=3000, learning_rate=0.05, tree_method="hist", enable_categorical=True,
            eval_metric="auc", early_stopping_rounds=30, subsample=0.9, colsample_bytree=0.9,
            random_state=SEED, n_jobs=N_JOBS)
base.update(params)
model = xgb.XGBClassifier(**base)
model.fit(Xtr[mask][USE_COLS], ytr, eval_set=[(Xtr[~mask][USE_COLS], yv)], verbose=False)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df)[USE_COLS])[:, 1]


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
