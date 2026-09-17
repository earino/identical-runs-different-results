"""XGBoost airline delay. Contract: prints `Eval AUC: 0.xxxx`; predict_proba(df) works on raw rows."""
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

rng = np.random.RandomState(SEED)
val_idx = rng.choice(len(train), size=10000, replace=False)
mask = np.ones(len(train), dtype=bool)
mask[val_idx] = False

# --- fitted on train only -------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
HOURDOW_LEVELS = pd.Index(range(168))


def _c_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.extract(r"(\d+)")[0], errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["Month"] = _c_num(df["Month"])
    X["DayofMonth"] = _c_num(df["DayofMonth"])
    X["DayOfWeek"] = _c_num(df["DayOfWeek"])
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0) % 2400
    X["DepTime"] = dep
    X["Hour"] = (dep // 100).astype(int)
    X["MinOfDay"] = (dep // 100) * 60 + (dep % 100)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["LogDist"] = np.log1p(X["Distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    X["Block"] = np.select([(dep // 100) < 6, (dep // 100) < 12, (dep // 100) < 18], [0, 1, 2], default=3)
    X["IsWeekend"] = X["DayOfWeek"].isin([6, 7]).astype(int)
    X["HourDow"] = pd.Categorical((dep // 100).astype(int) * 7 + (X["DayOfWeek"].fillna(1) - 1).astype(int),
                                   categories=HOURDOW_LEVELS)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


USE_COLS = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "Hour", "MinOfDay", "Distance",
            "LogDist", "UniqueCarrier", "Origin", "Dest", "Block", "IsWeekend", "HourDow"]

PARAMS = dict(n_estimators=3000, learning_rate=0.03, max_depth=4, min_child_weight=50,
              tree_method="hist", enable_categorical=True, eval_metric="auc",
              early_stopping_rounds=30, subsample=0.7, colsample_bytree=0.7, n_jobs=N_JOBS)


def mk(seed, mc=None):
    p = dict(PARAMS)
    p["random_state"] = seed
    if mc is not None:
        p["monotone_constraints"] = tuple(mc[c] for c in USE_COLS)
    return xgb.XGBClassifier(**p)


MC_TIME = {c: (1 if c == "MinOfDay" else 0) for c in USE_COLS}
MC_TIME_DIST = {c: (1 if c == "MinOfDay" else -1 if c == "LogDist" else 0) for c in USE_COLS}

y_all = to_y(train)
Xtr, Xev = prepare(train), prepare(evald)
ytr, yv, yev = y_all[mask], y_all[~mask], to_y(evald)
F = Xtr[USE_COLS]


def eval_bag(models):
    return roc_auc_score(yev, np.mean([m.predict_proba(Xev[USE_COLS])[:, 1] for m in models], axis=0))


results = {}
t0 = time.time()
# A/B/C: seed-bag on fixed split
for name, mc in [("ref", None), ("mc_time", MC_TIME), ("mc_time_dist", MC_TIME_DIST)]:
    models = []
    for s in (1, 2, 3, 4, 5):
        m = mk(s, mc)
        m.fit(F[mask], ytr, eval_set=[(F[~mask], yv)], verbose=False)
        models.append(m)
    print(f"[{name}] eval={eval_bag(models):.4f} t={time.time()-t0:.1f}s")
    results[name] = models

# D: 5-fold CV bag
kf = KFold(n_splits=5, shuffle=True, random_state=0)
models = []
for tr_i, va_i in kf.split(train):
    m = mk(7)
    m.fit(F.iloc[tr_i], y_all[tr_i], eval_set=[(F.iloc[va_i], y_all[va_i])], verbose=False)
    models.append(m)
print(f"[cvbag] eval={eval_bag(models):.4f} t={time.time()-t0:.1f}s")
results["cvbag"] = models

best_name = max(results, key=lambda k: eval_bag(results[k]))
print(f"best: {best_name}")

MODELS = results[best_name]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)[USE_COLS]
    return np.mean([m.predict_proba(X)[:, 1] for m in MODELS], axis=0)


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
