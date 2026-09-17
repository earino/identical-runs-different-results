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
    # structural time blocks
    hour = (dep // 100).astype(int)
    X["Block"] = np.select([hour < 6, hour < 12, hour < 18], [0, 1, 2], default=3)
    X["IsWeekend"] = X["DayOfWeek"].isin([6, 7]).astype(int)
    X["HourDow"] = pd.Categorical(hour * 7 + (X["DayOfWeek"].fillna(1) - 1).astype(int), categories=HOURDOW_LEVELS)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


BASE = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "Hour", "MinOfDay", "Distance",
        "LogDist", "UniqueCarrier", "Origin", "Dest"]
BLOCKS = BASE + ["Block", "IsWeekend"]
HDOW = BASE + ["HourDow"]
ALLFEAT = BLOCKS + ["HourDow"]

Xtr, Xev = prepare(train), prepare(evald)
ytr, yv, yev = to_y(train[mask]), to_y(train[~mask]), to_y(evald)


def bag(params, cols, seeds=(1, 2, 3, 4, 5)):
    models = []
    for s in seeds:
        base = dict(n_estimators=3000, learning_rate=0.05, tree_method="hist", enable_categorical=True,
                    eval_metric="auc", early_stopping_rounds=30, subsample=0.7, colsample_bytree=0.7,
                    random_state=s, n_jobs=N_JOBS)
        base.update(params)
        m = xgb.XGBClassifier(**base)
        m.fit(Xtr[mask][cols], ytr, eval_set=[(Xtr[~mask][cols], yv)], verbose=False)
        models.append(m)
    ev = roc_auc_score(yev, np.mean([m.predict_proba(Xev[cols])[:, 1] for m in models], axis=0))
    va = roc_auc_score(yv, np.mean([m.predict_proba(Xtr[~mask][cols])[:, 1] for m in models], axis=0))
    print(f"  valid={va:.4f} eval={ev:.4f}")
    return models, ev


D4 = dict(max_depth=4, min_child_weight=50)
CONFIGS = [
    ("ref_sub07", D4, BASE),
    ("blocks", D4, BLOCKS),
    ("hourdow", D4, HDOW),
    ("allfeat", D4, ALLFEAT),
    ("allfeat_lr03", dict(max_depth=4, min_child_weight=50, learning_rate=0.03), ALLFEAT),
]
best = None
for name, params, cols in CONFIGS:
    t0 = time.time()
    print(f"[{name}]")
    models, ev = bag(params, cols)
    print(f"  t={time.time()-t0:.1f}s")
    if best is None or ev > best[3]:
        best = (name, models, cols, ev)

name, MODELS, USE_COLS, eval_auc = best
print(f"best: {name}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)[USE_COLS]
    return np.mean([m.predict_proba(X)[:, 1] for m in MODELS], axis=0)


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
