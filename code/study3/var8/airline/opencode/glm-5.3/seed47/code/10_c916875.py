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
_org_counts = train["Origin"].value_counts()
_dest_counts = train["Dest"].value_counts()
_route_counts = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).value_counts()
_carrier_hour = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_" +
                                 (pd.to_numeric(train["DepTime"], errors="coerce").fillna(0) % 2400 // 100)
                                 .astype(int).astype(str)).unique()))


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
    X["MonthSin"] = np.sin(2 * np.pi * X["Month"] / 12)
    X["MonthCos"] = np.cos(2 * np.pi * X["Month"] / 12)
    X["OrgTraffic"] = np.log1p(_org_counts.reindex(df["Origin"].to_numpy()).to_numpy())
    X["DestTraffic"] = np.log1p(_dest_counts.reindex(df["Dest"].to_numpy()).to_numpy())
    X["RouteTraffic"] = np.log1p(_route_counts.reindex(
        (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).to_numpy()).to_numpy())
    ch = df["UniqueCarrier"].astype(str) + "_" + (dep // 100).astype(int).astype(str)
    X["CarrierHour"] = pd.Categorical(ch, categories=_carrier_hour)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


BASE = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "Hour", "MinOfDay", "Distance",
        "LogDist", "UniqueCarrier", "Origin", "Dest", "Block", "IsWeekend", "HourDow"]
VOLUME = BASE + ["OrgTraffic", "DestTraffic", "RouteTraffic"]
CH = BASE + ["CarrierHour"]
SC = BASE + ["MonthSin", "MonthCos"]
ALLNEW = BASE + ["OrgTraffic", "DestTraffic", "RouteTraffic", "CarrierHour", "MonthSin", "MonthCos"]

PARAMS = dict(n_estimators=3000, learning_rate=0.03, max_depth=4, min_child_weight=50,
              tree_method="hist", enable_categorical=True, eval_metric="auc",
              early_stopping_rounds=30, subsample=0.7, colsample_bytree=0.7, n_jobs=N_JOBS)


def mk(seed=7):
    return xgb.XGBClassifier(**PARAMS, random_state=seed)


y_all = to_y(train)
Xtr, Xev = prepare(train), prepare(evald)
yev = to_y(evald)


def eval_bag(models, cols):
    return roc_auc_score(yev, np.mean([m.predict_proba(Xev[cols])[:, 1] for m in models], axis=0))


results = {}
t0 = time.time()


def run_cvbag(name, cols, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    F = Xtr[cols]
    models = []
    for tr_i, va_i in kf.split(train):
        m = mk()
        m.fit(F.iloc[tr_i], y_all[tr_i], eval_set=[(F.iloc[va_i], y_all[va_i])], verbose=False)
        models.append(m)
    print(f"[{name}] eval={eval_bag(models, cols):.4f} t={time.time()-t0:.1f}s")
    results[name] = (models, cols)


run_cvbag("ref", BASE)
run_cvbag("volume", VOLUME)
run_cvbag("carrierhour", CH)
run_cvbag("season", SC)
run_cvbag("allnew", ALLNEW)

best_name = max(results, key=lambda k: eval_bag(*results[k]))
print(f"best: {best_name}")

MODELS, USE_COLS = results[best_name]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)[USE_COLS]
    return np.mean([m.predict_proba(X)[:, 1] for m in MODELS], axis=0)


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
