"""XGBoost — experiment 4: additive feature-group ablation at baseline complexity (30 trees, d6, lr .1).
Groups: RAW (baseline cols) / +TIME / +ROUTE / +FREQ; best combo kept.
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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: sorted(train[c].astype(str).unique()) for c in CAT_COLS}
route_levels = sorted((train["Origin"].astype(str) + ">" + train["Dest"].astype(str)).unique())
freq_maps = {c: train[c].value_counts(normalize=True) for c in ["UniqueCarrier", "Origin", "Dest"]}
freq_maps["Route"] = (train["Origin"].astype(str) + ">" + train["Dest"].astype(str)).value_counts(normalize=True)
CUMDAYS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    m = df["Month"].str[2:].astype(int).to_numpy()
    dom = df["DayofMonth"].str[2:].astype(int).to_numpy()
    dow = df["DayOfWeek"].str[2:].astype(int).to_numpy()
    dt = df["DepTime"].astype("int64").to_numpy()
    dep_min = (dt // 100) * 60 + dt % 100
    doy = CUMDAYS[m - 1] + dom
    X["DepTime"] = dt
    X["Distance"] = df["Distance"].to_numpy()
    X["dep_min"] = dep_min
    X["dep_sin"] = np.sin(2 * np.pi * dep_min / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * dep_min / 1440.0)
    X["DepTime_mod"] = dt % 100
    X["hour"] = dt // 100
    X["doy"] = doy
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    X["dow"] = dow
    X["log_dist"] = np.log1p(df["Distance"].to_numpy())
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    X["Route"] = pd.Categorical(
        df["Origin"].astype(str) + ">" + df["Dest"].astype(str), categories=route_levels
    )
    X["o_freq"] = freq_maps["Origin"].reindex(df["Origin"]).fillna(0.0).to_numpy()
    X["d_freq"] = freq_maps["Dest"].reindex(df["Dest"]).fillna(0.0).to_numpy()
    X["c_freq"] = freq_maps["UniqueCarrier"].reindex(df["UniqueCarrier"]).fillna(0.0).to_numpy()
    X["r_freq"] = freq_maps["Route"].reindex(df["Origin"].astype(str) + ">" + df["Dest"].astype(str)).fillna(0.0).to_numpy()
    return X


GROUPS = {
    "RAW": ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "DepTime", "Distance"],
    "TIME": ["dep_min", "dep_sin", "dep_cos", "DepTime_mod", "hour", "doy", "doy_sin", "doy_cos", "dow", "log_dist"],
    "ROUTE": ["Route"],
    "FREQ": ["o_freq", "d_freq", "c_freq", "r_freq"],
}
SETS = [
    ("raw", ["RAW"]),
    ("raw+time", ["RAW", "TIME"]),
    ("raw+route", ["RAW", "ROUTE"]),
    ("raw+freq", ["RAW", "FREQ"]),
    ("raw+time+route", ["RAW", "TIME", "ROUTE"]),
    ("all", ["RAW", "TIME", "ROUTE", "FREQ"]),
]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


X_full, y_all = prepare(train), to_y(train)
X_ev_full, y_ev = prepare(evald), to_y(evald)

results, best = [], None
for name, groups in SETS:
    cols = [c for g in groups for c in GROUPS[g]]
    m = xgb.XGBClassifier(
        n_estimators=30, max_depth=6, learning_rate=0.1, tree_method="hist",
        enable_categorical=True, random_state=SEED, n_jobs=N_JOBS, eval_metric="auc",
    )
    m.fit(X_full[cols], y_all)
    auc = roc_auc_score(y_ev, m.predict_proba(X_ev_full[cols])[:, 1])
    results.append((auc, name, groups))
    print(f"Eval AUC: {auc:.4f}   features={name}")
    if best is None or auc > best[0]:
        best = (auc, cols, name, m)

# also n=100 for the two best feature sets
for auc0, name0, groups0 in sorted(results, reverse=True)[:2]:
    cols = [c for g in groups0 for c in GROUPS[g]]
    m = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS, eval_metric="auc",
    )
    m.fit(X_full[cols], y_all)
    auc = roc_auc_score(y_ev, m.predict_proba(X_ev_full[cols])[:, 1])
    print(f"Eval AUC: {auc:.4f}   features={name0} n=100 d6 ss.8 cs.8")
    if auc > best[0]:
        best = (auc, cols, f"{name0} n100", m)

model = best[3]
print(f"best: {best[2]}")
print(f"Eval AUC: {best[0]:.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df)[best[1]])[:, 1]
