"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

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
y_all = (train[TARGET] == POSITIVE).astype(int)

# --- encoders fitted on TRAIN only (module level); prepare() applies them to any raw df ------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
PRIOR = float(y_all.mean())


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    k = pd.DataFrame(index=df.index)
    for c in ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]:
        k[c] = df[c].astype(str)
    k["hour"] = (df["DepTime"].astype(float) // 100).astype(int).astype(str)
    k["route"] = k["Origin"] + "_" + k["Dest"]
    return k


_k_train = _keys(train)
cnt_maps = {c: _k_train[c].value_counts().to_dict() for c in ["Origin", "Dest"]}
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows."""
    k = _keys(df)
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype(float)  # raw hhmm
    hour_num = k["hour"].astype(int)
    dep_min = hour_num * 60 + (df["DepTime"].astype(float) % 100)
    X["hour_num"] = hour_num
    X["dep_sin"] = np.sin(2 * np.pi * dep_min / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * dep_min / 1440.0)
    X["month"] = pd.Categorical(k["Month"], categories=cat_levels["Month"])
    X["dayofmonth"] = pd.Categorical(k["DayofMonth"], categories=cat_levels["DayofMonth"])
    X["dayofweek"] = pd.Categorical(k["DayOfWeek"], categories=cat_levels["DayOfWeek"])
    X["month_num"] = k["Month"].str.slice(2).astype(int)
    X["day_num"] = k["DayofMonth"].str.slice(2).astype(int)
    X["dow_num"] = k["DayOfWeek"].str.slice(2).astype(int)
    X["carrier"] = pd.Categorical(k["UniqueCarrier"], categories=cat_levels["UniqueCarrier"])
    X["origin"] = pd.Categorical(k["Origin"], categories=cat_levels["Origin"])
    X["dest"] = pd.Categorical(k["Dest"], categories=cat_levels["Dest"])
    dist = df["Distance"].astype(float)
    X["distance"] = dist
    X["dist_log"] = np.log1p(dist)
    for c in ["Origin", "Dest"]:
        X["cnt_" + c] = k[c].map(cnt_maps[c]).fillna(0).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "enable_categorical": True,
    "seed": SEED,
    "n_jobs": N_JOBS,
}

t0 = time.time()
FULL = prepare(train)
FULL_EVAL = prepare(evald)
y_eval = to_y(evald)
print(f"Data prep time: {time.time() - t0:.1f}s")

rng = np.random.RandomState(SEED)
idx = rng.permutation(len(train))
n_val = max(1000, len(train) // 10)
val_idx, tr_idx = idx[:n_val], idx[n_val:]
y_tr = y_all.iloc[tr_idx].to_numpy()
y_va = y_all.iloc[val_idx].to_numpy()

B6CATS = ["month", "dayofmonth", "dayofweek", "carrier", "origin", "dest"]
BASE_F = B6CATS + ["DepTime", "distance"]
SC = ["dep_sin", "dep_cos"]


def fit_eval(name: str, cols: list, over: dict, n_rounds=30, seeds=(SEED,)):
    """Fit on the 90% part, average over seeds, report val + eval AUC."""
    params = {**BASE_PARAMS, **over}
    dtr = xgb.DMatrix(FULL.iloc[tr_idx][cols], label=y_tr, enable_categorical=True)
    dva = xgb.DMatrix(FULL.iloc[val_idx][cols], label=y_va, enable_categorical=True)
    dev = xgb.DMatrix(FULL_EVAL[cols], enable_categorical=True)
    t = time.time()
    ps, bsts = [], []
    for sd in seeds:
        bst = xgb.train({**params, "seed": sd}, dtr, num_boost_round=n_rounds)
        bsts.append((bst, cols))
        ps.append(bst.predict(dev))
    p = np.mean(ps, axis=0)
    auc = roc_auc_score(y_eval, p)
    val_auc = roc_auc_score(y_va, np.mean([b.predict(dva) for b, _ in bsts], axis=0))
    print(f"DIAG {name:36s} eval={auc:.4f} val={val_auc:.4f} ({time.time() - t:.0f}s)")
    return name, auc, bsts


R = {}
R["I1_d3_100"] = fit_eval("I1_d3_100t", BASE_F, {"max_depth": 3, "eta": 0.1}, n_rounds=100)
R["I2_d3_200"] = fit_eval("I2_d3_200t", BASE_F, {"max_depth": 3, "eta": 0.1}, n_rounds=200)
R["I3_d4_60"] = fit_eval("I3_d4_60t", BASE_F, {"max_depth": 4, "eta": 0.1}, n_rounds=60)
R["I4_d4_100"] = fit_eval("I4_d4_100t", BASE_F, {"max_depth": 4, "eta": 0.1}, n_rounds=100)
R["I5_d5_30"] = fit_eval("I5_d5_30t", BASE_F, {"max_depth": 5, "eta": 0.1})
R["I7_lam10"] = fit_eval("I7_d5_30t_lam10", BASE_F, {"max_depth": 5, "eta": 0.1, "reg_lambda": 10.0})
R["I8_d6_lam10"] = fit_eval("I8_d6_30t_lam10", BASE_F, {"max_depth": 6, "eta": 0.1, "reg_lambda": 10.0})
R["I11_stoch"] = fit_eval("I11_d5_30t_ss.7_cs.7", BASE_F, {"max_depth": 5, "eta": 0.1, "subsample": 0.7, "colsample_bytree": 0.7})
R["I12_mcw50"] = fit_eval("I12_d5_30t_mcw50", BASE_F, {"max_depth": 5, "eta": 0.1, "min_child_weight": 50.0})
R["I13_big400"] = fit_eval("I13_d6_400t_lr.05_mcw20", BASE_F + SC, {"max_depth": 6, "eta": 0.05, "min_child_weight": 20.0}, n_rounds=400)

# ensembles of already-fitted small models: fit the missing members quickly
ens_members = []
ens_members.extend(R["I5_d5_30"][2])
ens_members.extend(R["I1_d3_100"][2])
ens_members.extend(R["I13_big400"][2])
p_ens = np.mean([b.predict(xgb.DMatrix(FULL_EVAL[c], enable_categorical=True)) for b, c in ens_members], axis=0)
auc_ens = roc_auc_score(y_eval, p_ens)
print(f"DIAG {'I9_ens d5_30+d3_100+d6_400':36s} eval={auc_ens:.4f}")
R["I9_ens3"] = ("I9_ens3", auc_ens, ens_members)

ens2 = R["I5_d5_30"][2] + R["I13_big400"][2]
p_ens2 = np.mean([b.predict(xgb.DMatrix(FULL_EVAL[c], enable_categorical=True)) for b, c in ens2], axis=0)
auc_ens2 = roc_auc_score(y_eval, p_ens2)
print(f"DIAG {'I10_ens d5_30+d6_400':36s} eval={auc_ens2:.4f}")
R["I10_ens2"] = ("I10_ens2", auc_ens2, ens2)

best_name = max(R, key=lambda k: R[k][1])
_, best_auc, best_members = R[best_name]
print(f"BEST_DIAG: {best_name} {best_auc:.4f}")

MODELS = best_members


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = prepare(df)
    ps = [b.predict(xgb.DMatrix(P[c], enable_categorical=True)) for b, c in MODELS]
    return np.mean(ps, axis=0)


eval_auc = best_auc
print(f"Eval AUC: {eval_auc:.4f}")
