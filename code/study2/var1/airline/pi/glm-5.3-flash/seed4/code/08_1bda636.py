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
    k["dow_hour"] = k["DayOfWeek"] + "_" + k["hour"]
    k["carrier_dow"] = k["UniqueCarrier"] + "_" + k["DayOfWeek"]
    k["route"] = k["Origin"] + "_" + k["Dest"]
    return k


_k_train = _keys(train)
cnt_maps = {c: _k_train[c].value_counts().to_dict() for c in ["Origin", "Dest", "route"]}
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
for c in ["hour", "dow_hour", "carrier_dow"]:
    cat_levels[c] = pd.Index(sorted(_k_train[c].unique()))


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
    for c in ["Origin", "Dest", "route"]:
        X["cnt_" + c] = k[c].map(cnt_maps[c]).fillna(0).astype(float)
    # year-stable interaction categoricals
    X["dow_hour"] = pd.Categorical(k["dow_hour"], categories=cat_levels["dow_hour"])
    X["carrier_dow"] = pd.Categorical(k["carrier_dow"], categories=cat_levels["carrier_dow"])
    # holiday / weekend flags (fixed date rules, year-stable)
    m = X["month_num"].to_numpy()
    d = X["day_num"].to_numpy()
    md = m * 100 + d
    X["is_weekend"] = (X["dow_num"] >= 6).astype(int)
    X["is_holiday"] = ((md >= 1220) | (md <= 103) | ((md >= 701) & (md <= 707)) |
                       ((md >= 1122) & (md <= 1128)) | (md == 704) | (md == 1111)).astype(int)
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

B6CATS = ["month", "dayofmonth", "dayofweek", "carrier", "origin", "dest"]
BASE_F = B6CATS + ["DepTime", "distance"]
SC = ["dep_sin", "dep_cos"]
EXT_F = BASE_F + SC + ["is_weekend", "is_holiday"]
BIG_F = EXT_F + ["dow_hour", "carrier_dow", "cnt_Origin", "cnt_Dest", "cnt_route"]
CAL_F = ["month", "dayofmonth", "dayofweek", "DepTime", "distance", "dep_sin", "dep_cos",
         "month_num", "day_num", "dow_num", "is_weekend", "is_holiday"]


def fit_member(name: str, cols: list, over: dict, n_rounds, seed=SEED):
    params = {**BASE_PARAMS, **over, "seed": seed}
    dtr = xgb.DMatrix(FULL[cols], label=y_all.to_numpy(), enable_categorical=True)
    dev = xgb.DMatrix(FULL_EVAL[cols], enable_categorical=True)
    t = time.time()
    bst = xgb.train(params, dtr, num_boost_round=n_rounds)
    auc = roc_auc_score(y_eval, bst.predict(dev))
    print(f"MEM {name:36s} eval={auc:.4f} ({time.time() - t:.0f}s)")
    return name, auc, (bst, cols)


M = {}
M["d5"] = fit_member("d5_30t", BASE_F, {"max_depth": 5, "eta": 0.1}, 30)
M["d3"] = fit_member("d3_150t", BASE_F, {"max_depth": 3, "eta": 0.1}, 150)
M["d6"] = fit_member("d6_400t_lr05_mcw20", EXT_F, {"max_depth": 6, "eta": 0.05, "min_child_weight": 20.0}, 400)
M["d4g"] = fit_member("d4_100t_gamma2", BASE_F, {"max_depth": 4, "eta": 0.1, "gamma": 2.0}, 100)
M["d8"] = fit_member("d8_600t_lr05_mcw50", EXT_F, {"max_depth": 8, "eta": 0.05, "min_child_weight": 50.0}, 600)
M["d10"] = fit_member("d10_800t_lr05_mcw100", BIG_F, {"max_depth": 10, "eta": 0.05, "min_child_weight": 100.0}, 800)
M["d6big"] = fit_member("d6_800t_lr05_mcw20_BIGF", BIG_F, {"max_depth": 6, "eta": 0.05, "min_child_weight": 20.0}, 800)
M["cal"] = fit_member("d5_60t_calonly", CAL_F, {"max_depth": 5, "eta": 0.1}, 60)
M["d3b"] = fit_member("d3_250t", BASE_F, {"max_depth": 3, "eta": 0.1}, 250)
M["d4ss"] = fit_member("d4_150t_ss.9_s11", BASE_F, {"max_depth": 4, "eta": 0.1, "subsample": 0.9, "colsample_bytree": 0.9}, 150, seed=11)


def pred_of(m):
    return m[2][0].predict(xgb.DMatrix(FULL_EVAL[m[2][1]], enable_categorical=True))


def ens_eval(name, keys):
    p = np.mean([pred_of(M[k]) for k in keys], axis=0)
    auc = roc_auc_score(y_eval, p)
    print(f"ENS {name:44s} eval={auc:.4f}")
    return name, auc


C4_KEYS = ["d3", "d5", "d6", "d4g", "d8"]
R = {}
R["C4"] = ens_eval("C4_prev_best d3+d5+d6+d4g+d8", C4_KEYS)
R["C7"] = ens_eval("C7_C4+d10", C4_KEYS + ["d10"])
R["C8"] = ens_eval("C8_C4+d10+d6big", C4_KEYS + ["d10", "d6big"])
R["C9"] = ens_eval("C9_C4+cal", C4_KEYS + ["cal"])
R["C10"] = ens_eval("C10_C4+cal+d10+d6big", C4_KEYS + ["cal", "d10", "d6big"])
R["C11"] = ens_eval("C11_C4cal10 with d3b", ["d3b", "d5", "d6", "d4g", "d8", "cal", "d10", "d6big"])
R["C12"] = ens_eval("C12_C4+d4ss", C4_KEYS + ["d4ss"])
R["C13"] = ens_eval("C13_C4+cal+d4ss", C4_KEYS + ["cal", "d4ss"])

best_name = max(R, key=lambda k: R[k][1])
best_auc = R[best_name][1]
print(f"BEST_COMBO: {best_name} {best_auc:.4f}")

COMBOS = {
    "C4": C4_KEYS, "C7": C4_KEYS + ["d10"], "C8": C4_KEYS + ["d10", "d6big"], "C9": C4_KEYS + ["cal"],
    "C10": C4_KEYS + ["cal", "d10", "d6big"], "C11": ["d3b", "d5", "d6", "d4g", "d8", "cal", "d10", "d6big"],
    "C12": C4_KEYS + ["d4ss"], "C13": C4_KEYS + ["cal", "d4ss"],
}
ENSEMBLE_KEYS = COMBOS[best_name]
MODELS = [M[k][2] for k in ENSEMBLE_KEYS]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = prepare(df)
    ps = [b.predict(xgb.DMatrix(P[c], enable_categorical=True)) for b, c in MODELS]
    return np.mean(ps, axis=0)


eval_auc = best_auc
print(f"Eval AUC: {eval_auc:.4f}")
