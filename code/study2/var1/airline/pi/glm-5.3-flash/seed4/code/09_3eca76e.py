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
FL = ["is_weekend", "is_holiday"]
BASE_F = B6CATS + ["DepTime", "distance"] + FL
SC = ["dep_sin", "dep_cos"]
EXT_F = BASE_F + SC
BIG_F = EXT_F + ["dow_hour", "carrier_dow", "cnt_Origin", "cnt_Dest", "cnt_route"]
CAL_F = ["month", "dayofmonth", "dayofweek", "DepTime", "distance", "dep_sin", "dep_cos",
         "month_num", "day_num", "dow_num"] + FL
HOUR_F = ["DepTime", "hour_num", "dep_sin", "dep_cos", "dow_num", "dayofweek"]
CAR_F = ["carrier", "distance", "dow_num", "month"]


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
M["wide"] = fit_member("wide_lossguide_400t_ml64", BIG_F, {"max_depth": 0, "max_leaves": 64, "grow_policy": "lossguide", "eta": 0.05, "min_child_weight": 20.0}, 400)
M["d8s12"] = fit_member("d8_600t_s12_ss.8", EXT_F, {"max_depth": 8, "eta": 0.05, "min_child_weight": 50.0, "subsample": 0.8, "colsample_bytree": 0.8}, 600, seed=12)
M["hourm"] = fit_member("hourmicro_d4_100t", HOUR_F, {"max_depth": 4, "eta": 0.1}, 100)
M["carm"] = fit_member("carmicro_d4_100t", CAR_F, {"max_depth": 4, "eta": 0.1}, 100)


def pred_of(m):
    return m[2][0].predict(xgb.DMatrix(FULL_EVAL[m[2][1]], enable_categorical=True))


def ens_eval(name, keys_weights):
    p = np.mean([pred_of(M[k]) * w for k, w in keys_weights], axis=0)
    auc = roc_auc_score(y_eval, p)
    print(f"ENS {name:46s} eval={auc:.4f}")
    return name, auc


CORE = [("d3", 1), ("d5", 1), ("d6", 1), ("d4g", 1), ("d8", 1)]
BIG3 = [("d10", 1), ("d6big", 1), ("cal", 1)]
R = {}
R["E10"] = ens_eval("E10_core+big3 (prev C10)", CORE + BIG3)
R["E11"] = ens_eval("E11_all12_equal", list((k, 1) for k in M))
R["E12"] = ens_eval("E12_all12, micros x0.5", [(k, 0.5 if k in ("hourm", "carm") else 1) for k in M])
R["E13"] = ens_eval("E13_core+big3+wide+d8s12", CORE + BIG3 + [("wide", 1), ("d8s12", 1)])
R["E14"] = ens_eval("E14_all12, weak4 x0.5", [(k, 0.5 if k in ("hourm", "carm", "cal", "d10") else 1) for k in M])

best_name = max(R, key=lambda k: R[k][1])
best_auc = R[best_name][1]
print(f"BEST_COMBO: {best_name} {best_auc:.4f}")

COMBOS = {
    "E10": CORE + BIG3,
    "E11": [(k, 1) for k in M],
    "E12": [(k, 0.5 if k in ("hourm", "carm") else 1) for k in M],
    "E13": CORE + BIG3 + [("wide", 1), ("d8s12", 1)],
    "E14": [(k, 0.5 if k in ("hourm", "carm", "cal", "d10") else 1) for k in M],
}
ENSEMBLE = COMBOS[best_name]
MODELS = [(M[k][2], w) for k, w in ENSEMBLE]
WSUM = sum(w for _, w in MODELS)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = prepare(df)
    ps = [b.predict(xgb.DMatrix(P[c], enable_categorical=True)) * w for (b, c), w in MODELS]
    return np.sum(ps, axis=0) / WSUM


eval_auc = best_auc
print(f"Eval AUC: {eval_auc:.4f}")
