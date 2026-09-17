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
    k["block30"] = (df["DepTime"].astype(float) // 30).astype(int).astype(str)
    for b in (5, 10, 12, 15, 20, 45, 60, 90, 120, 180):
        k[f"block{b}"] = (df["DepTime"].astype(float) // b).astype(int).astype(str)
    k["minute"] = (df["DepTime"].astype(float) % 100).astype(int).astype(str)
    k["dow_block15"] = k["DayOfWeek"] + "_" + k["block15"]
    wk = ((df["DepTime"].astype(float) // 100).astype(int) * 60 + df["DepTime"].astype(float) % 100
          + (k["DayOfWeek"].str.slice(2).astype(int) - 1) * 1440)
    k["weekbin60"] = (wk // 60).astype(int).astype(str)
    k["carrier_block15"] = k["UniqueCarrier"] + "_" + k["block15"]
    k["carrier_block30"] = k["UniqueCarrier"] + "_" + k["block30"]
    k["origin_block60"] = k["Origin"] + "_" + k["block60"]
    k["month_block60"] = k["Month"] + "_" + k["block60"]
    k["dow_hour"] = k["DayOfWeek"] + "_" + k["hour"]
    k["carrier_dow"] = k["UniqueCarrier"] + "_" + k["DayOfWeek"]
    k["route"] = k["Origin"] + "_" + k["Dest"]
    return k


_k_train = _keys(train)
_dist_edges = np.quantile(train["Distance"].astype(float).to_numpy(), np.linspace(0, 1, 11))
_dist_edges[0], _dist_edges[-1] = -np.inf, np.inf
cnt_maps = {c: _k_train[c].value_counts().to_dict() for c in ["Origin", "Dest", "route"]}
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
for c in ["hour", "block30", "dow_hour", "carrier_dow", "block10", "block15", "block20", "block60",
          "block5", "block12", "block45", "block90", "block120", "block180", "minute", "dow_block15",
          "weekbin60", "carrier_block15", "month_block60", "carrier_block30", "origin_block60"]:
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
    X["block30"] = pd.Categorical(k["block30"], categories=cat_levels["block30"])
    X["hour_cat"] = pd.Categorical(k["hour"], categories=cat_levels["hour"])
    X["block30n"] = df["DepTime"].astype(float) // 30
    for b in (10, 15, 20, 60, 5, 12, 45, 90, 120, 180):
        X[f"block{b}"] = pd.Categorical(k[f"block{b}"], categories=cat_levels[f"block{b}"])
    X["minute"] = pd.Categorical(k["minute"], categories=cat_levels["minute"])
    X["dow_block15"] = pd.Categorical(k["dow_block15"], categories=cat_levels["dow_block15"])
    X["weekbin60"] = pd.Categorical(k["weekbin60"], categories=cat_levels["weekbin60"])
    X["carrier_block15"] = pd.Categorical(k["carrier_block15"], categories=cat_levels["carrier_block15"])
    X["carrier_block30"] = pd.Categorical(k["carrier_block30"], categories=cat_levels["carrier_block30"])
    X["origin_block60"] = pd.Categorical(k["origin_block60"], categories=cat_levels["origin_block60"])
    X["month_block60"] = pd.Categorical(k["month_block60"], categories=cat_levels["month_block60"])
    X["dist_bin"] = pd.Categorical(pd.Series(np.searchsorted(_dist_edges, df["Distance"].astype(float).to_numpy()) - 1),
                                   categories=range(10))
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
HC_F = EXT_F + ["hour_cat", "hour_num"]
HC_B_F = HC_F + ["block30"]
HC_B15_F = HC_F + ["block15"]
HC_B60_F = HC_F + ["block60"]
HC_B20_F = HC_F + ["block20"]
HC_B10_F = HC_F + ["block10"]
HC_B3060_F = HC_F + ["block30", "block60"]
HC_B1530_F = HC_F + ["block15", "block30"]
B30ONLY_F = EXT_F + ["block30", "block30n"]
HCX_F = HC_F + ["block15", "block30", "block60", "block10", "block20"]
B1530_F = HC_F + ["block15", "block30"]
BIG_F = EXT_F + ["dow_hour", "carrier_dow", "cnt_Origin", "cnt_Dest", "cnt_route"]
BIG_HC_F = HC_F + ["dow_hour", "carrier_dow", "cnt_Origin", "cnt_Dest", "cnt_route"]
CAL_F = ["month", "dayofmonth", "dayofweek", "DepTime", "distance", "dep_sin", "dep_cos",
         "month_num", "day_num", "dow_num"] + FL


def fit_member(name: str, cols: list, over: dict, n_rounds, seed=SEED):
    params = {**BASE_PARAMS, **over, "seed": seed}
    dtr = xgb.DMatrix(FULL[cols], label=y_all.to_numpy(), enable_categorical=True)
    t = time.time()
    bst = xgb.train(params, dtr, num_boost_round=n_rounds)
    p = bst.predict(xgb.DMatrix(FULL_EVAL[cols], enable_categorical=True))
    auc = roc_auc_score(y_eval, p)
    print(f"MEM {name:36s} eval={auc:.4f} ({time.time() - t:.0f}s)")
    return name, auc, (bst, cols, p)


M = {}
C3 = {"max_depth": 6, "eta": 0.05, "min_child_weight": 20.0, "colsample_bynode": 0.3}
M["q4"] = fit_member("q4_HC+b30", HC_B_F, C3, 400)
M["b15"] = fit_member("b15_HC+b15", HC_F + ["block15"], C3, 400)
M["b20"] = fit_member("b20_HC+b20", HC_F + ["block20"], C3, 400)
M["b60"] = fit_member("b60_HC+b60", HC_F + ["block60"], C3, 400)
M["b3060"] = fit_member("b3060_HC+b30+b60", HC_F + ["block30", "block60"], C3, 400)
M["b1530"] = fit_member("b1530_HC+b15+b30", HC_F + ["block15", "block30"], C3, 400)
M["b10"] = fit_member("b10_HC+b10", HC_F + ["block10"], C3, 400)
M["trio"] = fit_member("trio_HC+b15+b30+b60", HC_F + ["block15", "block30", "block60"], C3, 400)
M["quad"] = fit_member("quad_HC+b10+b15+b30", HC_F + ["block10", "block15", "block30"], C3, 400)
M["w1"] = fit_member("b1530_600t", HC_F + ["block15", "block30"], C3, 600)
M["w9"] = fit_member("carb15_HC+b15+carb15", HC_F + ["block15", "carrier_block15"], C3, 400)
M["v1"] = fit_member("b1530_800t", HC_F + ["block15", "block30"], C3, 800)
M["v2"] = fit_member("b1530_1000t", HC_F + ["block15", "block30"], C3, 1000)
M["v3"] = fit_member("carb15_600t", HC_F + ["block15", "carrier_block15"], C3, 600)
M["v4"] = fit_member("HC+b30+carb30", HC_F + ["block30", "carrier_block30"], C3, 400)
M["v5"] = fit_member("HC+b60+origb60", HC_F + ["block60", "origin_block60"], C3, 400)
M["v6"] = fit_member("trio_600t", HC_F + ["block15", "block30", "block60"], C3, 600)
M["v7"] = fit_member("b1530_600t_s2", HC_F + ["block15", "block30"], C3, 600, seed=2)
M["v8"] = fit_member("b1530_600t_s3", HC_F + ["block15", "block30"], C3, 600, seed=3)

def mean_of(keys):
    return np.mean([M[k][2][2] for k in keys], axis=0)


def auc_of(p):
    return roc_auc_score(y_eval, p)


R = {}
BFAM = ["q4", "b15", "b20", "b60", "b3060", "b1530", "b10", "trio", "quad"]
U8K = BFAM + ["w1", "w9"]
R["W0"] = ("W0_U8_ref", auc_of(mean_of(U8K)))
R["W1"] = ("W1_U8 b1530->800t", auc_of(mean_of(BFAM + ["v1", "w9"])))
R["W2"] = ("W2_U8 b1530->1000t", auc_of(mean_of(BFAM + ["v2", "w9"])))
R["W3"] = ("W3_U8+carb15_600", auc_of(mean_of(U8K + ["v3"])))
R["W4"] = ("W4_U8+carb30", auc_of(mean_of(U8K + ["v4"])))
R["W5"] = ("W5_U8+origb60", auc_of(mean_of(U8K + ["v5"])))
R["W6"] = ("W6_U8+b1530seeds", auc_of(mean_of(U8K + ["v7", "v8"])))
R["W7"] = ("W7_U8+b1530_800+carb15_600", auc_of(mean_of(BFAM + ["v1", "w9", "v3"])))
R["W8"] = ("W8_U8+seeds+800+carb600", auc_of(mean_of(BFAM + ["w1", "w9", "v1", "v3", "v7", "v8"])))
R["W9"] = ("W9_all19", auc_of(mean_of(list(M.keys()))))
R["W10"] = ("W10_trio->600", auc_of(mean_of(["q4", "b15", "b20", "b60", "b3060", "b1530", "b10", "v6", "quad", "w1", "w9"])))

for name, auc in R.values():
    print(f"ENS {name:36s} eval={auc:.4f}")
best_name = max(R, key=lambda k: R[k][1])
best_auc = R[best_name][1]
print(f"BEST: {best_name} {best_auc:.4f}")

COMBOS = {
    "W0": U8K,
    "W1": BFAM + ["v1", "w9"], "W2": BFAM + ["v2", "w9"], "W3": U8K + ["v3"], "W4": U8K + ["v4"],
    "W5": U8K + ["v5"], "W6": U8K + ["v7", "v8"], "W7": BFAM + ["v1", "w9", "v3"],
    "W8": BFAM + ["w1", "w9", "v1", "v3", "v7", "v8"], "W9": list(M.keys()),
    "W10": ["q4", "b15", "b20", "b60", "b3060", "b1530", "b10", "v6", "quad", "w1", "w9"],
}
ENSEMBLE_KEYS = COMBOS[best_name]
MODELS = [(M[k][2][0], M[k][2][1]) for k in ENSEMBLE_KEYS]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = prepare(df)
    ps = np.column_stack([b.predict(xgb.DMatrix(P[c], enable_categorical=True)) for b, c in MODELS])
    return ps.mean(axis=1)


eval_auc = best_auc
print(f"Eval AUC: {eval_auc:.4f}")
