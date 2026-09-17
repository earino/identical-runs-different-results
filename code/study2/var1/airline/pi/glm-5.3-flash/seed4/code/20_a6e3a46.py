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
          "weekbin60", "carrier_block15", "month_block60"]:
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
M["q4"] = fit_member("q4_d6_400_cbn.3_HC+b30", HC_B_F, C3, 400)
M["b15"] = fit_member("d6_400_cbn.3_HC+b15", HC_B15_F, C3, 400)
M["b60"] = fit_member("d6_400_cbn.3_HC+b60", HC_B60_F, C3, 400)
M["b20"] = fit_member("d6_400_cbn.3_HC+b20", HC_B20_F, C3, 400)
M["b10"] = fit_member("d6_400_cbn.3_HC+b10", HC_B10_F, C3, 400)
M["b3060"] = fit_member("d6_400_cbn.3_HC+b30+b60", HC_B3060_F, C3, 400)
M["b1530"] = fit_member("d6_400_cbn.3_HC+b15+b30", HC_B1530_F, C3, 400)
M["b30only"] = fit_member("d6_400_cbn.3_b30only", B30ONLY_F, C3, 400)
M["q4x6"] = fit_member("q4_600t", HC_B_F, C3, 600)
M["q4c25"] = fit_member("q4_cbn.25", HC_B_F, {**C3, "colsample_bynode": 0.25}, 400)
M["q4w50"] = fit_member("q4_mcw50", HC_B_F, {**C3, "min_child_weight": 50.0}, 400)
M["q4fast"] = fit_member("q4_eta.1_200t", HC_B_F, {**C3, "eta": 0.1}, 200)
M["trio"] = fit_member("d6_400_cbn.3_HC+b15+b30+b60", HC_F + ["block15", "block30", "block60"], C3, 400)
M["quad"] = fit_member("d6_400_cbn.3_HC+b10+b15+b30", HC_F + ["block10", "block15", "block30"], C3, 400)
M["w1"] = fit_member("b1530_600t", B1530_F, C3, 600)
M["w2"] = fit_member("b1530_cbn.25", B1530_F, {**C3, "colsample_bynode": 0.25}, 400)
M["w3"] = fit_member("b1530_cbn.35", B1530_F, {**C3, "colsample_bynode": 0.35}, 400)
M["w4"] = fit_member("b1530_mcw50", B1530_F, {**C3, "min_child_weight": 50.0}, 400)
M["w5"] = fit_member("b1530_eta.03_800t", B1530_F, {**C3, "eta": 0.03}, 800)
M["w6"] = fit_member("HC+b15+b20", HC_F + ["block15", "block20"], C3, 400)
M["w7"] = fit_member("HC+b12+b15+b30", HC_F + ["block12", "block15", "block30"], C3, 400)
M["w8"] = fit_member("HC+b15+weekbin60", HC_F + ["block15", "weekbin60"], C3, 400)
M["w9"] = fit_member("HC+b15+carrierb15", HC_F + ["block15", "carrier_block15"], C3, 400)
M["w10"] = fit_member("HC+b15+monthb60", HC_F + ["block15", "month_block60"], C3, 400)
M["w11"] = fit_member("HC+b15+distbin", HC_F + ["block15", "dist_bin"], C3, 400)
M["d3"] = fit_member("d3_150t", BASE_F, {"max_depth": 3, "eta": 0.1}, 150)
M["d5"] = fit_member("d5_30t", BASE_F, {"max_depth": 5, "eta": 0.1}, 30)
M["cal"] = fit_member("d5_60t_calonly", CAL_F, {"max_depth": 5, "eta": 0.1}, 60)

LEG = ["d6c4big", "d8c4", "d8node", "d6big", "d3", "d5", "cal"]


def mean_of(keys):
    return np.mean([M[k][2][2] for k in keys], axis=0)


def auc_of(p):
    return roc_auc_score(y_eval, p)


L1 = ["d6c3"] + LEG
R = {}
BFAM = ["q4", "b15", "b20", "b60", "b3060", "b1530", "b10", "trio", "quad"]
R["N1"] = ("N1_T2_ref", auc_of(mean_of(BFAM)))
R["U1"] = ("U1_T2 b1530->600t", auc_of(mean_of(["q4", "b15", "b20", "b60", "b3060", "w1", "b10", "trio", "quad"])))
R["U2"] = ("U2_T2 b1530->cbn.25", auc_of(mean_of(["q4", "b15", "b20", "b60", "b3060", "w2", "b10", "trio", "quad"])))
R["U3"] = ("U3_T2 b1530->cbn.35", auc_of(mean_of(["q4", "b15", "b20", "b60", "b3060", "w3", "b10", "trio", "quad"])))
R["U4"] = ("U4_T2 b1530->mcw50", auc_of(mean_of(["q4", "b15", "b20", "b60", "b3060", "w4", "b10", "trio", "quad"])))
R["U5"] = ("U5_T2 b1530->eta.03", auc_of(mean_of(["q4", "b15", "b20", "b60", "b3060", "w5", "b10", "trio", "quad"])))
R["U6"] = ("U6_T2+w6+w7", auc_of(mean_of(BFAM + ["w6", "w7"])))
R["U7"] = ("U7_T2+w8(weekbin)", auc_of(mean_of(BFAM + ["w8"])))
R["U8"] = ("U8_T2+w9(carrierb15)", auc_of(mean_of(BFAM + ["w9"])))
R["U9"] = ("U9_T2+w10(monthb60)", auc_of(mean_of(BFAM + ["w10"])))
R["U10"] = ("U10_T2+w11(distbin)", auc_of(mean_of(BFAM + ["w11"])))
R["U11"] = ("U11_T2+all_new8", auc_of(mean_of(BFAM + ["w1", "w4", "w6", "w7", "w8", "w9", "w10", "w11"])))
R["U12"] = ("U12_best13_nodups", auc_of(mean_of(["q4", "b15", "b20", "b60", "b3060", "b1530", "b10", "trio", "quad", "w1", "w2", "w6", "w7"])))

for name, auc in R.values():
    print(f"ENS {name:36s} eval={auc:.4f}")
best_name = max(R, key=lambda k: R[k][1])
best_auc = R[best_name][1]
print(f"BEST: {best_name} {best_auc:.4f}")

COMBOS = {
    "N1": BFAM,
    "U1": ["q4", "b15", "b20", "b60", "b3060", "w1", "b10", "trio", "quad"],
    "U2": ["q4", "b15", "b20", "b60", "b3060", "w2", "b10", "trio", "quad"],
    "U3": ["q4", "b15", "b20", "b60", "b3060", "w3", "b10", "trio", "quad"],
    "U4": ["q4", "b15", "b20", "b60", "b3060", "w4", "b10", "trio", "quad"],
    "U5": ["q4", "b15", "b20", "b60", "b3060", "w5", "b10", "trio", "quad"],
    "U6": BFAM + ["w6", "w7"], "U7": BFAM + ["w8"], "U8": BFAM + ["w9"], "U9": BFAM + ["w10"],
    "U10": BFAM + ["w11"], "U11": BFAM + ["w1", "w4", "w6", "w7", "w8", "w9", "w10", "w11"],
    "U12": ["q4", "b15", "b20", "b60", "b3060", "b1530", "b10", "trio", "quad", "w1", "w2", "w6", "w7"],
}
ENSEMBLE_KEYS = COMBOS[best_name]
MODELS = [(M[k][2][0], M[k][2][1]) for k in ENSEMBLE_KEYS]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = prepare(df)
    ps = np.column_stack([b.predict(xgb.DMatrix(P[c], enable_categorical=True)) for b, c in MODELS])
    return ps.mean(axis=1)


eval_auc = best_auc
print(f"Eval AUC: {eval_auc:.4f}")
