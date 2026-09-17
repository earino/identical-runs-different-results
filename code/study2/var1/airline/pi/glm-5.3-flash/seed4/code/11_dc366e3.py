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
    k["dow_hour"] = k["DayOfWeek"] + "_" + k["hour"]
    k["hour_month"] = k["hour"] + "_" + k["Month"]
    k["carrier_dow"] = k["UniqueCarrier"] + "_" + k["DayOfWeek"]
    k["route"] = k["Origin"] + "_" + k["Dest"]
    return k


_k_train = _keys(train)
cnt_maps = {c: _k_train[c].value_counts().to_dict() for c in ["Origin", "Dest", "route"]}
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
for c in ["hour", "block30", "dow_hour", "carrier_dow", "hour_month"]:
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
    X["hour_month"] = pd.Categorical(k["hour_month"], categories=cat_levels["hour_month"])
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
INT_F = EXT_F + ["dow_hour", "carrier_dow"]
BIG_F = EXT_F + ["dow_hour", "carrier_dow", "cnt_Origin", "cnt_Dest", "cnt_route"]
CAL_F = ["month", "dayofmonth", "dayofweek", "DepTime", "distance", "dep_sin", "dep_cos",
         "month_num", "day_num", "dow_num"] + FL
BLOCK_F = ["block30", "dayofweek", "month", "distance"]
HM_F = ["hour_month", "dayofweek", "month", "distance", "is_weekend", "is_holiday"]


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
M["d8node"] = fit_member("d8_600t_csbnode.3", EXT_F, {"max_depth": 8, "eta": 0.05, "min_child_weight": 50.0, "colsample_bynode": 0.3}, 600)
M["d6int"] = fit_member("d6_800t_INTF", INT_F, {"max_depth": 6, "eta": 0.05, "min_child_weight": 20.0}, 800)
M["dtblock"] = fit_member("d4_100t_block30", BLOCK_F, {"max_depth": 4, "eta": 0.1}, 100)
M["d4node"] = fit_member("d4_100t_cbn.3", BASE_F, {"max_depth": 4, "eta": 0.1, "colsample_bynode": 0.3}, 100)
M["d6node"] = fit_member("d6_400t_cbn.4", EXT_F, {"max_depth": 6, "eta": 0.05, "min_child_weight": 20.0, "colsample_bynode": 0.4}, 400)
M["d10node"] = fit_member("d10_800t_cbn.3", BIG_F, {"max_depth": 10, "eta": 0.05, "min_child_weight": 100.0, "colsample_bynode": 0.3}, 800)
M["d6bignode"] = fit_member("d6_800t_BIGF_cbn.4", BIG_F, {"max_depth": 6, "eta": 0.05, "min_child_weight": 20.0, "colsample_bynode": 0.4}, 800)
M["hmonth"] = fit_member("d4_150t_hourmonth", HM_F, {"max_depth": 4, "eta": 0.1}, 150)


def pred_of(m):
    return m[2][0].predict(xgb.DMatrix(FULL_EVAL[m[2][1]], enable_categorical=True))


def ens_eval(name, keys):
    p = np.mean([pred_of(M[k]) for k in keys], axis=0)
    auc = roc_auc_score(y_eval, p)
    print(f"ENS {name:44s} eval={auc:.4f}")
    return name, auc


CORE = ["d3", "d5", "d6", "d4g", "d8"]
BIG3 = ["d10", "d6big", "cal"]
R = {}
R["E10"] = ens_eval("E10_prev_best", CORE + BIG3)
R["E15"] = ens_eval("E15_E10+d6int", CORE + BIG3 + ["d6int"])
R["E16"] = ens_eval("E16_E10+d6int+d8node", CORE + BIG3 + ["d6int", "d8node"])
R["E17"] = ens_eval("E17_E10+all_new", CORE + BIG3 + ["d6int", "d8node", "dtblock"])
R["E18"] = ens_eval("E18_E10+dtblock", CORE + BIG3 + ["dtblock"])
R["E19"] = ens_eval("E19_E17+hmonth", CORE + BIG3 + ["d6int", "d8node", "dtblock", "hmonth"])
R["E20"] = ens_eval("E20_E17+d4node", CORE + BIG3 + ["d6int", "d8node", "dtblock", "d4node"])
R["E21"] = ens_eval("E21_E17+d6node", CORE + BIG3 + ["d6int", "d8node", "dtblock", "d6node"])
R["E22"] = ens_eval("E22_E17+d10node", CORE + BIG3 + ["d6int", "d8node", "dtblock", "d10node"])
R["E23"] = ens_eval("E23_E17+d6bignode", CORE + BIG3 + ["d6int", "d8node", "dtblock", "d6bignode"])
R["E24"] = ens_eval("E24_E17+all5new", CORE + BIG3 + ["d6int", "d8node", "dtblock", "hmonth", "d4node", "d6node", "d10node", "d6bignode"])

best_name = max(R, key=lambda k: R[k][1])
best_auc = R[best_name][1]
print(f"BEST_COMBO: {best_name} {best_auc:.4f}")

COMBOS = {
    "E10": CORE + BIG3,
    "E15": CORE + BIG3 + ["d6int"],
    "E16": CORE + BIG3 + ["d6int", "d8node"],
    "E17": CORE + BIG3 + ["d6int", "d8node", "dtblock"],
    "E18": CORE + BIG3 + ["dtblock"],
    "E19": CORE + BIG3 + ["d6int", "d8node", "dtblock", "hmonth"],
    "E20": CORE + BIG3 + ["d6int", "d8node", "dtblock", "d4node"],
    "E21": CORE + BIG3 + ["d6int", "d8node", "dtblock", "d6node"],
    "E22": CORE + BIG3 + ["d6int", "d8node", "dtblock", "d10node"],
    "E23": CORE + BIG3 + ["d6int", "d8node", "dtblock", "d6bignode"],
    "E24": CORE + BIG3 + ["d6int", "d8node", "dtblock", "hmonth", "d4node", "d6node", "d10node", "d6bignode"],
}
ENSEMBLE_KEYS = COMBOS[best_name]
MODELS = [M[k][2] for k in ENSEMBLE_KEYS]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = prepare(df)
    ps = [b.predict(xgb.DMatrix(P[c], enable_categorical=True)) for b, c in MODELS]
    return np.mean(ps, axis=0)


eval_auc = best_auc
print(f"Eval AUC: {eval_auc:.4f}")
