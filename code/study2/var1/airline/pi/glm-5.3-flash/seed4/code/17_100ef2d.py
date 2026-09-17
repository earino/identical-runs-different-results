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
    k["carrier_dow"] = k["UniqueCarrier"] + "_" + k["DayOfWeek"]
    k["route"] = k["Origin"] + "_" + k["Dest"]
    return k


_k_train = _keys(train)
cnt_maps = {c: _k_train[c].value_counts().to_dict() for c in ["Origin", "Dest", "route"]}
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
for c in ["hour", "block30", "dow_hour", "carrier_dow"]:
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
M["q1"] = fit_member("d6_400_cbn.3_HC", HC_F, C3, 400)
M["q2"] = fit_member("d6_400_cbn.25_HC", HC_F, {**C3, "colsample_bynode": 0.25}, 400)
M["q3"] = fit_member("d6_600_cbn.3_HC", HC_F, C3, 600)
M["q4"] = fit_member("d6_400_cbn.3_HC+block30", HC_B_F, C3, 400)
M["q5"] = fit_member("d6_800_cbn.3_HC_BIGF", BIG_HC_F, C3, 800)
M["q6"] = fit_member("d8_600_cbn.3_HC", HC_F, {"max_depth": 8, "eta": 0.05, "min_child_weight": 50.0, "colsample_bynode": 0.3}, 600)
M["q7"] = fit_member("d4_150_cbn.3_HC", HC_F, {"max_depth": 4, "eta": 0.1, "colsample_bynode": 0.3}, 150)
M["q8"] = fit_member("d3_300_cbn.3_HC", HC_F, {"max_depth": 3, "eta": 0.1, "colsample_bynode": 0.3}, 300)
M["d6c3"] = fit_member("d6_400_cbn.3_noHC_ref", EXT_F, C3, 400)
M["d3"] = fit_member("d3_150t", BASE_F, {"max_depth": 3, "eta": 0.1}, 150)
M["d5"] = fit_member("d5_30t", BASE_F, {"max_depth": 5, "eta": 0.1}, 30)
M["d8node"] = fit_member("d8_600t_cbn.3", EXT_F, {"max_depth": 8, "eta": 0.05, "min_child_weight": 50.0, "colsample_bynode": 0.3}, 600)
M["d6big"] = fit_member("d6_800t_BIGF", BIG_F, {"max_depth": 6, "eta": 0.05, "min_child_weight": 20.0}, 800)
M["d8c4"] = fit_member("d8_600t_cbn.4", EXT_F, {"max_depth": 8, "eta": 0.05, "min_child_weight": 50.0, "colsample_bynode": 0.4}, 600)
M["d6c4big"] = fit_member("d6_800t_cbn.4_BIGF", BIG_F, {"max_depth": 6, "eta": 0.05, "min_child_weight": 20.0, "colsample_bynode": 0.4}, 800)
M["cal"] = fit_member("d5_60t_calonly", CAL_F, {"max_depth": 5, "eta": 0.1}, 60)

LEG = ["d6c4big", "d8c4", "d8node", "d6big", "d3", "d5", "cal"]


def mean_of(keys):
    return np.mean([M[k][2][2] for k in keys], axis=0)


def auc_of(p):
    return roc_auc_score(y_eval, p)


L1 = ["d6c3"] + LEG
R = {}
R["N1"] = ("N1_L1_ref(1)", auc_of(mean_of(L1)))
R["Q1"] = ("Q1_L1 champ->HC", auc_of(mean_of(["q1"] + LEG)))
R["Q2"] = ("Q2_L1 champ->cbn.25HC", auc_of(mean_of(["q2"] + LEG)))
R["Q3"] = ("Q3_L1+q1+q2+q3", auc_of(mean_of(L1 + ["q1", "q2", "q3"])))
R["Q4"] = ("Q4_qfam(q1..q4)+LEG", auc_of(mean_of(["q1", "q2", "q3", "q4"] + LEG)))
R["Q5"] = ("Q5_qfam(q1..q6)+LEG", auc_of(mean_of(["q1", "q2", "q3", "q4", "q5", "q6"] + LEG)))
R["Q6"] = ("Q6_q1..q8+LEG_all", auc_of(mean_of(["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"] + LEG)))
R["Q7"] = ("Q7_q1..q8+d6c3+LEG", auc_of(mean_of(["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "d6c3"] + LEG)))
R["Q8"] = ("Q8_qfam+d6c3+LEG_smallMCW", auc_of(mean_of(["q1", "q2", "q3", "q4", "q5", "d6c3", "d3", "d5", "cal"])))

for name, auc in R.values():
    print(f"ENS {name:36s} eval={auc:.4f}")
best_name = max(R, key=lambda k: R[k][1])
best_auc = R[best_name][1]
print(f"BEST: {best_name} {best_auc:.4f}")

COMBOS = {
    "N1": L1, "Q1": ["q1"] + LEG, "Q2": ["q2"] + LEG, "Q3": L1 + ["q1", "q2", "q3"],
    "Q4": ["q1", "q2", "q3", "q4"] + LEG, "Q5": ["q1", "q2", "q3", "q4", "q5", "q6"] + LEG,
    "Q6": ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"] + LEG,
    "Q7": ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "d6c3"] + LEG,
    "Q8": ["q1", "q2", "q3", "q4", "q5", "d6c3", "d3", "d5", "cal"],
}
ENSEMBLE_KEYS = COMBOS[best_name]
MODELS = [(M[k][2][0], M[k][2][1]) for k in ENSEMBLE_KEYS]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = prepare(df)
    ps = np.column_stack([b.predict(xgb.DMatrix(P[c], enable_categorical=True)) for b, c in MODELS])
    return ps.mean(axis=1)


eval_auc = best_auc
print(f"Eval AUC: {eval_auc:.4f}")
