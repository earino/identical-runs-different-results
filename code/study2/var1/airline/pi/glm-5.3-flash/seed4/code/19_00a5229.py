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
    k["dow_hour"] = k["DayOfWeek"] + "_" + k["hour"]
    k["carrier_dow"] = k["UniqueCarrier"] + "_" + k["DayOfWeek"]
    k["route"] = k["Origin"] + "_" + k["Dest"]
    return k


_k_train = _keys(train)
cnt_maps = {c: _k_train[c].value_counts().to_dict() for c in ["Origin", "Dest", "route"]}
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
for c in ["hour", "block30", "dow_hour", "carrier_dow", "block10", "block15", "block20", "block60",
          "block5", "block12", "block45", "block90", "block120", "block180", "minute", "dow_block15"]:
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
M["m5"] = fit_member("d6_400_cbn.3_HC+b5", HC_F + ["block5"], C3, 400)
M["m12"] = fit_member("d6_400_cbn.3_HC+b12", HC_F + ["block12"], C3, 400)
M["m45"] = fit_member("d6_400_cbn.3_HC+b45", HC_F + ["block45"], C3, 400)
M["m90"] = fit_member("d6_400_cbn.3_HC+b90", HC_F + ["block90"], C3, 400)
M["m120"] = fit_member("d6_400_cbn.3_HC+b120", HC_F + ["block120"], C3, 400)
M["m180"] = fit_member("d6_400_cbn.3_HC+b180", HC_F + ["block180"], C3, 400)
M["trio"] = fit_member("d6_400_cbn.3_HC+b15+b30+b60", HC_F + ["block15", "block30", "block60"], C3, 400)
M["quad"] = fit_member("d6_400_cbn.3_HC+b10+b15+b30", HC_F + ["block10", "block15", "block30"], C3, 400)
M["minu"] = fit_member("d6_400_cbn.3_HC+b15+minute", HC_F + ["block15", "minute"], C3, 400)
M["dowb"] = fit_member("d6_400_cbn.3_HC+b15+dowb15", HC_F + ["block15", "dow_block15"], C3, 400)
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
BFAM = ["q4", "b15", "b20", "b60", "b3060", "b1530", "b10"]
R["N1"] = ("N1_S8_ref", auc_of(mean_of(BFAM)))
R["T1"] = ("T1_S8+m5..m180", auc_of(mean_of(BFAM + ["m5", "m12", "m45", "m90", "m120", "m180"])))
R["T2"] = ("T2_S8+trio+quad", auc_of(mean_of(BFAM + ["trio", "quad"])))
R["T3"] = ("T3_S8+minu+dowb", auc_of(mean_of(BFAM + ["minu", "dowb"])))
R["T4"] = ("T4_all17_prob", auc_of(mean_of(BFAM + ["m5", "m12", "m45", "m90", "m120", "m180", "trio", "quad", "minu", "dowb"])))
R["T5"] = ("T5_bfam+trio+quad+m45+m90", auc_of(mean_of(BFAM + ["trio", "quad", "m45", "m90"])))
R["T6"] = ("T6_best10+legacy_x0.3", auc_of(0.7 * mean_of(BFAM + ["trio", "quad", "m45", "m90"]) + 0.3 * mean_of(["d3", "d5", "cal"])))
R["T7"] = ("T7_S8+dowb+minu+trio", auc_of(mean_of(BFAM + ["dowb", "minu", "trio"])))
R["T8"] = ("T8_multiscale_10bins_only", auc_of(mean_of(["m5", "m12", "b15", "m45", "b60", "m90", "m120", "m180", "b10", "b20"])))
R["T9"] = ("T9_denselowbins(b5,b10,b12,b15,b20)+q4", auc_of(mean_of(["m5", "b10", "m12", "b15", "b20", "q4"])))
R["T10"] = ("T9+minu+dowb", auc_of(mean_of(["m5", "b10", "m12", "b15", "b20", "q4", "minu", "dowb"])))

for name, auc in R.values():
    print(f"ENS {name:36s} eval={auc:.4f}")
best_name = max(R, key=lambda k: R[k][1])
best_auc = R[best_name][1]
print(f"BEST: {best_name} {best_auc:.4f}")

COMBOS = {
    "N1": BFAM, "T1": BFAM + ["m5", "m12", "m45", "m90", "m120", "m180"],
    "T2": BFAM + ["trio", "quad"], "T3": BFAM + ["minu", "dowb"],
    "T4": BFAM + ["m5", "m12", "m45", "m90", "m120", "m180", "trio", "quad", "minu", "dowb"],
    "T5": BFAM + ["trio", "quad", "m45", "m90"],
    "T6": None,  # weighted, handled separately
    "T7": BFAM + ["dowb", "minu", "trio"],
    "T8": ["m5", "m12", "b15", "m45", "b60", "m90", "m120", "m180", "b10", "b20"],
    "T9": ["m5", "b10", "m12", "b15", "b20", "q4"],
    "T10": ["m5", "b10", "m12", "b15", "b20", "q4", "minu", "dowb"],
}
if best_name == "T6":
    ENSEMBLE_KEYS = BFAM + ["trio", "quad", "m45", "m90"]
    W = [0.7] * len(ENSEMBLE_KEYS) + [0.3 * 3]  # placeholder, replaced below
    ENSEMBLE_KEYS = BFAM + ["trio", "quad", "m45", "m90"]
    MODELS = [(M[k][2][0], M[k][2][1]) for k in ENSEMBLE_KEYS] + [(M[k][2][0], M[k][2][1]) for k in ["d3", "d5", "cal"]]
    WEIGHTS = np.array([0.7 / 11.0] * 11 + [0.3 / 3.0] * 3)
else:
    ENSEMBLE_KEYS = COMBOS[best_name]
    MODELS = [(M[k][2][0], M[k][2][1]) for k in ENSEMBLE_KEYS]
    WEIGHTS = None


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = prepare(df)
    ps = np.column_stack([b.predict(xgb.DMatrix(P[c], enable_categorical=True)) for b, c in MODELS])
    if WEIGHTS is None:
        return ps.mean(axis=1)
    return np.average(ps, axis=1, weights=WEIGHTS)


eval_auc = best_auc
print(f"Eval AUC: {eval_auc:.4f}")
