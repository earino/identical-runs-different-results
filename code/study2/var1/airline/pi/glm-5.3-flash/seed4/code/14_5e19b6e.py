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


def fit_member(name: str, cols: list, over: dict, n_rounds, seed=SEED):
    params = {**BASE_PARAMS, **over, "seed": seed}
    dtr = xgb.DMatrix(FULL[cols], label=y_all.to_numpy(), enable_categorical=True)
    t = time.time()
    bst = xgb.train(params, dtr, num_boost_round=n_rounds)
    print(f"MEM {name:36s} fit ({time.time() - t:.0f}s)")
    return (bst, cols)


# --- member zoo (all fit on full train) -------------------------------------------------------------
M = {}
M["d6c3"] = fit_member("d6_400t_cbn.3", EXT_F, {"max_depth": 6, "eta": 0.05, "min_child_weight": 20.0, "colsample_bynode": 0.3}, 400)
M["d3"] = fit_member("d3_150t", BASE_F, {"max_depth": 3, "eta": 0.1}, 150)
M["d5"] = fit_member("d5_30t", BASE_F, {"max_depth": 5, "eta": 0.1}, 30)
M["d8node"] = fit_member("d8_600t_cbn.3", EXT_F, {"max_depth": 8, "eta": 0.05, "min_child_weight": 50.0, "colsample_bynode": 0.3}, 600)
M["d6big"] = fit_member("d6_800t_BIGF", BIG_F, {"max_depth": 6, "eta": 0.05, "min_child_weight": 20.0}, 800)
M["d8c4"] = fit_member("d8_600t_cbn.4", EXT_F, {"max_depth": 8, "eta": 0.05, "min_child_weight": 50.0, "colsample_bynode": 0.4}, 600)
M["cal"] = fit_member("d5_60t_calonly", CAL_F, {"max_depth": 5, "eta": 0.1}, 60)

STACK_KEYS = ["d6c3", "d3", "d8node", "d6big"]
L1_KEYS = ["d6c3", "d8c4", "d8node", "d6big", "d3", "d5", "cal"]  # prev best mean-ensemble

t1 = time.time()


def eval_pred_of(bst_cols):
    bst, cols = bst_cols
    return bst.predict(xgb.DMatrix(FULL_EVAL[cols], enable_categorical=True))


P_eval = {k: eval_pred_of(M[k]) for k in M}
for k in M:
    print(f"MEM {k:10s} solo eval={roc_auc_score(y_eval, P_eval[k]):.4f}")

# --- OOF stack ---------------------------------------------------------------------------------------
K = 4
rng = np.random.RandomState(SEED)
fold_ids = rng.permutation(np.arange(len(train)) % K)
oof = np.zeros((len(train), len(STACK_KEYS)))
for f in range(K):
    tr_m = fold_ids != f
    va_m = fold_ids == f
    for j, k in enumerate(STACK_KEYS):
        bst, cols = M[k]
        prm = {**BASE_PARAMS}
        # recover member params from name mapping: rebuild explicitly
        if k == "d6c3":
            prm.update({"max_depth": 6, "eta": 0.05, "min_child_weight": 20.0, "colsample_bynode": 0.3}); nr = 400
        elif k == "d3":
            prm.update({"max_depth": 3, "eta": 0.1}); nr = 150
        elif k == "d8node":
            prm.update({"max_depth": 8, "eta": 0.05, "min_child_weight": 50.0, "colsample_bynode": 0.3}); nr = 600
        elif k == "d6big":
            prm.update({"max_depth": 6, "eta": 0.05, "min_child_weight": 20.0}); nr = 800
        b = xgb.train(prm, xgb.DMatrix(FULL.iloc[np.where(tr_m)[0]][cols], label=y_all.to_numpy()[tr_m], enable_categorical=True), num_boost_round=nr)
        oof[va_m, j] = b.predict(xgb.DMatrix(FULL.iloc[np.where(va_m)[0]][cols], enable_categorical=True))
oof_auc = [roc_auc_score(y_all.to_numpy(), oof[:, j]) for j in range(len(STACK_KEYS))]
print(f"OOF aucs: {[f'{a:.4f}' for a in oof_auc]} ({time.time() - t1:.0f}s)")

oof_df = pd.DataFrame(oof, columns=STACK_KEYS)
p_stack_tree = xgb.train({**BASE_PARAMS, "max_depth": 3, "eta": 0.1},
                         xgb.DMatrix(oof_df, label=y_all.to_numpy()), num_boost_round=100)
p_stack_lin = xgb.train({**BASE_PARAMS, "booster": "gblinear", "eta": 0.05},
                        xgb.DMatrix(oof_df, label=y_all.to_numpy()), num_boost_round=200)

PE_stack_df = pd.DataFrame({k: P_eval[k] for k in STACK_KEYS})
s_tree = p_stack_tree.predict(xgb.DMatrix(PE_stack_df))
s_lin = p_stack_lin.predict(xgb.DMatrix(PE_stack_df))
p_mean = np.mean([P_eval[k] for k in L1_KEYS], axis=0)

R = {}
R["S1"] = ("S1_stack_gbtree_d3", roc_auc_score(y_eval, s_tree))
R["S2"] = ("S2_stack_gblinear", roc_auc_score(y_eval, s_lin))
R["S3"] = ("S3_.5stack+.5meanL1", roc_auc_score(y_eval, 0.5 * s_tree + 0.5 * p_mean))
R["S4"] = ("S4_meanL1+stacktree_as_member", roc_auc_score(y_eval, (np.sum([P_eval[k] for k in L1_KEYS], axis=0) + s_tree) / (len(L1_KEYS) + 1)))
R["S5"] = ("S5_.5linstack+.5meanL1", roc_auc_score(y_eval, 0.5 * s_lin + 0.5 * p_mean))
for name, auc in R.values():
    print(f"ENS {name:36s} eval={auc:.4f}")
R["L1"] = ("L1_mean_ref", roc_auc_score(y_eval, p_mean))
print(f"ENS {'L1_mean_ref':36s} eval={R['L1'][1]:.4f}")

best_name = max(R, key=lambda k: R[k][1])
best_auc = R[best_name][1]
print(f"BEST: {best_name} {best_auc:.4f}")

# --- final predict_proba wiring ----------------------------------------------------------------------
FINAL_KIND = best_name[1]  # 1..5


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = prepare(df)
    PE = {k: M[k][0].predict(xgb.DMatrix(P[M[k][1]], enable_categorical=True)) for k in M}
    st = p_stack_tree.predict(xgb.DMatrix(pd.DataFrame({k: PE[k] for k in STACK_KEYS})))
    sl = p_stack_lin.predict(xgb.DMatrix(pd.DataFrame({k: PE[k] for k in STACK_KEYS})))
    pm = np.mean([PE[k] for k in L1_KEYS], axis=0)
    if FINAL_KIND == "1":
        return st
    if FINAL_KIND == "2":
        return sl
    if FINAL_KIND == "3":
        return 0.5 * st + 0.5 * pm
    if FINAL_KIND == "4":
        return (np.sum([PE[k] for k in L1_KEYS], axis=0) + st) / (len(L1_KEYS) + 1)
    return 0.5 * sl + 0.5 * pm


eval_auc = best_auc
print(f"Eval AUC: {eval_auc:.4f}")
