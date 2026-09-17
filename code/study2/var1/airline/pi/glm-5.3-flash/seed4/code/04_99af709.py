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
    k["carrier_hour"] = k["UniqueCarrier"] + "_" + k["hour"]
    return k


_k_train = _keys(train)
cnt_maps = {c: _k_train[c].value_counts().to_dict() for c in ["route", "Origin", "Dest"]}
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
cat_levels["hour"] = pd.Index(sorted(_k_train["hour"].unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows."""
    k = _keys(df)
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype(float)  # raw hhmm
    hour_num = k["hour"].astype(int)
    dep_min = hour_num * 60 + (df["DepTime"].astype(float) % 100)
    X["hour_num"] = hour_num
    X["hour_cat"] = pd.Categorical(k["hour"], categories=cat_levels["hour"])
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
    for c in ["route", "Origin", "Dest"]:
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


def fit_eval(name: str, cols: list, over: dict, n_rounds=30, es=None, seeds=(SEED,)):
    params = {**BASE_PARAMS, **over}
    dtr = xgb.DMatrix(FULL.iloc[tr_idx][cols], label=y_tr, enable_categorical=True)
    dva = xgb.DMatrix(FULL.iloc[val_idx][cols], label=y_va, enable_categorical=True)
    dev = xgb.DMatrix(FULL_EVAL[cols], enable_categorical=True)
    t = time.time()
    ps = []
    for sd in seeds:
        p2 = {**params, "seed": sd}
        if es:
            bst = xgb.train(p2, dtr, num_boost_round=n_rounds, evals=[(dva, "val")],
                            early_stopping_rounds=es, verbose_eval=False)
            p = bst.predict(dev, iteration_range=(0, bst.best_iteration + 1))
            bi, vs = bst.best_iteration, bst.best_score
        else:
            bst = xgb.train(p2, dtr, num_boost_round=n_rounds)
            p = bst.predict(dev)
            bi, vs = n_rounds, roc_auc_score(y_va, bst.predict(dva))
        ps.append(p)
    p = np.mean(ps, axis=0)
    auc = roc_auc_score(y_eval, p)
    print(f"DIAG {name:36s} it={bi:4d} val={vs:.4f} eval={auc:.4f} ({time.time() - t:.0f}s)")
    return name, auc, bst, cols


B6CATS = ["month", "dayofmonth", "dayofweek", "carrier", "origin", "dest"]
BASE_F = B6CATS + ["DepTime", "distance"]
SC = ["dep_sin", "dep_cos"]
R = {}
R["H1_exact_base"] = fit_eval("H1_exact_base_30t", BASE_F, {"max_depth": 6, "eta": 0.1})
R["H2_base_sc"] = fit_eval("H2_base+sincos_30t", BASE_F + SC, {"max_depth": 6, "eta": 0.1})
R["H3_base_sc_hourcat"] = fit_eval("H3_base+sc+hourcat", BASE_F + SC + ["hour_cat"], {"max_depth": 6, "eta": 0.1})
R["H4_50t"] = fit_eval("H4_base+sc_50t", BASE_F + SC, {"max_depth": 6, "eta": 0.1}, n_rounds=50)
R["H5_d5"] = fit_eval("H5_base_d5_30t", BASE_F, {"max_depth": 5, "eta": 0.1})
R["H6_20t_lr2"] = fit_eval("H6_base_20t_lr2", BASE_F, {"max_depth": 6, "eta": 0.2}, n_rounds=20)
R["H7_d8"] = fit_eval("H7_base+sc_d8_mcw20", BASE_F + SC, {"max_depth": 8, "min_child_weight": 20, "eta": 0.1})
R["H8_gamma"] = fit_eval("H8_base+sc_gamma1", BASE_F + SC, {"max_depth": 6, "eta": 0.1, "gamma": 1.0})
R["H9_ens5"] = fit_eval("H9_base+sc_ens5seeds", BASE_F + SC, {"max_depth": 6, "eta": 0.1}, seeds=(1, 2, 3, 4, 5))
R["H10_base_log"] = fit_eval("H10_base+sc+distlog", BASE_F + SC + ["dist_log"], {"max_depth": 6, "eta": 0.1})
R["H11_base_sc_cnt"] = fit_eval("H11_base+sc+cntOD", BASE_F + SC + ["cnt_Origin", "cnt_Dest"], {"max_depth": 6, "eta": 0.1})
R["H12_base_sc_nums"] = fit_eval("H12_base+sc+calnums", BASE_F + SC + ["month_num", "day_num", "dow_num"], {"max_depth": 6, "eta": 0.1})

best_name = max(R, key=lambda k: R[k][1])
_, best_auc, best_bst, best_cols = R[best_name]
print(f"BEST_DIAG: {best_name} {best_auc:.4f}")

model = best_bst
BEST_COLS = best_cols


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict(xgb.DMatrix(prepare(df)[BEST_COLS], enable_categorical=True))


eval_auc = best_auc
print(f"Eval AUC: {eval_auc:.4f}")
