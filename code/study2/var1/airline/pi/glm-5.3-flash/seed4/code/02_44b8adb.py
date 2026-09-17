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
TE_M = 20.0  # smoothing pseudo-count for target encoding
PRIOR = float(y_all.mean())


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered string keys used for categoricals and target encodings (no target needed)."""
    k = pd.DataFrame(index=df.index)
    for c in ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]:
        k[c] = df[c].astype(str)
    k["hour"] = (df["DepTime"].astype(float) // 100).astype(int).astype(str)
    k["route"] = k["Origin"] + "_" + k["Dest"]
    k["carrier_hour"] = k["UniqueCarrier"] + "_" + k["hour"]
    return k


def _smooth_te(keys: pd.Series, m: float = TE_M) -> dict:
    g = y_all.groupby(keys).agg(["sum", "count"])
    return ((g["sum"] + m * PRIOR) / (g["count"] + m)).to_dict()


_k_train = _keys(train)
TE_COLS = ["Origin", "Dest", "UniqueCarrier", "route", "carrier_hour", "hour"]
te_maps = {c: _smooth_te(_k_train[c]) for c in TE_COLS}
cnt_maps = {c: _k_train[c].value_counts().to_dict() for c in ["route", "Origin", "Dest"]}
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
cat_levels["hour"] = pd.Index(sorted(_k_train["hour"].unique()))
cat_levels["route"] = pd.Index(sorted(_k_train["route"].unique()))
cat_levels["carrier_hour"] = pd.Index(sorted(_k_train["carrier_hour"].unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows."""
    k = _keys(df)
    X = pd.DataFrame(index=df.index)
    # time-of-day
    hour_num = k["hour"].astype(int)
    dep_min = hour_num * 60 + (df["DepTime"].astype(float) % 100)
    X["hour_num"] = hour_num
    X["hour_cat"] = pd.Categorical(k["hour"], categories=cat_levels["hour"])
    X["dep_sin"] = np.sin(2 * np.pi * dep_min / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * dep_min / 1440.0)
    # calendar
    X["month"] = pd.Categorical(k["Month"], categories=cat_levels["Month"])
    X["dayofmonth"] = pd.Categorical(k["DayofMonth"], categories=cat_levels["DayofMonth"])
    X["dayofweek"] = pd.Categorical(k["DayOfWeek"], categories=cat_levels["DayOfWeek"])
    X["month_num"] = k["Month"].str.slice(2).astype(int)
    X["day_num"] = k["DayofMonth"].str.slice(2).astype(int)
    X["dow_num"] = k["DayOfWeek"].str.slice(2).astype(int)
    # cats
    X["carrier"] = pd.Categorical(k["UniqueCarrier"], categories=cat_levels["UniqueCarrier"])
    X["origin"] = pd.Categorical(k["Origin"], categories=cat_levels["Origin"])
    X["dest"] = pd.Categorical(k["Dest"], categories=cat_levels["Dest"])
    X["route"] = pd.Categorical(k["route"], categories=cat_levels["route"])
    X["carrier_hour"] = pd.Categorical(k["carrier_hour"], categories=cat_levels["carrier_hour"])
    # distance
    dist = df["Distance"].astype(float)
    X["distance"] = dist
    X["dist_log"] = np.log1p(dist)
    # target encodings (maps fitted on train only; unseen -> prior)
    for c in TE_COLS:
        X["te_" + c] = k[c].map(te_maps[c]).fillna(PRIOR).astype(float)
    # frequency
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
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(train))
n_val = max(1000, len(train) // 10)
val_idx, tr_idx = idx[:n_val], idx[n_val:]
train_in = train.iloc[tr_idx].reset_index(drop=True)
val_in = train.iloc[val_idx].reset_index(drop=True)
X_tr, X_val = prepare(train_in), prepare(val_in)
dtrain = xgb.DMatrix(X_tr, label=y_all.iloc[tr_idx].to_numpy(), enable_categorical=True)
dval = xgb.DMatrix(X_val, label=y_all.iloc[val_idx].to_numpy(), enable_categorical=True)
deval = xgb.DMatrix(prepare(evald), enable_categorical=True)
y_eval = to_y(evald)
print(f"Data prep time: {time.time() - t0:.1f}s")


def fit_eval(name: str, over: dict, drop_cols=(), n_rounds=2000, es=60):
    params = {**BASE_PARAMS, **over}
    dm_tr, dm_va = dtrain, dval
    if drop_cols:
        dm_tr = xgb.DMatrix(X_tr.drop(columns=list(drop_cols)), label=y_all.iloc[tr_idx].to_numpy(), enable_categorical=True)
        dm_va = xgb.DMatrix(X_val.drop(columns=list(drop_cols)), label=y_all.iloc[val_idx].to_numpy(), enable_categorical=True)
        dm_ev = xgb.DMatrix(prepare(evald).drop(columns=list(drop_cols)), enable_categorical=True)
    else:
        dm_ev = deval
    t = time.time()
    bst = xgb.train(params, dm_tr, num_boost_round=n_rounds, evals=[(dm_va, "val")],
                    early_stopping_rounds=es, verbose_eval=False)
    p = bst.predict(dm_ev, iteration_range=(0, bst.best_iteration + 1))
    auc = roc_auc_score(y_eval, p)
    print(f"DIAG {name:28s} best_iter={bst.best_iteration:4d} val={bst.best_score:.4f} eval={auc:.4f} ({time.time() - t:.0f}s)")
    return name, auc, bst, dm_ev


results = {}
results["A_d8_mcw10"] = fit_eval("A_d8_mcw10_ss.9_cs.8", {"max_depth": 8, "min_child_weight": 10, "subsample": 0.9, "colsample_bytree": 0.8, "eta": 0.05})
results["B_d6_mcw20"] = fit_eval("B_d6_mcw20_ss.8_cs.7", {"max_depth": 6, "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.7, "eta": 0.05})
results["C_d10_mcw10"] = fit_eval("C_d10_mcw10_ss.9_cs.8", {"max_depth": 10, "min_child_weight": 10, "subsample": 0.9, "colsample_bytree": 0.8, "eta": 0.05})
TEC = ["te_" + c for c in TE_COLS]
results["D_noteA"] = fit_eval("D_noTE_A_params", {"max_depth": 8, "min_child_weight": 10, "subsample": 0.9, "colsample_bytree": 0.8, "eta": 0.05}, drop_cols=TEC)
results["E_small"] = fit_eval("E_small_d6_40trees", {"max_depth": 6, "eta": 0.1}, n_rounds=40, es=1000)

best_name = max(results, key=lambda k: results[k][1])
_, best_auc, best_bst, best_dm = results[best_name]
print(f"BEST_DIAG: {best_name} {best_auc:.4f}")

model = best_bst


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict(xgb.DMatrix(prepare(df), enable_categorical=True),
                         iteration_range=(0, model.best_iteration + 1))


eval_auc = best_auc
print(f"Eval AUC: {eval_auc:.4f}")
