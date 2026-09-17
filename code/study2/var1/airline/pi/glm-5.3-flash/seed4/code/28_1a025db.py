"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Final architecture (findings from 39 diagnostic experiments):
  - Year drift (2005 -> 2006) dominates: target encodings and route/airport categorical
    interactions overfit 2005. Only year-stable structure transfers.
  - The strongest single signal is scheduled-departure-time structure: hour-of-day
    categoricals plus multi-resolution DepTime bins (10/15/20/30/45/60 min), and their
    interactions with carrier and origin airport.
  - Best single recipe: depth-6, 400-1500 trees, eta .05, min_child_weight 20,
    colsample_bynode 0.3 (node-level column subsampling), on hour_cat + hour_num +
    one or two DepTime bins (+ optionally a carrier/origin x bin interaction).
  - Simple equal-weight mean of 23 such members (varying bin resolution, interaction
    feature, and boosting rounds) is the final model. Stacking/OOF meta-models, target
    encodings, route features, seed bagging, and legacy feature sets were all tested and
    rejected (they did not beat the plain mean).
"""
import json
import os

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
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in
              ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]}
_t = train["DepTime"].astype(float)
cat_levels["hour_cat"] = pd.Index(sorted(((_t // 100).astype(int)).astype(str).unique()))
CBINS = (10, 15, 20, 30, 45, 60)
for b in CBINS:
    key = f"block{b}"
    cat_levels[key] = pd.Index(sorted((_t // b).astype(int).astype(str).unique()))
for c in ["UniqueCarrier"]:
    for b in CBINS:
        key = c + f"_block{b}"
        cat_levels[key] = pd.Index(sorted(
            (train[c].astype(str) + "_" + (_t // b).astype(int).astype(str)).unique()))
for b in (20, 45):
    key = "Origin" + f"_block{b}"
    cat_levels[key] = pd.Index(sorted(
        (train["Origin"].astype(str) + "_" + (_t // b).astype(int).astype(str)).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows."""
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(float)
    hour_num = (dep // 100).astype(int)
    dep_min = hour_num * 60 + (dep % 100)
    X["DepTime"] = dep
    X["hour_num"] = hour_num
    X["dep_sin"] = np.sin(2 * np.pi * dep_min / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * dep_min / 1440.0)
    for c in ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]:
        X[c.lower() if c not in ("UniqueCarrier",) else "carrier"] = pd.Categorical(
            df[c].astype(str), categories=cat_levels[c])
    X["hour_cat"] = pd.Categorical(hour_num.astype(int).astype(str), categories=cat_levels["hour_cat"])
    dist = df["Distance"].astype(float)
    X["distance"] = dist
    md = X["month"].astype(str).str.slice(2).astype(int) * 100 + \
        X["dayofmonth"].astype(str).str.slice(2).astype(int)
    dow = X["dayofweek"].astype(str).str.slice(2).astype(int)
    X["is_weekend"] = (dow >= 6).astype(int)
    X["is_holiday"] = ((md >= 1220) | (md <= 103) | ((md >= 701) & (md <= 707)) |
                       ((md >= 1122) & (md <= 1128)) | (md == 704) | (md == 1111)).astype(int)
    for b in CBINS:
        X[f"block{b}"] = pd.Categorical((dep // b).astype(int).astype(str),
                                        categories=cat_levels[f"block{b}"])
    for b in CBINS:
        X[f"carrier_block{b}"] = pd.Categorical(
            df["UniqueCarrier"].astype(str) + "_" + (dep // b).astype(int).astype(str),
            categories=cat_levels[f"UniqueCarrier_block{b}"])
    for b in (20, 45):
        X[f"origin_block{b}"] = pd.Categorical(
            df["Origin"].astype(str) + "_" + (dep // b).astype(int).astype(str),
            categories=cat_levels[f"Origin_block{b}"])
    return X


BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "enable_categorical": True,
    "seed": SEED,
    "n_jobs": N_JOBS,
}

FULL = prepare(train)
FULL_EVAL = prepare(evald)
y_eval = (evald[TARGET] == POSITIVE).astype(int).to_numpy()

B6CATS = ["month", "dayofmonth", "dayofweek", "carrier", "origin", "dest"]
FL = ["is_weekend", "is_holiday"]
EXT_F = B6CATS + ["DepTime", "distance"] + FL + ["dep_sin", "dep_cos"]
HC_F = EXT_F + ["hour_cat", "hour_num"]


def fit_member(name, cols, over, n_rounds, seed=SEED):
    params = {**BASE_PARAMS, **over, "seed": seed}
    dtr = xgb.DMatrix(FULL[cols], label=y_all.to_numpy(), enable_categorical=True)
    bst = xgb.train(params, dtr, num_boost_round=n_rounds)
    p = bst.predict(xgb.DMatrix(FULL_EVAL[cols], enable_categorical=True))
    auc = roc_auc_score(y_eval, p)
    print(f"MEM {name:24s} eval={auc:.4f}")
    return (bst, cols, p)


C3 = {"max_depth": 6, "eta": 0.05, "min_child_weight": 20.0, "colsample_bynode": 0.3}
HCB30 = HC_F + ["block30"]
HCB15 = HC_F + ["block15"]
HCB20 = HC_F + ["block20"]
HCB10 = HC_F + ["block10"]
HCB45 = HC_F + ["block45"]
HCB60 = HC_F + ["block60"]

MODELS = []
MODELS.append(fit_member("q4_b30_400", HCB30, C3, 400))
MODELS.append(fit_member("b15_400", HCB15, C3, 400))
MODELS.append(fit_member("b20_400", HCB20, C3, 400))
MODELS.append(fit_member("b1530_400", HC_F + ["block15", "block30"], C3, 400))
MODELS.append(fit_member("b10_400", HCB10, C3, 400))
MODELS.append(fit_member("trio_400", HC_F + ["block15", "block30", "block60"], C3, 400))
MODELS.append(fit_member("quad_400", HC_F + ["block10", "block15", "block30"], C3, 400))
MODELS.append(fit_member("b1530_600", HC_F + ["block15", "block30"], C3, 600))
MODELS.append(fit_member("carb15_400", HC_F + ["block15", "carrier_block15"], C3, 400))
MODELS.append(fit_member("carb30_400", HC_F + ["block30", "carrier_block30"], C3, 400))
MODELS.append(fit_member("carb45_400", HC_F + ["block45", "carrier_block45"], C3, 400))
MODELS.append(fit_member("carb20_400", HC_F + ["block20", "carrier_block20"], C3, 400))
MODELS.append(fit_member("carb15_800", HC_F + ["block15", "carrier_block15"], C3, 800))
MODELS.append(fit_member("carb30_600", HC_F + ["block30", "carrier_block30"], C3, 600))
MODELS.append(fit_member("carb30_1500", HC_F + ["block30", "carrier_block30"], C3, 1500))
MODELS.append(fit_member("carb15_1000", HC_F + ["block15", "carrier_block15"], C3, 1000))
MODELS.append(fit_member("carb20_800", HC_F + ["block20", "carrier_block20"], C3, 800))
MODELS.append(fit_member("carb45_800", HC_F + ["block45", "carrier_block45"], C3, 800))
MODELS.append(fit_member("carb10_800", HC_F + ["block10", "carrier_block10"], C3, 800))
MODELS.append(fit_member("carb60_800", HCB60 + ["carrier_block60"], C3, 800))
MODELS.append(fit_member("origb20_800", HCB20 + ["origin_block20"], C3, 800))
MODELS.append(fit_member("origb45_800", HCB45 + ["origin_block45"], C3, 800))
MODELS.append(fit_member("carb30_1000", HC_F + ["block30", "carrier_block30"], C3, 1000))


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = prepare(df)
    ps = np.column_stack([b.predict(xgb.DMatrix(P[c], enable_categorical=True)) for b, c, _ in MODELS])
    return ps.mean(axis=1)


eval_auc = roc_auc_score(y_eval, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
