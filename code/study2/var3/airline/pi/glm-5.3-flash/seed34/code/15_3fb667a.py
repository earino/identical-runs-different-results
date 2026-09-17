"""Airline delay XGBoost classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# volume counts from training data only (no target information)
_v = pd.DataFrame({
    "Origin": train["Origin"].astype(str),
    "Dest": train["Dest"].astype(str),
    "UniqueCarrier": train["UniqueCarrier"].astype(str),
    "Route": train["Origin"].astype(str) + "_" + train["Dest"].astype(str),
})
VOL = {c: _v.groupby(c).size().to_dict() for c in ["Origin", "Dest", "UniqueCarrier", "Route"]}
VOL["CarrierOrigin"] = _v.groupby(["UniqueCarrier", "Origin"]).size().to_dict()
VOL["OriginDegree"] = _v.groupby("Origin")["Dest"].nunique().to_dict()
VOL["DestDegree"] = _v.groupby("Dest")["Origin"].nunique().to_dict()
VOL["CarrierDegree"] = _v.groupby("UniqueCarrier")["Dest"].nunique().to_dict()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["Month"] = df["Month"].str.lstrip("c-").astype(float)
    X["DayofMonth"] = df["DayofMonth"].str.lstrip("c-").astype(float)
    X["DayOfWeek"] = df["DayOfWeek"].str.lstrip("c-").astype(float)
    X["Distance"] = df["Distance"].astype(float)
    X["LogDistance"] = np.log1p(X["Distance"])
    X["DistBucket"] = (X["Distance"] // 250).astype(float)
    t = df["DepTime"].fillna(-1).astype(int)
    hour = (t // 100) % 24
    X["DepHour"] = hour
    X["DepMin"] = hour * 60 + t % 100
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    rt = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["OriginVol"] = np.log1p(df["Origin"].astype(str).map(VOL["Origin"]).fillna(0).astype(float))
    X["DestVol"] = np.log1p(df["Dest"].astype(str).map(VOL["Dest"]).fillna(0).astype(float))
    X["CarrierVol"] = np.log1p(df["UniqueCarrier"].astype(str).map(VOL["UniqueCarrier"]).fillna(0).astype(float))
    X["RouteVol"] = np.log1p(rt.map(VOL["Route"]).fillna(0).astype(float))
    X["CarrierOriginVol"] = np.log1p((df["UniqueCarrier"].astype(str) + "|" + df["Origin"].astype(str)).map(VOL["CarrierOrigin"]).fillna(0).astype(float))
    X["OriginDegree"] = np.log1p(df["Origin"].astype(str).map(VOL["OriginDegree"]).fillna(0).astype(float))
    X["DestDegree"] = np.log1p(df["Dest"].astype(str).map(VOL["DestDegree"]).fillna(0).astype(float))
    X["CarrierDegree"] = np.log1p(df["UniqueCarrier"].astype(str).map(VOL["CarrierDegree"]).fillna(0).astype(float))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


PARAMS = dict(
    n_estimators=800,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.8,
    min_child_weight=10.0,
    reg_lambda=2.0,
    eval_metric="auc",
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    max_bin=512,
)


def make_model(**kw):
    p = dict(PARAMS)
    p.update(kw)
    return xgb.XGBClassifier(**p)


# base config: CV-selected twice (experiments 11/13): colsample .4, depth 14, ~238 trees
# CV-selected (exp 37 grid, within-2005 2-fold): col .35, depth 20, mcw 1, n=68
BEST_CFG = {"colsample_bytree": 0.35, "max_depth": 20, "min_child_weight": 1.0}
BEST_N = 68


t0 = time.time()
X = prepare(train)
y = to_y(train)
# recency weighting: later months of 2005 are closer to the 2006 target year
_month = X["Month"].to_numpy()
sw = 1.0 + 2.0 * (_month - 1) / 11.0
models = []
for seed in [42, 1337, 2024, 7, 99, 555, 314, 2718]:
    cfg = dict(BEST_CFG)
    cfg["n_estimators"] = BEST_N
    cfg["random_state"] = seed
    m = make_model(**cfg)
    m.fit(X, y, sample_weight=sw)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xd = prepare(df)
    ps = [m.predict_proba(Xd)[:, 1] for m in models]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
