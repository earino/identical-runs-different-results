"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, module-level `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives inside `prepare(df)`; fitted statistics come from train.csv only.
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
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature engineering (all inside prepare: must apply to the hidden holdout) --
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
HOUR_LEVELS = pd.Index(sorted((pd.to_numeric(train["DepTime"]) // 100).unique()))
T30_LEVELS = pd.Index(sorted((pd.to_numeric(train["DepTime"]) // 30).unique()))
_hstr = (pd.to_numeric(train["DepTime"]) // 100).astype(str)
CARRIER_HOUR_LEVELS = pd.Index(sorted((train["UniqueCarrier"] + "_" + _hstr).unique()))
CNT_OHOUR = (train["Origin"] + "_" + _hstr).value_counts().to_dict()
CNT_DHOUR = (train["Dest"] + "_" + _hstr).value_counts().to_dict()
# hub-ness: share of a carrier's 2005 flights at this origin/dest
_oc = pd.crosstab(train["UniqueCarrier"], train["Origin"], normalize="index")
OC_SHARE = _oc.stack().to_dict()
_dc = pd.crosstab(train["UniqueCarrier"], train["Dest"], normalize="index")
DC_SHARE = _dc.stack().to_dict()
# schedule shape: share of an airport's 2005 flights departing at this hour
SHARE_O_HOUR = pd.crosstab(train["Origin"], pd.to_numeric(train["DepTime"]) // 100,
                           normalize="index").stack().to_dict()
SHARE_D_HOUR = pd.crosstab(train["Dest"], pd.to_numeric(train["DepTime"]) // 100,
                           normalize="index").stack().to_dict()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = pd.to_numeric(df["DepTime"], errors="coerce")
    X["hour"] = X["DepTime"] // 100
    X["minute"] = X["DepTime"] % 100
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["cat_hour"] = pd.Categorical(X["hour"], categories=HOUR_LEVELS)
    X["cat_t30"] = pd.Categorical(X["DepTime"] // 30, categories=T30_LEVELS)
    for c in CAT_COLS:
        X[f"cat_{c}"] = pd.Categorical(df[c], categories=CAT_LEVELS[c])  # unseen -> NaN
    X["cat_carrier_hour"] = pd.Categorical(df["UniqueCarrier"] + "_" + X["hour"].astype(str),
                                           categories=CARRIER_HOUR_LEVELS)
    X["cnt_ohour"] = [np.log1p(CNT_OHOUR.get(f"{o}_{h}", 0)) for o, h in zip(df["Origin"], X["hour"])]
    X["cnt_dhour"] = [np.log1p(CNT_DHOUR.get(f"{d}_{h}", 0)) for d, h in zip(df["Dest"], X["hour"])]
    X["oc_share"] = [OC_SHARE.get((c, o), 0.0) for o, c in zip(df["Origin"], df["UniqueCarrier"])]
    X["dc_share"] = [DC_SHARE.get((c, o), 0.0) for o, c in zip(df["Dest"], df["UniqueCarrier"])]
    X["share_o_hour"] = [SHARE_O_HOUR.get((o, h), np.nan) for o, h in zip(df["Origin"], X["hour"])]
    X["share_d_hour"] = [SHARE_D_HOUR.get((o, h), np.nan) for o, h in zip(df["Dest"], X["hour"])]
    return X


FEATURES = ["DepTime", "Distance", "hour", "minute",
            "cat_Month", "cat_DayofMonth", "cat_DayOfWeek", "cat_UniqueCarrier",
            "cat_Origin", "cat_Dest", "cat_hour", "cat_t30", "cat_carrier_hour",
            "cnt_ohour", "cnt_dhour", "oc_share", "dc_share",
            "share_o_hour", "share_d_hour"]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: heterogeneous ensemble of XGBoost models ---------------------------
BASE_PARAMS = dict(max_depth=8, learning_rate=0.02, colsample_bytree=0.5, reg_lambda=2.0,
                   tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)
MEMBERS = [
    dict(n_estimators=1200, random_state=1),
    dict(n_estimators=1200, random_state=2),
    dict(n_estimators=1200, random_state=3),
    dict(n_estimators=900, max_depth=9, random_state=11),
    dict(n_estimators=1600, learning_rate=0.015, random_state=12),
]

t0 = time.time()
X_train = prepare(train)[FEATURES]
y_train = to_y(train)
models = []
for kw in MEMBERS:
    p = dict(BASE_PARAMS)
    p.update(kw)
    m = xgb.XGBClassifier(**p)
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)[FEATURES]
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
