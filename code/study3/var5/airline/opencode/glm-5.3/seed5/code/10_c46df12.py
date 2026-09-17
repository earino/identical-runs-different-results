"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Notes: 2005 -> 2006 time shift means (a) date/calendar features and high-cardinality
route/dest categoricals let the model memorize 2005 noise and HURT eval; (b) random-split
early stopping inside train.csv picks too many rounds. So: robust features only
(time-of-day, distance, carrier, origin), fixed round count, deep trees + L1.
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
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in CAT_COLS}

# label-free traffic statistics (fit on train only; schedule structure repeats year-over-year)
_hour_tr = (pd.to_numeric(train["DepTime"], errors="coerce") // 100).astype("Int64")
_tod_tr = _hour_tr * 60 + (pd.to_numeric(train["DepTime"], errors="coerce") % 100)
_arr_tr = ((_tod_tr + 30 + pd.to_numeric(train["Distance"], errors="coerce") * 0.12) % 1440) // 60
_oh_cnt = train.assign(k=train["Origin"].astype(str) + "_" + _hour_tr.astype(str)).groupby("k").size()
_dh_cnt = train.assign(k=train["Dest"].astype(str) + "_" + _hour_tr.astype(str)).groupby("k").size()
_ah_cnt = train.assign(k=train["Dest"].astype(str) + "_" + _arr_tr.astype("Int64").astype(str)).groupby("k").size()
_co_cnt = train.assign(k=train["UniqueCarrier"].astype(str) + "_" + train["Origin"].astype(str)).groupby("k").size()
_o_tot = train["Origin"].astype(str).value_counts()
_d_tot = train["Dest"].astype(str).value_counts()
_r_cnt = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).value_counts()
_g_hour = _hour_tr.value_counts()
_g_tot = float(len(train))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dt // 100).astype("float64")
    X["DepTime"] = dt
    X["TOD"] = hour * 60 + (dt - hour * 100)
    X["Hour"] = hour
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    oh = df["Origin"].astype(str) + "_" + hour.astype("Int64").astype(str)
    dh = df["Dest"].astype(str) + "_" + hour.astype("Int64").astype(str)
    X["OHcnt"] = np.log1p(oh.map(_oh_cnt).fillna(0).astype(float))
    X["OHfrac"] = oh.map(_oh_cnt).fillna(0).astype(float) / df["Origin"].astype(str).map(_o_tot).fillna(1).astype(float)
    X["DHcnt"] = np.log1p(dh.map(_dh_cnt).fillna(0).astype(float))
    X["Rcnt"] = np.log1p((df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).map(_r_cnt).fillna(0).astype(float))
    X["HourShare"] = hour.astype("Int64").map(_g_hour).fillna(0).astype(float) / _g_tot
    arr = ((X["TOD"] + 30 + X["Distance"] * 0.12) % 1440) // 60
    X["AHcnt"] = np.log1p((df["Dest"].astype(str) + "_" + arr.astype("Int64").astype(str)).map(_ah_cnt).fillna(0).astype(float))
    X["AHfrac"] = (df["Dest"].astype(str) + "_" + arr.astype("Int64").astype(str)).map(_ah_cnt).fillna(0).astype(float) / df["Dest"].astype(str).map(_d_tot).fillna(1).astype(float)
    X["COcnt"] = np.log1p((df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str)).map(_co_cnt).fillna(0).astype(float))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: heterogeneous ensemble --------------------------------------------
# Deep members capture fine structure; shallow/long-boosted members (d1-d3, lr .02-.05)
# are far more robust to the 2005->2006 drift and add strong ensemble diversity;
# subset members (numeric-only, cats-heavy) decorrelate further.
_NUM_COLS = ["DepTime", "TOD", "Hour", "Distance", "OHcnt", "OHfrac", "DHcnt", "Rcnt", "HourShare", "AHcnt", "AHfrac", "COcnt"]
_CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
_CAT_HEAVY = ["DepTime", "TOD"] + _CAT_COLS + ["Distance"]
CONFIGS = [
    (None, dict(n_estimators=300, max_depth=16, reg_alpha=1.0, max_bin=1024)),
    (None, dict(n_estimators=300, max_depth=20, reg_alpha=1.0, max_bin=512)),
    (_NUM_COLS, dict(n_estimators=400, max_depth=16, reg_alpha=1.0, max_bin=512)),
    (_CAT_HEAVY, dict(n_estimators=400, max_depth=16, reg_alpha=1.0, max_bin=512)),
    (None, dict(n_estimators=1200, max_depth=3, reg_alpha=1.0, max_bin=512), 0.05),
    (None, dict(n_estimators=1500, max_depth=2, reg_alpha=1.0, max_bin=512), 0.05),
    (_NUM_COLS, dict(n_estimators=1200, max_depth=3, reg_alpha=1.0, max_bin=512), 0.05),
    (None, dict(n_estimators=3000, max_depth=1, reg_alpha=1.0, max_bin=512), 0.02),
]

t0 = time.time()
X_train = prepare(train)
y_train = to_y(train)
models = []
for cfg in CONFIGS:
    cols, kw = cfg[0], cfg[1]
    lr = cfg[2] if len(cfg) > 2 else 0.1
    X = X_train if cols is None else X_train[cols]
    m = xgb.XGBClassifier(learning_rate=lr, tree_method="hist", enable_categorical=True,
                          random_state=SEED, n_jobs=N_JOBS, **kw)
    m.fit(X, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    P = np.array([m.predict_proba(X if c[0] is None else X[c[0]])[:, 1]
                  for c, m in zip(CONFIGS, models)])
    Pc = np.clip(P, 1e-6, 1 - 1e-6)  # average in log-odds space: better than prob-space mean
    return 1 / (1 + np.exp(-np.log(Pc / (1 - Pc)).mean(axis=0)))


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
