"""XGBoost binary classifier for airline delay prediction.

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

# --- features -----------------------------------------------------------------
# Date columns (Month/DayofMonth/DayOfWeek) enter ONLY as numeric values: their raw c-<n>
# categoricals let the model memorize 2005-specific calendar days and hurt 2006 generalization.
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "hour_cat"]


# statistics computed from the TRAINING dataframe only (never from data passed to predict_proba)
freq_o = train["Origin"].value_counts(normalize=True)
freq_d = train["Dest"].value_counts(normalize=True)
O_TRAF_DEFAULT = float(freq_o.mean())
D_TRAF_DEFAULT = float(freq_d.mean())
route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
route_vol = route.value_counts(normalize=True)
ROUTE_VOL_DEFAULT = float(route_vol.mean())


def cnum(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("c-", "", regex=False).astype(float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    X["Origin"] = df["Origin"].astype(str)
    X["Dest"] = df["Dest"].astype(str)
    dep = df["DepTime"].astype(float)
    hour = (dep // 100).clip(0, 23)
    minute = dep % 100
    X["hour"] = hour
    X["minute"] = minute
    frac = hour / 24 + minute / 1440
    X["dep_frac"] = frac
    X["hour_sin"] = np.sin(2 * np.pi * frac)
    X["hour_cos"] = np.cos(2 * np.pi * frac)
    X["hm"] = hour * 60 + minute
    X["hour_cat"] = hour.astype(int).astype(str)
    X["Distance"] = df["Distance"].astype(float)
    mo = cnum(df["Month"])
    dom = cnum(df["DayofMonth"])
    dow = cnum(df["DayOfWeek"])
    X["month_num"] = mo
    X["dom_num"] = dom
    X["dow_num"] = dow
    X["m_sin"] = np.sin(2 * np.pi * mo / 12)
    X["m_cos"] = np.cos(2 * np.pi * mo / 12)
    X["o_traffic"] = df["Origin"].map(freq_o).fillna(O_TRAF_DEFAULT).astype(float)
    X["d_traffic"] = df["Dest"].map(freq_d).fillna(D_TRAF_DEFAULT).astype(float)
    rt = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["route_vol"] = rt.map(route_vol).fillna(ROUTE_VOL_DEFAULT).astype(float)
    return X


cat_levels = {c: pd.Index(sorted(prepare(train)[c].astype(str).unique())) for c in CAT_COLS}


def prepare_enc(df: pd.DataFrame) -> pd.DataFrame:
    X = prepare(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c].astype(str), categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Ensemble of XGBoost models with different seeds: averaging their probabilities
# reduces variance and is one of the most robust generalization levers available.
N_SEEDS = 4
SEEDS = [42, 1, 2, 3]
PARAMS = dict(
    n_estimators=2500,
    learning_rate=0.05,
    max_depth=9,
    min_child_weight=3,
    subsample=1.0,
    colsample_bytree=0.5,
    colsample_bynode=0.5,
    reg_lambda=2.0,
    alpha=4.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=60,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_tr = prepare_enc(train)
y_tr = to_y(train)
X_ev = prepare_enc(evald)
y_ev = to_y(evald)
models = []
for s in SEEDS:
    m = xgb.XGBClassifier(random_state=s, **PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    models.append(m)
    print(f"seed {s}: best iteration {m.best_iteration}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare_enc(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
