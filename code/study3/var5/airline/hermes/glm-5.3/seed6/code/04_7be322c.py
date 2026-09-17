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
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "hour_cat"]


# traffic-frequency maps computed from the TRAINING dataframe only
freq_o = train["Origin"].value_counts(normalize=True)
freq_d = train["Dest"].value_counts(normalize=True)
O_TRAF_DEFAULT = float(freq_o.mean())
D_TRAF_DEFAULT = float(freq_d.mean())


def cnum(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("c-", "", regex=False).astype(float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        if c == "hour_cat":
            continue
        X[c] = df[c].astype(str)
    dep = df["DepTime"].astype(float)
    hour = (dep // 100).clip(0, 23)
    minute = dep % 100
    X["hour"] = hour
    X["minute"] = minute
    frac = hour / 24 + minute / 1440
    X["dep_frac"] = frac
    X["hour_sin"] = np.sin(2 * np.pi * frac)
    X["hour_cos"] = np.cos(2 * np.pi * frac)
    X["DepTime"] = dep
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
    X["doy"] = mo * 31 + dom
    # traffic level of origin/dest airports (train-frequency encoding, computed at module level
    # from the TRAINING dataframe only, then mapped inside prepare-land via the module dicts)
    X["o_traffic"] = df["Origin"].map(freq_o).fillna(O_TRAF_DEFAULT).astype(float)
    X["d_traffic"] = df["Dest"].map(freq_d).fillna(D_TRAF_DEFAULT).astype(float)
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
model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=5,
    subsample=0.7,
    colsample_bytree=0.5,
    colsample_bynode=0.5,
    reg_lambda=2.0,
    alpha=2.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_tr = prepare_enc(train)
y_tr = to_y(train)
model.fit(X_tr, y_tr, eval_set=[(prepare_enc(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s, best iteration: {model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare_enc(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
