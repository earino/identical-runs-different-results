"""XGBoost binary classifier for airline delays — experiment 2.
Feature engineering (time cycles, route, frequencies) + bigger early-stopped model.
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

# --- fitted stats (train only) -------------------------------------------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: sorted(train[c].astype(str).unique()) for c in CAT_COLS}
route_levels = sorted((train["Origin"].astype(str) + ">" + train["Dest"].astype(str)).unique())
FREQ_COLS = ["UniqueCarrier", "Origin", "Dest"]
freq_maps = {c: train[c].value_counts(normalize=True) for c in FREQ_COLS}
freq_maps["Route"] = (train["Origin"].astype(str) + ">" + train["Dest"].astype(str)).value_counts(normalize=True)
CUMDAYS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here; fitted maps come from train only.
    X = pd.DataFrame(index=df.index)
    m = df["Month"].str[2:].astype(int).to_numpy()
    dom = df["DayofMonth"].str[2:].astype(int).to_numpy()
    dow = df["DayOfWeek"].str[2:].astype(int).to_numpy()
    dt = df["DepTime"].astype("int64").to_numpy()
    dep_min = (dt // 100) * 60 + dt % 100
    doy = CUMDAYS[m - 1] + dom
    X["DepTime"] = dt
    X["dep_min"] = dep_min
    X["dep_sin"] = np.sin(2 * np.pi * dep_min / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * dep_min / 1440.0)
    X["DepTime_mod"] = dt % 100
    X["hour"] = dt // 100
    X["doy"] = doy
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    X["dow"] = dow
    X["Distance"] = df["Distance"].to_numpy()
    X["log_dist"] = np.log1p(df["Distance"].to_numpy())
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    X["Route"] = pd.Categorical(
        df["Origin"].astype(str) + ">" + df["Dest"].astype(str), categories=route_levels
    )
    X["o_freq"] = freq_maps["Origin"].reindex(df["Origin"]).fillna(0.0).to_numpy()
    X["d_freq"] = freq_maps["Dest"].reindex(df["Dest"]).fillna(0.0).to_numpy()
    X["c_freq"] = freq_maps["UniqueCarrier"].reindex(df["UniqueCarrier"]).fillna(0.0).to_numpy()
    X["r_freq"] = freq_maps["Route"].reindex(df["Origin"].astype(str) + ">" + df["Dest"].astype(str)).fillna(0.0).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model ---------------------------------------------------------------------
def make_model(n_est, es_rounds=None):
    params = dict(
        n_estimators=n_est,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=5.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
        eval_metric="auc",
    )
    if es_rounds is not None:
        params["early_stopping_rounds"] = es_rounds
    return xgb.XGBClassifier(**params)


rng = np.random.default_rng(SEED)
idx = rng.permutation(len(train))
val_idx = idx[:15000]
tr_idx = idx[15000:]
X_tr, y_tr = prepare(train.iloc[tr_idx]), to_y(train.iloc[tr_idx])
X_val, y_val = prepare(train.iloc[val_idx]), to_y(train.iloc[val_idx])

t0 = time.time()
es_model = make_model(3000, es_rounds=100)
es_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
best_n = es_model.best_iteration + 1
print(f"ES model: best_iteration={best_n}, val_auc={es_model.best_score:.5f}, time={time.time()-t0:.1f}s")

t0 = time.time()
model = make_model(int(best_n * 1.15) + 1)
model.fit(prepare(train), to_y(train))
print(f"Full-fit time: {time.time()-t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time()-t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
