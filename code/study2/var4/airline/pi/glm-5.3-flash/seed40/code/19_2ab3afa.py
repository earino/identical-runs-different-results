"""XGBoost airline delay classifier — final architecture:
equal-weight blend of 4 full-data XGBoost models on engineered features.
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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: sorted(train[c].astype(str).unique()) for c in CAT_COLS}
dt5_levels = sorted((train["DepTime"].astype("int64") // 5).unique())
CUMDAYS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
PRIOR = float((train[TARGET] == POSITIVE).mean())


def _rate(keys: pd.Series, y: np.ndarray, k: float) -> pd.Series:
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + k * PRIOR) / (g["count"] + k)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


_y = to_y(train)
rate_tables = {
    "r_month": _rate(train["Month"], _y, 75.0),
    "r_carrier": _rate(train["UniqueCarrier"], _y, 75.0),
    "r_origin": _rate(train["Origin"], _y, 100.0),
    "r_dest": _rate(train["Dest"], _y, 100.0),
    "r_hb": _rate(pd.Series(((train["DepTime"].astype("int64") // 100) // 3), index=train.index), _y, 100.0),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here; fitted stats come from train only.
    X = pd.DataFrame(index=df.index)
    m = df["Month"].str[2:].astype(int).to_numpy()
    dom = df["DayofMonth"].str[2:].astype(int).to_numpy()
    dow = df["DayOfWeek"].str[2:].astype(int).to_numpy()
    dt = df["DepTime"].astype("int64").to_numpy()
    dep_min = (dt // 100) * 60 + dt % 100
    doy = CUMDAYS[m - 1] + dom
    hb = (dt // 100) // 3
    X["DepTime"] = dt
    X["Distance"] = df["Distance"].to_numpy()
    X["dep_min"] = dep_min
    X["dep_sin"] = np.sin(2 * np.pi * dep_min / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * dep_min / 1440.0)
    X["DepTime_mod"] = dt % 100
    X["hour"] = dt // 100
    X["hb"] = hb
    X["doy"] = doy
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    X["dow"] = dow
    X["log_dist"] = np.log1p(df["Distance"].to_numpy())
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    X["dt5"] = pd.Categorical(dt // 5, categories=dt5_levels)
    X["r_month"] = rate_tables["r_month"].reindex(df["Month"]).fillna(PRIOR).to_numpy()
    X["r_carrier"] = rate_tables["r_carrier"].reindex(df["UniqueCarrier"]).fillna(PRIOR).to_numpy()
    X["r_origin"] = rate_tables["r_origin"].reindex(df["Origin"]).fillna(PRIOR).to_numpy()
    X["r_dest"] = rate_tables["r_dest"].reindex(df["Dest"]).fillna(PRIOR).to_numpy()
    X["r_hb"] = rate_tables["r_hb"].reindex(pd.Series(hb, index=df.index)).fillna(PRIOR).to_numpy()
    ld = np.log1p(df["Distance"].to_numpy())
    X["p_sd"] = X["dep_sin"].to_numpy() * ld
    X["p_cd"] = X["dep_cos"].to_numpy() * ld
    X["p_hd"] = (dt // 100) * ld
    X["p_sD"] = X["dep_sin"].to_numpy() * X["doy_sin"].to_numpy()
    X["p_cD"] = X["dep_cos"].to_numpy() * X["doy_cos"].to_numpy()
    X["p_sDc"] = X["dep_sin"].to_numpy() * X["doy_cos"].to_numpy()
    X["p_cDs"] = X["dep_cos"].to_numpy() * X["doy_sin"].to_numpy()
    return X


BASE = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "DepTime", "Distance",
        "dep_min", "dep_sin", "dep_cos", "DepTime_mod", "hour", "hb", "doy", "doy_sin", "doy_cos", "dow", "log_dist"]
RATES1 = ["r_month", "r_carrier", "r_origin", "r_dest", "r_hb"]
PROD = ["p_sd", "p_cd", "p_hd", "p_sD", "p_cD", "p_sDc", "p_cDs"]
COLS = BASE + RATES1
COLS_PROD = COLS + PROD
COLS_DT5 = COLS_PROD + ["dt5"]

X_full, y_all = prepare(train), to_y(train)
X_ev, y_ev = prepare(evald), to_y(evald)

MEMC = {
    "cs3bp": (dict(n_estimators=400, max_depth=4, learning_rate=0.05, colsample_bytree=0.3), COLS_PROD, 7),
    "d4g": (dict(n_estimators=800, max_depth=4, learning_rate=0.03, gamma=2.0), COLS, SEED),
    "forest": (dict(n_estimators=1, num_parallel_tree=20, learning_rate=1.0, subsample=0.8,
                    colsample_bynode=0.8, max_depth=4), COLS_PROD, 99),
    "cs3dt5": (dict(n_estimators=400, max_depth=4, learning_rate=0.05, colsample_bytree=0.3), COLS_DT5, 7),
}

t00 = time.time()
models, cols_list = [], []
for n, (cfg, cols, seed) in MEMC.items():
    mdl = xgb.XGBClassifier(
        tree_method="hist", enable_categorical=True, random_state=seed, n_jobs=N_JOBS,
        eval_metric="auc", **cfg,
    )
    mdl.fit(X_full[cols], y_all)
    p = mdl.predict_proba(X_ev[cols])[:, 1]
    print(f"Eval AUC: {roc_auc_score(y_ev, p):.4f}   member {n}")
    models.append(mdl)
    cols_list.append(cols)

out = np.mean([m.predict_proba(X_ev[c])[:, 1] for m, c in zip(models, cols_list)], axis=0)
final_auc = roc_auc_score(y_ev, out)
print(f"blend total {time.time()-t00:.0f}s")
print(f"Eval AUC: {final_auc:.4f}")


class Bag:
    def __init__(self, models, cols_list):
        self.models, self.cols_list = models, cols_list

    def predict_proba(self, X):
        ps = [m.predict_proba(X[cols])[:, 1] for m, cols in zip(self.models, self.cols_list)]
        out = np.mean(ps, axis=0)
        return np.column_stack([1 - out, out])


model = Bag(models, cols_list)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]
