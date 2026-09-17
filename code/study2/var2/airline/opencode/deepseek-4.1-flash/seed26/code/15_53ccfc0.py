"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

FREQ_SPECS = {
    "freq_carrier": train["UniqueCarrier"],
    "freq_origin": train["Origin"],
    "freq_dest": train["Dest"],
    "freq_route": train["Origin"] + "_" + train["Dest"],
}
_th = ((train["DepTime"].to_numpy() // 100) % 24).astype(str)
_freq_oh = (train["Origin"].to_numpy().astype(str) + "_" + _th)
_freq_ch = (train["UniqueCarrier"].to_numpy().astype(str) + "_" + _th)
freq_maps = {k: v.value_counts() for k, v in FREQ_SPECS.items()}
freq_maps["freq_origin_hour"] = pd.Series(_freq_oh).value_counts()
freq_maps["freq_carrier_hour"] = pd.Series(_freq_ch).value_counts()
_freq_dh = (train["Dest"].to_numpy().astype(str) + "_" + _th)
freq_maps["freq_dest_hour"] = pd.Series(_freq_dh).value_counts()
_freq_rh = ((train["Origin"] + "_" + train["Dest"]).to_numpy().astype(str) + "_" + _th)
freq_maps["freq_route_hour"] = pd.Series(_freq_rh).value_counts()


def _add_freq(X: pd.DataFrame, df: pd.DataFrame) -> None:
    X["freq_carrier"] = df["UniqueCarrier"].map(freq_maps["freq_carrier"]).fillna(0).to_numpy()
    X["freq_origin"] = df["Origin"].map(freq_maps["freq_origin"]).fillna(0).to_numpy()
    X["freq_dest"] = df["Dest"].map(freq_maps["freq_dest"]).fillna(0).to_numpy()
    r = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["freq_route"] = r.map(freq_maps["freq_route"]).fillna(0).to_numpy()
    h = ((df["DepTime"].to_numpy() // 100) % 24).astype(str)
    oh = df["Origin"].to_numpy().astype(str) + "_" + h
    ch = df["UniqueCarrier"].to_numpy().astype(str) + "_" + h
    X["freq_origin_hour"] = pd.Series(oh).map(freq_maps["freq_origin_hour"]).fillna(0).to_numpy()
    X["freq_carrier_hour"] = pd.Series(ch).map(freq_maps["freq_carrier_hour"]).fillna(0).to_numpy()
    dh = df["Dest"].to_numpy().astype(str) + "_" + h
    X["freq_dest_hour"] = pd.Series(dh).map(freq_maps["freq_dest_hour"]).fillna(0).to_numpy()
    rh = r.to_numpy().astype(str) + "_" + h
    X["freq_route_hour"] = pd.Series(rh).map(freq_maps["freq_route_hour"]).fillna(0).to_numpy()
    X["origin_hour_share"] = X["freq_origin_hour"].to_numpy() / np.maximum(X["freq_origin"].to_numpy(), 1.0)
    X["route_hour_share"] = X["freq_route_hour"].to_numpy() / np.maximum(X["freq_route"].to_numpy(), 1.0)


TIME_FEATS = ["hour", "minute", "dep_min"]
_MONTH_START = np.array([0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    t = X["DepTime"].to_numpy()
    hour = (t // 100) % 24
    X["hour"] = hour
    X["minute"] = t % 100
    X["dep_min"] = hour * 60 + (t % 100)
    ang = 2.0 * np.pi * X["dep_min"] / 1440.0
    X["dep_sin"] = np.sin(ang)
    X["dep_cos"] = np.cos(ang)
    mon = X["Month"].str.replace("c-", "", regex=False).astype(int).to_numpy()
    day = X["DayofMonth"].str.replace("c-", "", regex=False).astype(int).to_numpy()
    doy = _MONTH_START[mon] + day
    X["doy"] = doy
    X["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.0)
    X["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.0)
    X["hour_cat"] = pd.Categorical(hour, categories=np.arange(24))
    X["dep15_cat"] = pd.Categorical(X["dep_min"] // 15, categories=np.arange(96))
    _add_freq(X, df)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model ensemble -----------------------------------------------------------
X_train = prepare(train)
y_train = to_y(train)

MODELS = []
t0 = time.time()
for s, depth, ss, cs in [
    (42, 6, 0.8, 0.8),
    (137, 6, 0.8, 0.8),
    (2024, 6, 0.8, 0.8),
    (7, 7, 0.7, 0.7),
    (999, 5, 0.9, 0.9),
    (555, 7, 0.8, 0.7),
    (314, 5, 0.8, 0.8),
    (271, 6, 0.7, 0.9),
    (11, 6, 0.9, 0.7),
    (22, 7, 0.9, 0.9),
    (33, 5, 0.7, 0.7),
    (44, 6, 0.6, 0.8),
    (55, 8, 0.8, 0.8),
    (66, 5, 0.85, 0.85),
    (77, 7, 0.75, 0.8),
    (88, 6, 0.95, 0.75),
]:
    m = xgb.XGBClassifier(
        n_estimators=450,
        max_depth=depth,
        learning_rate=0.025,
        subsample=ss,
        colsample_bytree=cs,
        min_child_weight=5,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=s,
        n_jobs=N_JOBS,
    )
    m.fit(X_train, y_train)
    MODELS.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.zeros(len(X))
    for m in MODELS:
        p += m.predict_proba(X)[:, 1]
    return p / len(MODELS)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
