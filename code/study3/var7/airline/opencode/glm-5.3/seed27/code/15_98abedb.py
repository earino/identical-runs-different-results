"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
y_all = (train[TARGET] == POSITIVE).astype(int).to_numpy()

# --- encodings fitted on train ONLY (applied inside prepare()) ------------------
P0 = float(y_all.mean())


def te_table(keys: pd.Series, y: np.ndarray, k: int) -> pd.Series:
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + k * P0) / (g["count"] + k)


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    K = pd.DataFrame(index=df.index)
    K["origin"] = df["Origin"].astype(str)
    K["dest"] = df["Dest"].astype(str)
    K["carrier"] = df["UniqueCarrier"].astype(str)
    K["hour"] = df["DepTime"].astype(int) // 100
    K["dow"] = df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False).astype(int)
    dep = df["DepTime"].astype(int)
    K["t15"] = (dep // 100 * 4 + (dep % 100) // 15).astype(str)
    K["t20"] = (dep // 100 * 3 + (dep % 100) // 20).astype(str)
    K["route"] = K["origin"] + "_" + K["dest"]
    K["oh15"] = K["origin"] + "_" + K["t15"]
    K["dh15"] = K["dest"] + "_" + K["t15"]
    K["ch15"] = K["carrier"] + "_" + K["t15"]
    K["r15"] = K["route"] + "_" + K["t15"]
    K["r20"] = K["route"] + "_" + K["t20"]
    return K


K_train = _keys(train)
TE = {"origin": (K_train["origin"], 50), "route": (K_train["route"], 150),
      "t15": (K_train["t15"], 300), "oh15": (K_train["oh15"], 250),
      "dh15": (K_train["dh15"], 250), "ch15": (K_train["ch15"], 250), "r15": (K_train["r15"], 400),
      "r20": (K_train["r20"], 400)}
te_tables = {c: te_table(keys, y_all, k) for c, (keys, k) in TE.items()}
cnt_o, cnt_d = K_train["origin"].value_counts(), K_train["dest"].value_counts()
cnt_c, cnt_r = K_train["carrier"].value_counts(), K_train["route"].value_counts()

# out-of-fold TE values for the training rows themselves (prevents TE leakage)
OOF = {f"te_{c}": np.zeros(len(train)) for c in TE}
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for trn_idx, val_idx in kf.split(train):
    y_trn = y_all[trn_idx]
    for c, (keys, k) in TE.items():
        tbl = te_table(keys.iloc[trn_idx], y_trn, k)
        OOF[f"te_{c}"][val_idx] = keys.iloc[val_idx].map(tbl).fillna(P0).to_numpy()

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["month"] = df["Month"].astype(str).str.replace("c-", "", regex=False).astype(int)
    X["day"] = df["DayofMonth"].astype(str).str.replace("c-", "", regex=False).astype(int)
    X["dow"] = df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False).astype(int)
    dep = df["DepTime"].astype(int)
    X["dep_time"] = dep
    X["dep_hour"] = dep // 100
    X["minute_of_day"] = dep // 100 * 60 + dep % 100
    X["distance"] = df["Distance"].astype(float)
    K = _keys(df)
    X["cnt_origin"] = K["origin"].map(cnt_o).fillna(0).to_numpy()
    X["cnt_dest"] = K["dest"].map(cnt_d).fillna(0).to_numpy()
    X["cnt_carrier"] = K["carrier"].map(cnt_c).fillna(0).to_numpy()
    X["cnt_route"] = K["route"].map(cnt_r).fillna(0).to_numpy()
    X["te_origin"] = K["origin"].map(te_tables["origin"]).fillna(P0).to_numpy()
    X["te_route"] = K["route"].map(te_tables["route"]).fillna(P0).to_numpy()
    X["te_t15"] = K["t15"].map(te_tables["t15"]).fillna(P0).to_numpy()
    X["te_oh15"] = K["oh15"].map(te_tables["oh15"]).fillna(P0).to_numpy()
    X["te_dh15"] = K["dh15"].map(te_tables["dh15"]).fillna(P0).to_numpy()
    X["te_ch15"] = K["ch15"].map(te_tables["ch15"]).fillna(P0).to_numpy()
    X["te_r15"] = K["r15"].map(te_tables["r15"]).fillna(P0).to_numpy()
    X["te_r20"] = K["r20"].map(te_tables["r20"]).fillna(P0).to_numpy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- ensemble of deep XGBoost models -------------------------------------------
X_all = prepare(train)
for c in TE:  # honest out-of-fold TE values on the training rows
    X_all[f"te_{c}"] = OOF[f"te_{c}"]

t0 = time.time()
models = []
for s in range(1, 11):
    mcw = 3 if s % 2 else 6
    depth = 16 if s % 3 else 18
    m = xgb.XGBClassifier(
        n_estimators=150, max_depth=depth, learning_rate=0.02,
        tree_method="hist", enable_categorical=True, min_child_weight=mcw,
        colsample_bytree=0.30, max_bin=512, random_state=s, n_jobs=N_JOBS,
    )
    m.fit(X_all, y_all, verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
