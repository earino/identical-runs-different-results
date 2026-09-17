"""XGBoost binary classifier on airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

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
y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "Route"]
TE_KEYS = ["UniqueCarrier", "Origin", "Dest", "Route", "CarrierHour", "OriginHour",
           "DayOfWeek", "DepHour", "RouteHour", "DestHour", "CarrierDoW", "Month", "DayofMonth",
           "DepSlot15", "CarrierSlot15", "DistBin"]
TE_SMOOTH = 10.0

cat_levels = {
    "Month": pd.Index(sorted(train["Month"].unique())),
    "DayofMonth": pd.Index(sorted(train["DayofMonth"].unique())),
    "DayOfWeek": pd.Index(sorted(train["DayOfWeek"].unique())),
    "UniqueCarrier": pd.Index(sorted(train["UniqueCarrier"].unique())),
    "Origin": pd.Index(sorted(train["Origin"].unique())),
    "Dest": pd.Index(sorted(train["Dest"].unique())),
    "Route": pd.Index(sorted((train["Origin"] + "_" + train["Dest"]).unique())),
}


def base_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic per-row features (no fitted statistics)."""
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = pd.to_numeric(df["DepTime"], errors="coerce")
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["DepHour"] = (X["DepTime"] // 100).fillna(-1).astype(int)
    X["DepMin"] = X["DepTime"] % 100
    X["LogDist"] = np.log1p(X["Distance"].clip(lower=0))
    X["UniqueCarrier"] = df["UniqueCarrier"]
    X["Origin"] = df["Origin"]
    X["Dest"] = df["Dest"]
    X["Route"] = df["Origin"] + "_" + df["Dest"]
    X["CarrierHour"] = X["UniqueCarrier"] + "_" + X["DepHour"].astype(str)
    X["OriginHour"] = X["Origin"] + "_" + X["DepHour"].astype(str)
    X["RouteHour"] = X["Route"] + "_" + X["DepHour"].astype(str)
    X["DestHour"] = X["Dest"] + "_" + X["DepHour"].astype(str)
    X["CarrierDoW"] = X["UniqueCarrier"] + "_" + df["DayOfWeek"].astype(str)
    X["DepSlot15"] = (X["DepHour"].astype(str) + "_" + (X["DepMin"] // 15).astype(str))
    X["CarrierSlot15"] = X["UniqueCarrier"] + "_" + X["DepSlot15"]
    X["DistBin"] = (X["Distance"] // 250).astype(int).astype(str)
    for c in ["Month", "DayofMonth", "DayOfWeek"]:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    for c in ["UniqueCarrier"]:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def te_map(keys: pd.Series, y: np.ndarray, smooth: float = TE_SMOOTH) -> pd.Series:
    """Smoothed target mean per key level: (sum + smooth*prior) / (count + smooth)."""
    prior = y.mean()
    df = pd.DataFrame({"k": keys, "y": y})
    g = df.groupby("k")["y"].agg(["sum", "count"])
    enc = (g["sum"] + smooth * prior) / (g["count"] + smooth)
    return enc


def te_apply(keys: pd.Series, mapping: pd.Series, prior: float) -> np.ndarray:
    return keys.astype("object").map(mapping).fillna(prior).astype(float).to_numpy()


# --- fit target encodings on TRAINING DATA ONLY ------------------------------
train_base = base_frame(train)
prior = float(y_train.mean())

# out-of-fold encodings for the training rows (avoid self-fit leakage)
oof_te = {k: np.zeros(len(train)) for k in TE_KEYS}
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for tr_idx, va_idx in kf.split(train_base):
    y_tr = y_train[tr_idx]
    for k in TE_KEYS:
        m = te_map(train_base[k].iloc[tr_idx], y_tr)
        oof_te[k][va_idx] = te_apply(train_base[k].iloc[va_idx], m, prior)
# full-train encodings used for eval / hidden holdout inside predict_proba
full_te = {k: te_map(train_base[k], y_train) for k in TE_KEYS}
# frequency (count) encodings, fit on train only
full_cnt = {k: train_base[k].value_counts() for k in
            ["Origin", "Dest", "Route", "UniqueCarrier", "CarrierHour", "RouteHour",
             "OriginHour", "DepSlot15", "DestHour", "CarrierDoW"]}
n_train = len(train_base)


def build_X(b: pd.DataFrame, te: dict) -> pd.DataFrame:
    X = b.drop(columns=TE_KEYS + ["OriginRaw", "DestRaw", "RouteRaw", "CarrierRaw"], errors="ignore").copy()
    for k in TE_KEYS:
        X["TE_" + k] = te[k]
    for k, cnt in full_cnt.items():
        X["CNT_" + k] = b[k].astype("object").map(cnt).fillna(0).astype(float) / n_train
    return X


X_train = build_X(train_base, oof_te)
feature_cols = list(X_train.columns)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    b = base_frame(df)
    te = {k: te_apply(b[k], full_te[k], prior) for k in TE_KEYS}
    return build_X(b, te)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def make_model(seed: int, depth: int, colsample: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=1200,
        max_depth=depth,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=colsample,
        reg_lambda=5.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


MODELS = [make_model(42, 6, 0.9), make_model(7, 7, 0.8), make_model(2024, 8, 0.6)]

t0 = time.time()
mon_num = train["Month"].astype(str).str.replace("c-", "", regex=False).astype(float)
sample_w = 1.0 + 5.0 * (mon_num.to_numpy() - 1) / 11.0  # Jan=1 ... Dec=6: later months resemble 2006 more
for m in MODELS:
    m.fit(X_train, y_train, sample_weight=sample_w)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    ps = [m.predict_proba(Xp)[:, 1] for m in MODELS]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
