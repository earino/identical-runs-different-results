"""XGBoost binary classifier for airline delays. THE ONLY FILE THE AGENT EDITS.

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


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


y_train = to_y(train)
PRIOR = float(y_train.mean())
SMOOTH = 30.0

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "hour_c", "slot_c"]
TE_KEYS = ["Month", "DayofMonth", "DayOfWeek"]


def base_features(df: pd.DataFrame, slotdiv: int = 15) -> pd.DataFrame:
    """Per-row derived features. No fitted statistics in here."""
    out = pd.DataFrame(index=df.index)
    out["DepTime"] = pd.to_numeric(df["DepTime"], errors="coerce")
    dt = out["DepTime"]
    hour = dt // 100
    out["hour_c"] = (hour % 24).astype("Int64").astype(str)
    out["dep_min"] = hour * 60 + dt % 100
    out["slot_c"] = (out["dep_min"] // slotdiv).astype("Int64").astype(str)
    out["dep_sin"] = np.sin(2 * np.pi * out["dep_min"] / 1440.0)
    out["dep_cos"] = np.cos(2 * np.pi * out["dep_min"] / 1440.0)
    out["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    out["log_distance"] = np.log1p(out["Distance"])
    out["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    out["Origin"] = df["Origin"].astype(str)
    out["Dest"] = df["Dest"].astype(str)
    out["Month"] = df["Month"].astype(str)
    out["DayofMonth"] = df["DayofMonth"].astype(str)
    out["DayOfWeek"] = df["DayOfWeek"].astype(str)
    return out


feat_train = base_features(train)
SLOTDIVS = (15, 10, 20, 12, 25)
cat_levels = {c: pd.Index(sorted(feat_train[c].unique())) for c in CAT_COLS if c != "slot_c"}
slot_levels = {
    sd: pd.Index(sorted((feat_train["dep_min"] // sd).astype(str).unique())) for sd in SLOTDIVS
}

_month_num = pd.to_numeric(feat_train["Month"].str.replace("c-", "", regex=False))
sample_w = (1.0 + 0.5 * (_month_num - 1) / 11.0).to_numpy()

te_maps = {}
for key in TE_KEYS:
    g = pd.DataFrame({"k": feat_train[key], "y": y_train}).groupby("k")["y"].agg(["size", "sum"])
    te_maps[key] = ((g["sum"] + SMOOTH * PRIOR) / (g["size"] + SMOOTH)).to_dict()

NUMERIC = ["DepTime", "dep_min", "dep_sin", "dep_cos", "Distance", "log_distance"]


def prepare(df: pd.DataFrame, f: pd.DataFrame | None = None) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    if f is None:
        f = base_features(df, 15)
    X = pd.DataFrame(index=df.index)
    for c in NUMERIC:
        X[c] = f[c]
    for c in CAT_COLS:
        if c == "slot_c":
            continue
        X[c] = pd.Categorical(f[c], categories=cat_levels[c])  # unseen levels -> NaN
    for key in TE_KEYS:
        X["te_" + key] = f[key].map(te_maps[key]).astype(float).fillna(PRIOR)
    return X


def with_slot(X: pd.DataFrame, df_index: pd.Index, f: pd.DataFrame, sd: int) -> pd.DataFrame:
    """Copy of X with the slot_c column re-cut at `sd` minutes."""
    Xs = X.copy()
    Xs["slot_c"] = pd.Categorical(
        (f["dep_min"] // sd).astype(str), categories=slot_levels[sd]
    )
    return Xs


def fit_one(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.033,
        max_depth=9,
        min_child_weight=50,
        gamma=5.0,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


SEEDS = [42, 7, 2026]

t0 = time.time()
X_base = prepare(train, feat_train)
X_by_sd = {sd: with_slot(X_base, train.index, feat_train, sd) for sd in SLOTDIVS}
print("features built", flush=True)
models = []  # (model, slotdiv)
for sd in SLOTDIVS:
    for seed in SEEDS:
        m = fit_one(seed)
        m.fit(X_by_sd[sd], y_train, sample_weight=sample_w)
        models.append((m, sd))
    print(f"sd={sd} done {time.time() - t0:.0f}s", flush=True)
print(f"models={len(models)}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    f = base_features(df, 15)
    X_base = prepare(df, f)
    Ps = [m.predict_proba(with_slot(X_base, df.index, f, sd))[:, 1] for m, sd in models]
    return np.mean(Ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
