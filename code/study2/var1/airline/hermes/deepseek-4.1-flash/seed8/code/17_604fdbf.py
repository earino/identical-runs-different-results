"""XGBoost ensemble for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Model: averaged ensemble of XGBoost classifiers (tree depths 1-16, learning rate 0.16).  train is
2005 and eval/holdout are 2006, so any single capacity/step-size setting is a year-transfer gamble;
averaging over a wide capacity range is the robust choice (see FINAL.md).

This run also compares the base feature set against an extended one that adds carrier-airport
structure counts, and keeps whichever scores better on eval.
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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


ytr, yev = to_y(train), to_y(evald)

# --- features -----------------------------------------------------------------
# Calendar columns (Month/DayofMonth/DayOfWeek) are dropped: ablation showed they contribute nothing on
# the year-shifted eval set.  Airport identity + scheduled time + carrier carry the signal.
BASE = ["DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
cat_cols = [c for c in BASE if not pd.api.types.is_numeric_dtype(train[c])]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
freq_tables = {c: train[c].value_counts() for c in ("Origin", "Dest", "UniqueCarrier")}
pair_co = train.groupby(["UniqueCarrier", "Origin"]).size()
pair_cd = train.groupby(["UniqueCarrier", "Dest"]).size()


def _pair(df, a, b):
    return df[a].astype(str) + "_" + df[b].astype(str)


def make_prepare(extended: bool):
    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        # ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows.
        X = df[BASE].copy()
        for c in cat_cols:
            X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
        X["freq_origin"] = np.log1p(df["Origin"].map(freq_tables["Origin"]).fillna(0).to_numpy())
        X["freq_dest"] = np.log1p(df["Dest"].map(freq_tables["Dest"]).fillna(0).to_numpy())
        X["freq_carrier"] = np.log1p(df["UniqueCarrier"].map(freq_tables["UniqueCarrier"]).fillna(0).to_numpy())
        if extended:
            co, cd = _pair(df, "UniqueCarrier", "Origin"), _pair(df, "UniqueCarrier", "Dest")
            nco = co.map(pair_co).fillna(0).to_numpy()
            ncd = cd.map(pair_cd).fillna(0).to_numpy()
            X["carrier_origin_freq"] = np.log1p(nco)
            X["carrier_dest_freq"] = np.log1p(ncd)
            X["carrier_origin_share"] = nco / (df["Origin"].map(freq_tables["Origin"]).fillna(0).to_numpy() + 1.0)
            X["carrier_dest_share"] = ncd / (df["Dest"].map(freq_tables["Dest"]).fillna(0).to_numpy() + 1.0)
        return X

    return prepare


GRID = [(depth, 0.16) for depth in range(1, 17)]


def fit_ensemble(prep):
    models = []
    for depth, lr in GRID:
        m = xgb.XGBClassifier(
            max_depth=depth, learning_rate=lr, n_estimators=300,
            min_child_weight=10, tree_method="hist", enable_categorical=True, n_jobs=N_JOBS,
        )
        m.fit(prep(train), ytr)
        models.append(m)
    return models


t0 = time.time()
arms = {}
for name, ext in (("base", False), ("extended", True)):
    prep = make_prepare(ext)
    models = fit_ensemble(prep)
    Xev = prep(evald)
    auc = roc_auc_score(yev, np.mean([m.predict_proba(Xev)[:, 1] for m in models], axis=0))
    arms[name] = (auc, prep, models)
    print(f"  {name:9s} Eval AUC {auc:.4f}  [{time.time() - t0:.0f}s]", flush=True)
print(f"Training time: {time.time() - t0:.1f}s")

win = max(arms, key=lambda k: arms[k][0])
_, prep, models = arms[win]
print(f"SELECTED {win}")

Eval_AUC = arms[win][0]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prep(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


print(f"Eval AUC: {Eval_AUC:.4f}")
