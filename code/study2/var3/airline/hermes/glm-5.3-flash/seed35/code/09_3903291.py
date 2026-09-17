"""XGBoost binary classifier for flight delays. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Features (screened): Month/DayofMonth/DayOfWeek/UniqueCarrier/Origin/Dest as categoricals with train-fixed
appearance-order levels; DepTime -> dep_min + sin/cos (cyclic); carrier x 30-min-block categorical.
Variants: +Distance / +minute-of-hour / +carrier x month / -Dest (Distance hurts in the base set but adds
diversity as a member variant). Model: 11-member XGBClassifier ensemble + 1 quantile-regression booster,
probability average. Diversity beats single-model tuning on this 2005->2006 time shift.
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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

cat_levels = {c: pd.Index(train[c].dropna().unique()) for c in CAT_COLS}  # appearance order: screened best
_mtr = (train["DepTime"] // 100) * 60 + (train["DepTime"] % 100)


def prepare(df: pd.DataFrame, variant: str = "base") -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[CAT_COLS].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    m = (df["DepTime"] // 100) * 60 + (df["DepTime"] % 100)
    X["dep_min"] = m.astype(float)
    X["dep_sin"] = np.sin(2 * np.pi * m / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * m / 1440.0)
    # carrier x 30-min block
    blk = (m // 30).astype(int)
    blk_train = (_mtr // 30).astype(int)
    X["car_blk"] = pd.Categorical(
        df["UniqueCarrier"] + "_" + blk.astype(str),
        categories=pd.Index((train["UniqueCarrier"] + "_" + blk_train.astype(str)).unique()),
    )
    if "dist" in variant:
        X["Distance"] = df["Distance"].astype(float)
    if "minute" in variant:
        X["dep_mm"] = (m % 60).astype(float)  # minute-of-hour: scheduling/turnaround signature
    if "carmon" in variant:
        X["car_mon"] = pd.Categorical(
            df["UniqueCarrier"] + "_" + df["Month"].astype(str),
            categories=pd.Index((train["UniqueCarrier"] + "_" + train["Month"].astype(str)).unique()),
        )
    if "nodest" in variant:
        X = X.drop(columns=["Dest"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: diverse ensemble ---------------------------------------------------
def make_model(**kw):
    p = dict(n_estimators=1200, learning_rate=0.03, max_depth=6, tree_method="hist",
             enable_categorical=True, random_state=SEED, n_jobs=N_JOBS)
    p.update(kw)
    return xgb.XGBClassifier(**p)


# (model kwargs, feature variant, seed)
MEMBERS = [
    ({}, "dist", 42),                                        # d6 + Distance
    ({"max_depth": 7}, "dist", 42),                          # d7 + Distance
    ({"subsample": 0.8}, "base", 7),                         # base bag
    ({"subsample": 0.8}, "dist", 99),                        # +dist bag
    ({"subsample": 0.8}, "base", 13),                        # base bag
    ({"subsample": 0.8}, "dist", 113),                       # +dist bag
    ({}, "carmon+dist", 211),                                # carrier x month + Distance
    ({"objective": "reg:pseudohubererror"}, "base", 81),     # huber member
    ({}, "nodest", 101),                                     # no-Dest member
    ({}, "minute", 441),                                     # + minute-of-hour member
    ({}, "minute+dist", 451),                                # + minute-of-hour + Distance
]

t0 = time.time()
y = to_y(train)
models = []
for kw, variant, seed in MEMBERS:
    kw = dict(kw)
    kw["random_state"] = seed
    m = make_model(**kw)
    m.fit(prepare(train, variant), y)
    models.append((m, variant))

# quantile-regression member (Booster API; median regression as a differently-biased classifier)
dtrain = xgb.DMatrix(prepare(train), label=y, enable_categorical=True)
qparams = {"objective": "reg:quantileerror", "quantile_alpha": 0.5, "max_depth": 6,
           "eta": 0.03, "seed": 521, "tree_method": "hist"}
qmodel = xgb.train(qparams, dtrain, num_boost_round=1200)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    ps = [m.predict_proba(prepare(df, variant))[:, 1] for m, variant in models]
    ps.append(np.clip(qmodel.predict(xgb.DMatrix(prepare(df), enable_categorical=True)), 0.0, 1.0))
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
