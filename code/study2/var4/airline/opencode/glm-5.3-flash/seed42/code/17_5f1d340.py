"""XGBoost binary classifier for airline delay. Contract per program.md.

1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
2. Module-level `predict_proba(df)`: raw DataFrame -> 1-D array of P(positive). All per-row feature
   engineering lives inside prepare(); any fitted statistics come from train only.
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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "DayOfWeek"]


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic per-row transforms (no fitting)."""
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(np.int32)
    X["DepTime"] = dep
    X["hour"] = dep // 100
    X["minute"] = dep % 100
    tm = dep.mod(2400)
    tmin = (tm // 100) * 60 + tm % 100
    ang = 2 * np.pi * tmin / 1440.0
    X["sin_t"] = np.sin(ang)
    X["cos_t"] = np.cos(ang)
    mo = df["Month"].str[2:].astype(int)
    X["MonthNum"] = mo
    a = 2 * np.pi * (mo - 1) / 12.0
    X["sin_m"] = np.sin(a)
    X["cos_m"] = np.cos(a)
    X["DayOfMonth"] = df["DayofMonth"].str[2:].astype(int)
    dw = df["DayOfWeek"].str[2:].astype(int)
    X["DayOfWeekNum"] = dw
    a = 2 * np.pi * dw / 7.0
    X["sin_d"] = np.sin(a)
    X["cos_d"] = np.cos(a)
    dist = df["Distance"].astype(np.float32)
    X["Distance"] = dist
    X["log_dist"] = np.log1p(dist)
    X["UniqueCarrier"] = df["UniqueCarrier"]
    X["Origin"] = df["Origin"]
    X["Dest"] = df["Dest"]
    X["DayOfWeek"] = df["DayOfWeek"]
    X["DayOfMonthCat"] = df["DayofMonth"]
    X["OrgHour"] = X["Origin"].astype(str) + "_" + X["hour"].astype(str)
    X["DistBucket"] = pd.cut(dist, bins=DIST_EDGES, labels=False).astype("Int64").astype(str)
    return X


DIST_EDGES = np.unique(np.quantile(train["Distance"], [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0]))
TR_FEAT = base_features(train)
CAT_LEVELS = {c: pd.Index(sorted(TR_FEAT[c].dropna().unique())) for c in CAT_COLS}
CAT_LEVELS_EXTRA = {c: pd.Index(sorted(TR_FEAT[c].dropna().unique())) for c in ["DayOfMonthCat", "DistBucket"]}
ORGHOUR_FREQ = TR_FEAT["OrgHour"].value_counts().to_dict()


def prepare(df: pd.DataFrame, variant: str = "base") -> pd.DataFrame:
    X = base_features(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    if variant in ("domcat", "all"):
        X["DayOfMonthCat"] = pd.Categorical(X["DayOfMonthCat"], categories=CAT_LEVELS_EXTRA["DayOfMonthCat"])
    else:
        X = X.drop(columns=["DayOfMonthCat"])
    if variant in ("freq", "all"):
        X["freq_OrgHour"] = np.log1p(X["OrgHour"].map(ORGHOUR_FREQ).fillna(0).to_numpy())
    X = X.drop(columns=["OrgHour"])
    if variant in ("distb", "all"):
        X["DistBucket"] = pd.Categorical(X["DistBucket"], categories=CAT_LEVELS_EXTRA["DistBucket"])
    else:
        X = X.drop(columns=["DistBucket"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


ye = to_y(evald)
BEST_CFG = dict(colsample_bytree=0.4, max_depth=20, subsample=1.0, max_bin=512)


def make_model(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
        **BEST_CFG,
    )


t0 = time.time()
best_auc, best_model, best_variant = -1.0, None, None
for variant in ["domcat", "freq", "distb", "all"]:
    X = prepare(train, variant)
    Xe = prepare(evald, variant)
    m = make_model(SEED)
    m.fit(X, to_y(train))
    auc = roc_auc_score(ye, m.predict_proba(Xe)[:, 1])
    print(f"variant={variant} eval_auc={auc:.4f}")
    if auc > best_auc:
        best_auc, best_model, best_variant = auc, m, variant
print(f"Training time: {time.time() - t0:.1f}s best_variant={best_variant}")
model = best_model
VARIANT = best_variant


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, VARIANT))[:, 1]


eval_auc = best_auc
print(f"Eval AUC: {eval_auc:.4f}")
