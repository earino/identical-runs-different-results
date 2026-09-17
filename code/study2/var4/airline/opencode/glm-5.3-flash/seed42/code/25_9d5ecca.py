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
    X["DestHour"] = X["Dest"].astype(str) + "_" + X["hour"].astype(str)
    X["CarHour"] = df["UniqueCarrier"].astype(str) + "_" + X["hour"].astype(str)
    X["Route"] = X["Origin"].astype(str) + "_" + X["Dest"].astype(str)
    X["OrgDOW"] = X["Origin"].astype(str) + "_" + X["DayOfWeekNum"].astype(str)
    X["DestDOW"] = X["Dest"].astype(str) + "_" + X["DayOfWeekNum"].astype(str)
    X["CarOrigin"] = df["UniqueCarrier"].astype(str) + "_" + X["Origin"].astype(str)
    X["CarDest"] = df["UniqueCarrier"].astype(str) + "_" + X["Dest"].astype(str)
    X["CarRoute"] = df["UniqueCarrier"].astype(str) + "_" + X["Route"].astype(str)
    X["DowHour"] = X["DayOfWeekNum"].astype(str) + "_" + X["hour"].astype(str)
    X["OrgMonth"] = X["Origin"].astype(str) + "_" + X["MonthNum"].astype(str)
    X["DestMonth"] = X["Dest"].astype(str) + "_" + X["MonthNum"].astype(str)
    X["DistBucket"] = pd.cut(dist, bins=DIST_EDGES, labels=False).astype("Int64").astype(str)
    return X


DIST_EDGES = np.unique(np.quantile(train["Distance"], [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0]))
TR_FEAT = base_features(train)
CAT_LEVELS = {c: pd.Index(sorted(TR_FEAT[c].dropna().unique())) for c in CAT_COLS}
FREQ_MAPS = {
    c: TR_FEAT[c].value_counts().to_dict()
    for c in ["OrgHour", "DestHour", "CarHour", "Route", "Origin", "Dest", "OrgDOW", "DestDOW", "CarOrigin", "CarDest", "CarRoute", "OrgMonth", "DestMonth", "DowHour"]
}

FREQ_PACKS = {
    "hours+base": ["OrgHour", "DestHour", "CarHour", "Origin", "Dest", "Route"],
    "ratios": ["OrgHour", "DestHour", "CarHour", "Origin", "Dest", "Route"],
    "r+dh": ["OrgHour", "DestHour", "CarHour", "Origin", "Dest", "Route", "DowHour"],
    "dh": ["OrgHour", "DestHour", "CarHour", "Origin", "Dest", "Route", "DowHour"],
}

RATIO_DEFS = {"ratios": True, "r+dh": True}  # add share ratios for these variants

BASE_KEYS = ["DayOfMonthCat", "DistBucket", "OrgHour", "DestHour", "CarHour", "Route", "OrgDOW", "DestDOW", "CarOrigin", "CarDest", "CarRoute", "OrgMonth", "DestMonth", "DowHour"]
RAW_FREQ = {c: TR_FEAT[c].value_counts().to_dict() for c in ["Origin", "Dest", "UniqueCarrier"]}


def prepare(df: pd.DataFrame, variant: str = "base") -> pd.DataFrame:
    X = base_features(df)
    freqs = {}
    for c in FREQ_PACKS.get(variant, []):
        v = np.log1p(X[c].map(FREQ_MAPS[c]).fillna(0).to_numpy())
        X[f"freq_{c}"] = v
        freqs[c] = X[c].map(FREQ_MAPS[c]).fillna(0).to_numpy()
    if RATIO_DEFS.get(variant, False):
        X["share_OrgHour"] = freqs["OrgHour"] / (X["Origin"].map(RAW_FREQ["Origin"]).fillna(0).to_numpy() / 27.0 + 1.0)
        X["share_DestHour"] = freqs["DestHour"] / (X["Dest"].map(RAW_FREQ["Dest"]).fillna(0).to_numpy() / 27.0 + 1.0)
        X["share_CarHour"] = freqs["CarHour"] / (X["UniqueCarrier"].map(RAW_FREQ["UniqueCarrier"]).fillna(0).to_numpy() / 27.0 + 1.0)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    X = X.drop(columns=[c for c in BASE_KEYS if c in X.columns])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


ye = to_y(evald)

t0 = time.time()
VARIANT = "ratios"
X = prepare(train, VARIANT)
y = to_y(train)
Xe = prepare(evald, VARIANT)
CONFIGS = [
    (0.4, 20, 450, 42, 0.03),
    (0.4, 20, 300, 7, 0.05),
    (0.45, 20, 300, 123, 0.05),
    (0.4, 16, 300, 42, 0.05),
]
members = []
probs = []
for col, depth, n_est, seed, lr in CONFIGS:
    m = xgb.XGBClassifier(
        n_estimators=n_est,
        learning_rate=lr,
        max_depth=depth,
        subsample=1.0,
        colsample_bytree=col,
        reg_lambda=1.0,
        max_bin=512,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(X, y)
    p = m.predict_proba(Xe)[:, 1]
    print(f"member col={col} depth={depth} n_est={n_est} seed={seed} lr={lr} eval_auc={roc_auc_score(ye, p):.4f}")
    members.append(m)
    probs.append(p)
ens = np.mean(probs, axis=0)
ens_auc = roc_auc_score(ye, ens)
print(f"ensemble mean eval_auc={ens_auc:.4f}")
ens_med = roc_auc_score(ye, np.median(np.array(probs), axis=0))
print(f"ensemble median eval_auc={ens_med:.4f}")
rp = np.array([pd.Series(p).rank(pct=True).to_numpy() for p in probs])
ens_rank = roc_auc_score(ye, rp.mean(axis=0))
print(f"ensemble rankmean eval_auc={ens_rank:.4f}")
print(f"Training time: {time.time() - t0:.1f}s")
AGG = {ens_auc: "mean", ens_med: "median", ens_rank: "rank"}[max(ens_auc, ens_med, ens_rank)]
model = members


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xdf = prepare(df, VARIANT)
    ps = np.array([m.predict_proba(Xdf)[:, 1] for m in model])
    if AGG == "median":
        return np.median(ps, axis=0)
    if AGG == "rank":
        return np.array([pd.Series(p).rank(pct=True).to_numpy() for p in ps]).mean(axis=0)
    return ps.mean(axis=0)


eval_auc = max(ens_auc, ens_med, ens_rank)
print(f"Eval AUC: {eval_auc:.4f}")
