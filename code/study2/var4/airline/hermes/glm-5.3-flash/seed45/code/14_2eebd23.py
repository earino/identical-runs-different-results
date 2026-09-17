"""XGBoost binary classifier for the airline delay task (the only agent-edited file).

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
from sklearn.model_selection import StratifiedKFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

NAT_CAT = ["UniqueCarrier", "Origin", "Dest", "Route"]
TE_KEYS = [
    "UniqueCarrier", "Origin", "Dest", "Route", "dep_h",
    "Dest|dep_h", "UniqueCarrier|dep_h", "Origin|dep_h", "Route|dep_h",
]
SMOOTH = 100
NUM_COLS = ["month", "dom", "dow", "dep_h", "dep_m", "dep_time", "Distance",
            "mo_sin", "mo_cos", "dow_sin", "dow_cos", "dt_sin", "dt_cos", "log_dist",
            "freq_UniqueCarrier", "freq_Origin", "freq_Dest", "freq_Route"]


def base_parts(df: pd.DataFrame) -> pd.DataFrame:
    """All raw engineered pieces; TE key columns get built here too."""
    p = pd.DataFrame(index=df.index)
    p["month"] = df["Month"].str[2:].astype(int)
    p["dom"] = df["DayofMonth"].str[2:].astype(int)
    p["dow"] = df["DayOfWeek"].str[2:].astype(int)
    dt = df["DepTime"].fillna(0).astype(int)
    p["dep_h"] = (dt // 100) % 24
    p["dep_m"] = dt % 100
    p["dep_time"] = p["dep_h"] * 60 + p["dep_m"]
    p["Distance"] = df["Distance"].astype(float)
    p["log_dist"] = np.log1p(p["Distance"])
    p["mo_sin"] = np.sin(2 * np.pi * p["month"] / 12)
    p["mo_cos"] = np.cos(2 * np.pi * p["month"] / 12)
    p["dow_sin"] = np.sin(2 * np.pi * p["dow"] / 7)
    p["dow_cos"] = np.cos(2 * np.pi * p["dow"] / 7)
    p["dt_sin"] = np.sin(2 * np.pi * p["dep_time"] / 1440)
    p["dt_cos"] = np.cos(2 * np.pi * p["dep_time"] / 1440)
    p["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    p["Origin"] = df["Origin"].astype(str)
    p["Dest"] = df["Dest"].astype(str)
    p["Route"] = p["Origin"] + "_" + p["Dest"]
    for c in ["UniqueCarrier", "Origin", "Dest", "Route"]:
        p["freq_" + c] = np.log1p(p[c].map(FREQ[c]).fillna(0).astype(float))
    for key in TE_KEYS:  # composite TE keys, e.g. Dest|dep_h
        parts = key.split("|")
        if len(parts) > 1:
            p[key] = p[parts[0]].str.cat([p[x].astype(str) for x in parts[1:]], sep="|")
    return p


def prepare(df: pd.DataFrame, te_matrix: np.ndarray = None) -> pd.DataFrame:
    """ALL feature engineering lives here. te_matrix (train rows only) supplies
    out-of-fold target encodings; scoring rows get the train-fitted maps."""
    p = base_parts(df)
    X = pd.DataFrame(index=p.index)
    for c in NUM_COLS:
        X[c] = p[c]
    for c in NAT_CAT:
        X[c] = pd.Categorical(p[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    for j, key in enumerate(TE_KEYS):
        name = "te_" + key.replace("|", "_")
        if te_matrix is not None:
            X[name] = te_matrix[:, j]
        else:
            X[name] = p[key].map(TE_MAP[key]).fillna(PRIOR).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- fit statistics on TRAIN only ----------------------------------------------
y = to_y(train)
_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
FREQ = {c: train[c].value_counts() for c in ["UniqueCarrier", "Origin", "Dest"]}
FREQ["Route"] = _route.value_counts()
PRIOR = float(y.mean())
p_tr = base_parts(train)
CAT_LEVELS = {c: pd.Index(sorted(p_tr[c].dropna().unique())) for c in NAT_CAT}

TE_MAP = {}
for key in TE_KEYS:
    g = pd.Series(y, index=p_tr.index).groupby(p_tr[key]).agg(["sum", "count"])
    TE_MAP[key] = ((g["sum"] + SMOOTH * PRIOR) / (g["count"] + SMOOTH)).astype(float)

# out-of-fold target encodings for the training rows (avoid self-leakage)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros((len(train), len(TE_KEYS)))
for itr, iva in skf.split(train, y):
    for j, key in enumerate(TE_KEYS):
        g = pd.Series(y[itr], index=p_tr.index[itr]).groupby(p_tr[key].iloc[itr]).agg(["sum", "count"])
        m = ((g["sum"] + SMOOTH * PRIOR) / (g["count"] + SMOOTH)).astype(float)
        oof[iva, j] = p_tr[key].iloc[iva].map(m).fillna(PRIOR).astype(float).to_numpy()

# --- model ----------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=5000,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.2,
    min_child_weight=2,
    reg_lambda=384.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=150,
)

t0 = time.time()
X_train = prepare(train, te_matrix=oof)
X_eval = prepare(evald)
print(f"Feature time: {time.time() - t0:.1f}s")

t0 = time.time()
model.fit(X_train, y, eval_set=[(X_eval, to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
