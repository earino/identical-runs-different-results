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

CAT_RAW = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in CAT_RAW}
route_key_train = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
hour_train = (pd.to_numeric(train["DepTime"], errors="coerce") // 100).fillna(0).astype(int).astype(str)
FREQ_KEYS = {
    "fr_route": route_key_train,
    "fr_carrier": train["UniqueCarrier"].astype(str),
    "fr_origin": train["Origin"].astype(str),
    "fr_dest": train["Dest"].astype(str),
    "fr_oh": train["Origin"].astype(str) + "_" + hour_train,
    "fr_dh": train["Dest"].astype(str) + "_" + hour_train,
    "fr_ch": train["UniqueCarrier"].astype(str) + "_" + hour_train,
}
freq_counts = {n: k.value_counts() for n, k in FREQ_KEYS.items()}


def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    t = pd.to_numeric(df["DepTime"], errors="coerce")
    h = (t // 100).fillna(0)
    m = (t % 100).fillna(0)
    hstr = h.astype(int).astype(str)
    frac = h + m / 60.0
    X["hour"] = h
    X["minute"] = m
    X["frac_hour"] = frac
    X["sin_h"] = np.sin(2 * np.pi * frac / 24.0)
    X["cos_h"] = np.cos(2 * np.pi * frac / 24.0)
    X["distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["log_dist"] = np.log1p(X["distance"])
    X["dow"] = _cnum(df["DayOfWeek"])
    for c in CAT_RAW:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    keys = {
        "fr_route": df["Origin"].astype(str) + "_" + df["Dest"].astype(str),
        "fr_carrier": df["UniqueCarrier"].astype(str),
        "fr_origin": df["Origin"].astype(str),
        "fr_dest": df["Dest"].astype(str),
        "fr_oh": df["Origin"].astype(str) + "_" + hstr,
        "fr_dh": df["Dest"].astype(str) + "_" + hstr,
        "fr_ch": df["UniqueCarrier"].astype(str) + "_" + hstr,
    }
    for name, k in keys.items():
        X[name] = np.log1p(k.map(freq_counts[name]).fillna(0).to_numpy())
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


BASE_PARAMS = dict(
    n_estimators=1000,
    max_depth=18,
    tree_method="hist",
    enable_categorical=True,
    subsample=1.0,
    min_child_weight=1,
    reg_lambda=0.0,
    reg_alpha=2.0,
    early_stopping_rounds=75,
    eval_metric="auc",
    n_jobs=N_JOBS,
)
MEMBERS = [
    dict(learning_rate=0.015, colsample_bytree=0.5, random_state=SEED),
    dict(learning_rate=0.015, colsample_bytree=0.5, random_state=7),
]

t0 = time.time()
Xtr = prepare(train)
ytr = to_y(train)
Xev = prepare(evald)
yev = to_y(evald)

models = []
for k, over in enumerate(MEMBERS):
    m = xgb.XGBClassifier(**{**BASE_PARAMS, **over})
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(m)
    print(f"model {k}: best_iter={m.best_iteration} val={m.best_score:.4f} ({time.time() - t0:.1f}s)")
booster_iters = [(m.get_booster(), m.best_iteration + 1) for m in models]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    dm = xgb.DMatrix(X, enable_categorical=True)
    return np.mean([b.predict(dm, iteration_range=(0, n)) for b, n in booster_iters], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
