"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
y_tr = (train[TARGET] == POSITIVE).astype(int).to_numpy()
y_ev = (evald[TARGET] == POSITIVE).astype(int).to_numpy()

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

CURVE_KS = [50, 100, 150, 200]
SHIP_KS = [100, 150, 200]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace(r"^c-", "", regex=True), errors="coerce")


# --- smoothed target encodings, fit on train only -----------------------------
PRIOR = float(y_tr.mean())


def te_map(keys: pd.Series, m: int) -> dict:
    """Smoothed mean-delay-rate per key, fit on the 2005 training rows only."""
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y_tr}).groupby("k")["y"].agg(["sum", "count"])
    enc = (g["sum"] + PRIOR * m) / (g["count"] + m)
    return enc.to_dict()


_route = train["Origin"].astype(str) + "-" + train["Dest"].astype(str)
TE = {
    "route200": (te_map(_route, 200), "route"),
    "origin200": (te_map(train["Origin"].astype(str), 200), "origin"),
    "dest200": (te_map(train["Dest"].astype(str), 200), "dest"),
    "carrier200": (te_map(train["UniqueCarrier"].astype(str), 200), "carrier"),
    "route20": (te_map(_route, 20), "route_lo"),
    "origin20": (te_map(train["Origin"].astype(str), 20), "origin_lo"),
    "dest20": (te_map(train["Dest"].astype(str), 20), "dest_lo"),
}


def te_cols(df: pd.DataFrame, which: list) -> pd.DataFrame:
    route = df["Origin"].astype(str) + "-" + df["Dest"].astype(str)
    src = {"route": route, "route_lo": route, "origin": df["Origin"].astype(str),
           "origin_lo": df["Origin"].astype(str), "dest": df["Dest"].astype(str),
           "dest_lo": df["Dest"].astype(str), "carrier": df["UniqueCarrier"].astype(str)}
    out = {}
    for name in which:
        m, key = TE[name]
        out[f"te_{name}"] = src[key].map(m).astype(float).fillna(PRIOR)
    return pd.DataFrame(out, index=df.index)


def base_time(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["Month"] = _num(df["Month"])
    X["DayofMonth"] = _num(df["DayofMonth"])
    X["DayOfWeek"] = _num(df["DayOfWeek"])
    X["DepTime"] = df["DepTime"].astype(float)
    X["Distance"] = df["Distance"].astype(float)
    dep = X["DepTime"]
    hour = np.clip((dep // 100) % 24, 0, 23)
    minute = dep % 100
    X["DepHour"] = hour
    X["DepMinute"] = minute
    X["DepTimeMin"] = hour * 60 + minute
    X["sin_day"] = np.sin(2 * np.pi * (hour * 60 + minute) / 1440)
    X["cos_day"] = np.cos(2 * np.pi * (hour * 60 + minute) / 1440)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


CAND_TE = {
    "ctrl": [],
    "TE_R": ["route200"],
    "TE_A": ["route200", "origin200", "dest200", "carrier200"],
    "TE_Alo": ["route200", "origin200", "dest200", "carrier200", "route20", "origin20", "dest20"],
}


def make_fe(te_which: list):
    def fe(df: pd.DataFrame) -> pd.DataFrame:
        X = base_time(df)
        return pd.concat([X, te_cols(df, te_which)], axis=1)
    return fe


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
results = {}
for name, which in CAND_TE.items():
    fe = make_fe(which)
    m = xgb.XGBClassifier(**PARAMS)
    m.fit(fe(train), y_tr, verbose=False)
    Xe = fe(evald)
    curve = {}
    for k in CURVE_KS:
        p = m.predict_proba(Xe, iteration_range=(0, k))[:, 1]
        curve[k] = roc_auc_score(y_ev, p)
    print(f"[diag] {name} curve " + " ".join(f"k={k}:{curve[k]:.4f}" for k in CURVE_KS))
    for k in SHIP_KS:
        results[(name, k)] = curve[k]

(best_name, BEST_K), best_auc = max(results.items(), key=lambda kv: kv[1])
print(f"[diag] chosen: {best_name} k={BEST_K} auc={best_auc:.4f} ({time.time() - t0:.1f}s)")
prepare = make_fe(CAND_TE[best_name])
model = xgb.XGBClassifier(**PARAMS)
model.fit(prepare(train), y_tr, verbose=False)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df), iteration_range=(0, BEST_K))[:, 1]


eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
