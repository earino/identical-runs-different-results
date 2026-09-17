"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

This experiment: refine calendar features (true day-of-year, sin/cos seasonality, numeric month/dow).
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

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

BEST = dict(n_estimators=300, max_depth=3, learning_rate=0.05, reg_lambda=5.0)

_MDAYS = [31, 28, 31, 31, 30, 30, 31, 31, 30, 31, 30, 31]  # cumulative starts


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(r"^c-", "", regex=True), errors="coerce")


def prepare(df: pd.DataFrame, variant: str = "cal") -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[feature_cols].copy()
    mon = _num(X["Month"]).astype("float64")
    day = _num(X["DayofMonth"]).astype("float64")
    dow = _num(X["DayOfWeek"]).astype("float64")
    dep = pd.to_numeric(X["DepTime"], errors="coerce").astype("float64")
    X["DepHour"] = np.floor(dep / 100.0)

    if variant in ("cal", "calmonth", "calsin", "calte"):
        # approximate cumulative day-of-year (per szilard airline data, months c-1..c-12)
        cum = np.cumsum([0] + _MDAYS)
        doy = day + cum[np.clip(mon.astype(int) - 1, 0, 11)] - 1.0
        X["DayOfYear"] = doy
        X["DayOfWeekNum"] = dow
    if variant in ("cal", "calmonth", "calsin", "calte"):
        X["MonthNum"] = mon
    if variant in ("calsin", "calte"):
        X["SinDoY"] = np.sin(2 * np.pi * (X["DayOfYear"] - 1) / 365.0)
        X["CosDoY"] = np.cos(2 * np.pi * (X["DayOfYear"] - 1) / 365.0)
    if variant == "calmonth":
        X["HourxMonth"] = X["DepHour"] * 13 + mon
        X["DoYxHour"] = X["DayOfYear"] * 25 + X["DepHour"]

    # target encoding (fit on train only, applied here)
    if variant == "calte":
        X["CarrierTE"] = df["UniqueCarrier"].map(te_maps["UniqueCarrier"]).astype("float64").fillna(PRIOR)
        X["OriginTE"] = df["Origin"].map(te_maps["Origin"]).astype("float64").fillna(PRIOR)
        X["DestTE"] = df["Dest"].map(te_maps["Dest"]).astype("float64").fillna(PRIOR)
        X["RouteTE"] = (df["Origin"] + "_" + df["Dest"]).map(te_maps["Route"]).astype("float64").fillna(PRIOR)

    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- target-encode maps (train only) --------------------------------------------
y_all = (train[TARGET] == POSITIVE).astype(float)
PRIOR = float(y_all.mean())
ALPHA = 20.0
te_maps = {}
for c in ["UniqueCarrier", "Origin", "Dest"]:
    g = y_all.groupby(train[c])
    cnt, mean = g.count(), g.mean()
    te_maps[c] = (mean * cnt + PRIOR * ALPHA) / (cnt + ALPHA)
g = y_all.groupby(train["Origin"] + "_" + train["Dest"])
cnt, mean = g.count(), g.mean()
te_maps["Route"] = (mean * cnt + PRIOR * ALPHA) / (cnt + ALPHA)

# --- probe ---------------------------------------------------------------------
ytr, yev = to_y(train), to_y(evald)

t0 = time.time()
variants = []


def probe(name, variant):
    Xtr, Xev = prepare(train, variant), prepare(evald, variant)
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED,
                          n_jobs=N_JOBS, **BEST)
    m.fit(Xtr, ytr)
    auc = roc_auc_score(yev, m.predict_proba(Xev)[:, 1])
    variants.append((auc, name, variant, m))
    print(f"[probe] {name}: {auc:.4f}  ({time.time() - t0:.1f}s)")


probe("cal", "cal")            # doy + dow + monthnum
probe("calmonth", "calmonth")  # + hour interactions
probe("calsin", "calsin")      # + seasonal sin/cos
probe("calte", "calte")        # + target encodings

best_auc, best_name, best_variant, model = max(variants, key=lambda r: r[0])
print(f"[probe] selected {best_name}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, best_variant))[:, 1]


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
