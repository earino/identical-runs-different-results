"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

All feature engineering lives inside `prepare()` (which predict_proba calls on unseen rows); every statistic it
uses is fitted once on the training frame and stored in module-level artifacts.
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


# --- feature engineering -------------------------------------------------------
CAT_SRC = ["UniqueCarrier", "Origin", "Dest"]

# smoothed target-encoding specs: (key columns, folding strength m)
TE_SPECS = {
    "te_carrier": (("UniqueCarrier",), 20.0),
    "te_origin": (("Origin",), 20.0),
    "te_dest": (("Dest",), 20.0),
    "te_origin_hour": (("Origin", "Hour"), 50.0),
    "te_carrier_hour": (("UniqueCarrier", "Hour"), 50.0),
}

FEATS = [
    "MonthNum", "DayofMonthNum", "DayOfWeekNum",
    "MonthSin", "MonthCos", "DoWSin", "DoWCos",
    "Hour", "Minute", "HourFrac", "IsRedEye",
    "Distance", "LogDistance",
    "UniqueCarrier", "Origin", "Dest",
] + list(TE_SPECS)

MINUTE_PER_DAY = 1440.0
MONTH_ANG = 2 * np.pi / 12.0
DOW_ANG = 2 * np.pi / 7.0


def _strip_c(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    """Raw string keys used by the target encodings and by the categorical features."""
    K = pd.DataFrame(index=df.index)
    K["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    K["Origin"] = df["Origin"].astype(str)
    K["Dest"] = df["Dest"].astype(str)
    K["Hour"] = (pd.to_numeric(df["DepTime"], errors="coerce") // 100.0).clip(0, 24).astype("int64")
    return K


y_train = to_y(train)
_prior = float(y_train.mean())
_K_train = _keys(train)
_cat_levels = {c: pd.Index(sorted(_K_train[c].dropna().unique())) for c in CAT_SRC}


def _te_key(K: pd.DataFrame, spec) -> pd.Series:
    if len(spec) == 1:
        return K[spec[0]]
    return K[list(spec)].astype(str).agg("|".join, axis=1)


def _te_map(K: pd.DataFrame, y: np.ndarray, spec, m: float) -> pd.Series:
    key = _te_key(K, spec)
    g = pd.DataFrame({"k": key.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + _prior * m) / (g["count"] + m)


# full-train maps (used for unseen frames) and out-of-fold values (used to train the model)
_te_maps = {name: _te_map(_K_train, y_train, spec, m) for name, (spec, m) in TE_SPECS.items()}
_te_oof = pd.DataFrame(index=train.index)
_rng = np.random.RandomState(SEED)
_folds = _rng.randint(0, 5, size=len(train))
for name, (spec, m) in TE_SPECS.items():
    key_all = _te_key(_K_train, spec)
    vals = np.full(len(train), _prior)
    for f in range(5):
        te = _folds == f
        mp = _te_map(_K_train[~te], y_train[~te], spec, m)
        vals[te] = key_all[te].map(mp).fillna(_prior).to_numpy()
    _te_oof[name] = vals


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    month, day, dow = _strip_c(df["Month"]), _strip_c(df["DayofMonth"]), _strip_c(df["DayOfWeek"])
    X["MonthNum"] = month
    X["DayofMonthNum"] = day
    X["DayOfWeekNum"] = dow
    X["MonthSin"] = np.sin(month * MONTH_ANG)
    X["MonthCos"] = np.cos(month * MONTH_ANG)
    X["DoWSin"] = np.sin(dow * DOW_ANG)
    X["DoWCos"] = np.cos(dow * DOW_ANG)

    dept = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dept // 100.0).clip(0, 24)
    minute = (dept % 100.0).clip(0, 59)
    hf = hour + minute / 60.0
    X["Hour"] = hour
    X["Minute"] = minute
    X["HourFrac"] = hf
    X["IsRedEye"] = (hf < 6.0).astype(np.float32)

    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["Distance"] = dist
    X["LogDistance"] = np.log1p(dist.clip(lower=0))

    K = _keys(df)
    for c in CAT_SRC:
        X[c] = pd.Categorical(K[c], categories=_cat_levels[c])
    for name, (spec, _m) in TE_SPECS.items():
        X[name] = _te_key(K, spec).map(_te_maps[name]).fillna(_prior).astype(np.float32)
    return X[FEATS]


# --- model --------------------------------------------------------------------
X = prepare(train)
X[list(TE_SPECS)] = _te_oof.to_numpy()  # train rows use out-of-fold encodings
y = y_train

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(X, y)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
