"""XGBoost binary classifier for the airline delay task.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     ALL feature engineering lives in prepare(), which is the only code path predict_proba uses.
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

# --- feature spec (fit on training data only) ---------------------------------
RAW_FEATS = [c for c in train.columns if c not in ID_COLS + [TARGET]]
CAT_RAW = ["UniqueCarrier", "Origin", "Dest"]

CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in CAT_RAW}
# route = Origin_Dest, encoded by frequency (fit on train) + categorical code
train_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
FREQ_MAPS = {
    "carrier": train["UniqueCarrier"].astype(str).value_counts(),
    "origin": train["Origin"].astype(str).value_counts(),
    "dest": train["Dest"].astype(str).value_counts(),
    "route": train_route.value_counts(),
}
ROUTE_LEVELS = pd.Index(sorted(FREQ_MAPS["route"].index))


def _codes(s: pd.Series, levels: pd.Index) -> np.ndarray:
    return pd.Categorical(s.astype(str), categories=levels).codes.astype(np.float32)


def _freq(s: pd.Series, counts: pd.Series) -> np.ndarray:
    return s.astype(str).map(counts).fillna(-1.0).astype(np.float32).to_numpy()


def _cnum(s: pd.Series) -> pd.Series:
    """'c-4' -> 4.0 for any c-<n> style column."""
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)

    # time of day -------------------------------------------------------------
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hh = np.floor(dt / 100.0)
    mm = dt - hh * 100.0
    bad = (hh < 0) | (hh > 23) | (mm < 0) | (mm > 59)
    hh = hh.where(~bad)
    mm = mm.where(~bad)
    hour = hh + mm / 60.0
    X["tod"] = hour.astype(np.float32)
    X["minute"] = mm.astype(np.float32)
    X["tod_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    X["tod_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    X["tod_missing"] = bad.astype(np.float32)

    # calendar ----------------------------------------------------------------
    month = _cnum(df["Month"])
    dom = _cnum(df["DayofMonth"])
    dow = _cnum(df["DayOfWeek"])
    X["month"] = month.astype(np.float32)
    X["day_of_month"] = dom.astype(np.float32)
    X["day_of_week"] = dow.astype(np.float32)
    X["is_weekend"] = (dow >= 6).astype(np.float32)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7).astype(np.float32)

    # distance ----------------------------------------------------------------
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["distance"] = dist.astype(np.float32)
    X["log_distance"] = np.log1p(dist).astype(np.float32)

    # categorical identities ---------------------------------------------------
    carrier = df["UniqueCarrier"].astype(str)
    origin = df["Origin"].astype(str)
    dest = df["Dest"].astype(str)
    route = origin + "_" + dest
    X["carrier"] = pd.Categorical(carrier, categories=CAT_LEVELS["UniqueCarrier"])
    X["carrier_freq"] = _freq(carrier, FREQ_MAPS["carrier"])
    X["origin_freq"] = _freq(origin, FREQ_MAPS["origin"])
    X["dest_freq"] = _freq(dest, FREQ_MAPS["dest"])
    X["route_freq"] = _freq(route, FREQ_MAPS["route"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=0,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=2,
    grow_policy="lossguide",
    max_leaves=96,
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=1,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
