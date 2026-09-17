"""XGBoost binary classifier for the airline delay task.

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
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42
TE_SMOOTH = 20.0  # target-encoding shrinkage towards the prior

train_raw = pd.read_csv("data/train.csv")
y_train = (train_raw[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(y_train.mean())

TE_COLS = ["carrier", "origin", "dest", "route"]


def _te_keys(df: pd.DataFrame) -> dict:
    return {
        "carrier": df["UniqueCarrier"].astype(str),
        "origin": df["Origin"].astype(str),
        "dest": df["Dest"].astype(str),
        "route": df["Origin"].astype(str) + "_" + df["Dest"].astype(str),
    }


def _te_fit(keys: pd.Series, y: np.ndarray) -> dict:
    g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    enc = (g["sum"] + TE_SMOOTH * PRIOR) / (g["count"] + TE_SMOOTH)
    return enc.to_dict()


def _te_apply(keys: pd.Series, enc: dict) -> pd.Series:
    return keys.map(enc).fillna(PRIOR).astype("float64")


# --- encodings fit on TRAINING data only --------------------------------------
_keys_train = _te_keys(train_raw)
TE_FULL = {c: _te_fit(_keys_train[c], y_train) for c in TE_COLS}
# out-of-fold encodings for training rows (unbiased targets for the model to learn from)
TE_OOF = {c: np.empty(len(train_raw), dtype="float64") for c in TE_COLS}
_kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for _tr, _va in _kf.split(train_raw):
    for c in TE_COLS:
        enc_f = _te_fit(_keys_train[c].iloc[_tr], y_train[_tr])
        TE_OOF[c][_va] = _te_apply(_keys_train[c].iloc[_va], enc_f).to_numpy()

# category dictionaries (fit on training data only)
CAT_FEATURES = {
    "carrier": sorted(train_raw["UniqueCarrier"].astype(str).unique().tolist()),
    "origin": sorted(train_raw["Origin"].astype(str).unique().tolist()),
    "dest": sorted(train_raw["Dest"].astype(str).unique().tolist()),
    "route": sorted(
        (train_raw["Origin"].astype(str) + "_" + train_raw["Dest"].astype(str)).unique().tolist()
    ),
}
CAT_CODES = {c: {v: i for i, v in enumerate(vals)} for c, vals in CAT_FEATURES.items()}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything
    # computed on train/eval outside this function would NOT be applied to the hidden holdout.
    # Encoders/statistics above are fit on training data only and referenced read-only here.
    X = pd.DataFrame(index=df.index)
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    X["sched_dep_time_min"] = (dt // 100) * 60 + (dt % 100)
    hour = np.floor(X["sched_dep_time_min"] / 60)
    X["hour"] = np.clip(hour, 0, 23)  # hhmm > 2359 exists (e.g. 24xx/26xx): clamp into the day
    X["night"] = ((X["hour"] >= 21) | (X["hour"] <= 4)).astype(float)
    month_num = pd.to_numeric(
        df["Month"].astype(str).str.removeprefix("c-"), errors="coerce"
    ).fillna(1.0)
    X["month_num"] = month_num
    ang = 2 * np.pi * (month_num - 1) / 12
    X["month_sin"] = np.sin(ang)
    X["month_cos"] = np.cos(ang)
    X["day_of_month"] = pd.to_numeric(
        df["DayofMonth"].astype(str).str.removeprefix("c-"), errors="coerce"
    ).fillna(15.0)
    X["day_of_week"] = pd.to_numeric(
        df["DayOfWeek"].astype(str).str.removeprefix("c-"), errors="coerce"
    ).fillna(4.0)
    X["distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["log_distance"] = np.log1p(X["distance"])
    keys = _te_keys(df)
    for c in TE_COLS:
        X["te_" + c] = _te_apply(keys[c], TE_FULL[c])
        X[c] = keys[c].map(CAT_CODES[c]).astype("float64")  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


model = xgb.XGBClassifier(
    n_estimators=900,
    max_depth=9,
    learning_rate=0.03,
    min_child_weight=8,
    subsample=0.9,
    colsample_bytree=0.7,
    reg_lambda=2.0,
    tree_method="hist",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_train = prepare(train_raw)
for c in TE_COLS:  # training rows see out-of-fold encodings (no target leakage)
    X_train["te_" + c] = TE_OOF[c]
evald = pd.read_csv("data/eval.csv")
model.fit(X_train, y_train)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
