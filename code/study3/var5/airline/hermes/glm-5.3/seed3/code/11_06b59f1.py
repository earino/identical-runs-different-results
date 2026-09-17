"""XGBoost binary classifier for airline delay prediction.

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

# --- features -----------------------------------------------------------------
# date columns are "c-<n>" strings -> also keep as integers for numeric splits
C_NUM = {"Month": "month", "DayofMonth": "day", "DayOfWeek": "dow"}
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"] + list(C_NUM.keys())
# global category levels from TRAIN ONLY (unseen levels in eval/holdout -> NaN -> missing)
CAT_LEVELS = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
# interaction categories (levels from TRAIN ONLY; unseen combos -> NaN -> missing)
_tr_t = train["DepTime"].astype(float)
_tr_hh = np.clip((_tr_t // 100), 0, 24).astype(int).astype(str)
CAT_LEVELS["ori_hh"] = pd.Index(sorted((train["Origin"].astype(str) + _tr_hh).unique()))
CAT_LEVELS["car_hh"] = pd.Index(sorted((train["UniqueCarrier"].astype(str) + _tr_hh).unique()))


def _cnum(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"c-(\d+)", expand=False).astype(float)


# --- train-only aggregates for relative-departure-time features -----------------
tr_t = train["DepTime"].astype(float)
tr_mins = np.clip((tr_t // 100), 0, 24) * 60 + (tr_t % 100)
_ori = pd.DataFrame({"k": train["Origin"], "t": tr_mins}).groupby("k")["t"].mean()
_car = pd.DataFrame({"k": train["UniqueCarrier"], "t": tr_mins}).groupby("k")["t"].mean()
ORI_MEAN_DEP = _ori.to_dict()
CAR_MEAN_DEP = _car.to_dict()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c, name in C_NUM.items():
        X[name] = _cnum(df[c])
    t = df["DepTime"].astype(float)
    hh = np.clip((t // 100), 0, 24)  # a few 24xx/25xx values exist
    minutes = hh * 60 + (t % 100)
    X["minutes"] = minutes
    X["tod_sin"] = np.sin(2 * np.pi * minutes / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * minutes / 1440.0)
    X["distance"] = df["Distance"].astype(float)
    X["dist_x_cos"] = X["distance"] * X["tod_cos"]
    X["hour"] = hh
    X["doy_approx"] = X["month"] * 31 + X["day"]
    om = df["Origin"].map(ORI_MEAN_DEP).astype(float)
    cm = df["UniqueCarrier"].map(CAR_MEAN_DEP).astype(float)
    X["rel_to_ori_mean"] = minutes - om
    X["rel_to_car_mean"] = minutes - cm
    X["hour_sin"] = np.sin(2 * np.pi * hh / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hh / 24.0)
    X["sin_x_dist"] = X["tod_sin"] * X["distance"]
    X["log_dist"] = np.log1p(X["distance"])
    X["sqrt_dist"] = np.sqrt(X["distance"])
    for c in ("UniqueCarrier", "Origin", "Dest", "DayOfWeek"):  # numeric month/day only
        X[c] = pd.Categorical(df[c].astype(str), categories=CAT_LEVELS[c])
    hh_str = hh.astype(int).astype(str)
    X["ori_hh"] = pd.Categorical(df["Origin"].astype(str) + hh_str, categories=CAT_LEVELS["ori_hh"])
    X["car_hh"] = pd.Categorical(df["UniqueCarrier"].astype(str) + hh_str, categories=CAT_LEVELS["car_hh"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: seed/depth ensemble of XGBoost classifiers --------------------------
SEEDS = [42, 1, 7, 13, 2, 3, 5, 11, 17, 19, 23, 29]
DEPTHS = [6, 6, 6, 6, 5, 5, 4, 4, 8, 8, 7, 3]
PARAMS = dict(
    n_estimators=400,
    learning_rate=0.1,
    reg_lambda=10,
    reg_alpha=5,
    colsample_bytree=0.6,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_train = prepare(train)
y_train = to_y(train)
models = []
for s, d in zip(SEEDS, DEPTHS):
    m = xgb.XGBClassifier(random_state=s, max_depth=d, **PARAMS)
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
