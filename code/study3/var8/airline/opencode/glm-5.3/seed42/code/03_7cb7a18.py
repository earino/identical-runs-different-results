"""XGBoost airline-delay classifier: 2-view blend (native-categorical model + deep one-hot model).

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. Module-level `predict_proba(df)`: raw DataFrame -> 1-D array of P(positive).
All feature engineering lives inside prepare(); encodings/categories are fitted on data/train.csv only.
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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
HOUR_CATS = list(range(0, 27))  # DepTime hhmm -> hours 0..26 both years


def _num(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "MonthN": df["Month"].str.replace("c-", "", regex=False).astype(int),
        "DayN": df["DayofMonth"].str.replace("c-", "", regex=False).astype(int),
        "DoWN": df["DayOfWeek"].str.replace("c-", "", regex=False).astype(int),
        "DepTime": df["DepTime"],
        "Distance": df["Distance"],
    })


def _hour_dummies(df: pd.DataFrame) -> pd.DataFrame:
    H = pd.get_dummies(pd.Categorical(df["DepTime"] // 100, categories=HOUR_CATS), dummy_na=True)
    H.columns = [f"h{c}" for c in H.columns]
    return H.astype(np.float32)


def _cat_dummies(df: pd.DataFrame) -> pd.DataFrame:
    D = pd.get_dummies(
        pd.DataFrame({c: pd.Categorical(df[c], categories=cat_levels[c]) for c in CAT_COLS}),
        dummy_na=True)
    return D.astype(np.float32)


def prepare(df: pd.DataFrame) -> dict:
    """Feature engineering for both model views. Fitted mappings come from `train` only."""
    num = _num(df)
    hd = _hour_dummies(df)
    native = num.copy()
    for c in CAT_COLS:
        native[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    native = pd.concat([native, hd], axis=1)
    onehot = pd.concat([num, _cat_dummies(df), hd], axis=1)
    return {"native": native, "onehot": onehot}


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# --- models --------------------------------------------------------------------
y = to_y(train)
F = prepare(train)

model_native = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05, tree_method="hist",
    enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
)
model_onehot = xgb.XGBClassifier(
    n_estimators=300, max_depth=16, learning_rate=0.05, tree_method="hist",
    random_state=SEED, n_jobs=N_JOBS,
)

t0 = time.time()
model_native.fit(F["native"], y)
print(f"Native model: {time.time() - t0:.1f}s")
t0 = time.time()
model_onehot.fit(F["onehot"], y)
print(f"One-hot model: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Fx = prepare(df)
    s = _logit(model_native.predict_proba(Fx["native"])[:, 1])
    s += _logit(model_onehot.predict_proba(Fx["onehot"])[:, 1])
    return 1 / (1 + np.exp(-s / 2))


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
