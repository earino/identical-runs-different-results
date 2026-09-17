"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Approach (found by probing 2006 eval, aiming for robustness to the 2005->2006 shift):
- Drop Month / DayofMonth: 2005 seasonal + day-of-month patterns do not transfer.
- Rare Origin/Dest airports (train count below a threshold) collapse to "OTHER": small airports
  carry year-specific noise; a ladder of thresholds (150..1500) gives diverse, complementary views.
- Deep trees (d14/d16, 100 rounds, lr .05) + colsample_bytree 0.85 (~5 of 6 features per tree):
  per-tree feature subsampling decorrelates trees and generalizes much better across the year shift.
- Final prediction = mean over the 16 members (8 thresholds x 2 depths).
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

RARES = [150, 200, 300, 400, 500, 700, 1000, 1500]
DEPTHS = [14, 16]

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
# Encoders/statistics below are fit on TRAIN only (never on the df passed to prepare).
lv = {c: pd.Index(sorted(train[c].dropna().unique())) for c in ["UniqueCarrier", "DayOfWeek"]}
ocnt, dcnt = train["Origin"].value_counts(), train["Dest"].value_counts()

def _bucket(v: pd.Series, cnt: pd.Series, rare: int) -> pd.Series:
    return v.where(v.map(cnt).fillna(0) >= rare, "OTHER")

def _cats(cnt: pd.Series, rare: int) -> pd.Index:
    return pd.Index(sorted(set(cnt[cnt >= rare].index) | {"OTHER"}))

def prepare(df: pd.DataFrame, rare: int) -> pd.DataFrame:
    # ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype(int)
    X["Distance"] = df["Distance"].astype(float)
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"], categories=lv["UniqueCarrier"])
    X["Origin"] = pd.Categorical(_bucket(df["Origin"], ocnt, rare), categories=_cats(ocnt, rare))
    X["Dest"] = pd.Categorical(_bucket(df["Dest"], dcnt, rare), categories=_cats(dcnt, rare))
    X["DayOfWeek"] = pd.Categorical(df["DayOfWeek"], categories=lv["DayOfWeek"])
    return X

def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()

# --- model: 16-member XGBoost ensemble ----------------------------------------
y_tr = to_y(train)
MODELS = []  # (rare, fitted model)
t0 = time.time()
for rare in RARES:
    Xtr = prepare(train, rare)
    for depth in DEPTHS:
        m = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=depth,
            learning_rate=0.05,
            colsample_bytree=0.85,
            tree_method="hist",
            enable_categorical=True,
            random_state=SEED,
            n_jobs=N_JOBS,
        )
        m.fit(Xtr, y_tr)
        MODELS.append((rare, m))
print(f"Training time: {time.time() - t0:.1f}s ({len(MODELS)} members)")

def predict_proba(df: pd.DataFrame) -> np.ndarray:
    ps = [m.predict_proba(prepare(df, rare))[:, 1] for rare, m in MODELS]
    return np.mean(ps, axis=0)

t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
