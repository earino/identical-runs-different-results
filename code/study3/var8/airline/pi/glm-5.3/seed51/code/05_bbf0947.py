"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Findings so far (probed on 2006 eval):
- Month / DayofMonth dropped: 2005 seasonal patterns do not transfer (+0.004 AUC).
- Rare Origin/Dest airports (train count < 500) collapsed to "OTHER": small airports carry year-specific
  noise; only major hubs have transferable delay patterns. This was worth ~+0.01 AUC and unlocked
  deep trees (previously depth > 6 overfit the year shift).
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
RARE = 500  # min train-count for an airport to keep its own category

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "DayOfWeek"]
lv = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
ocnt, dcnt = train["Origin"].value_counts(), train["Dest"].value_counts()
o_cats = pd.Index(sorted(set(ocnt[ocnt >= RARE].index) | {"OTHER"}))
d_cats = pd.Index(sorted(set(dcnt[dcnt >= RARE].index) | {"OTHER"}))

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype(int)
    X["Distance"] = df["Distance"].astype(float)
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"], categories=lv["UniqueCarrier"])
    X["Origin"] = pd.Categorical(df["Origin"].where(df["Origin"].map(ocnt).fillna(0) >= RARE, "OTHER"), categories=o_cats)
    X["Dest"] = pd.Categorical(df["Dest"].where(df["Dest"].map(dcnt).fillna(0) >= RARE, "OTHER"), categories=d_cats)
    X["DayOfWeek"] = pd.Categorical(df["DayOfWeek"], categories=lv["DayOfWeek"])
    return X

def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()

# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=12,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
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
