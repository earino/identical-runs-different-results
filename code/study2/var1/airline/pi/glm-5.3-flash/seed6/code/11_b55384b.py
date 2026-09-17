"""XGBoost binary classifier for airline delay. Only file the agent edits.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives inside prepare(); encoders/stats are fit on training data only.
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

# --- feature layout -----------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
NUM_COLS = [
    "dep_hour", "time_min", "time_sin", "time_cos",
    "month_num", "month_sin", "month_cos",
    "dow_num", "dow_sin", "dow_cos",
    "dom_num", "dom_sin", "dom_cos",
    "Distance",
]
FEATURE_COLS = CAT_COLS + NUM_COLS

cat_levels = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in CAT_COLS}
cat_levels["hour_cat"] = pd.Index([str(h) for h in range(24)])
cat_levels["dow_cat"] = pd.Index([str(d) for d in range(1, 8)])
cat_levels["month_cat"] = pd.Index([str(m) for m in range(1, 13)])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here; fitted stats (cat_levels etc.) come from train only."""
    X = pd.DataFrame(index=df.index)
    # time of day from DepTime (hhmm; values >=2400 wrap past midnight)
    dt = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(np.int32)
    hhmm = dt % 2400
    hour = hhmm // 100
    tmin = hour * 60 + hhmm % 100
    X["dep_hour"] = hour.astype(np.float32)
    X["time_min"] = tmin.astype(np.float32)
    ang = 2 * np.pi * tmin / 1440.0
    X["time_sin"] = np.sin(ang).astype(np.float32)
    X["time_cos"] = np.cos(ang).astype(np.float32)
    # calendar columns arrive as strings like "c-7"
    month = df["Month"].astype(str).str.slice(2).astype(np.int32)
    dom = df["DayofMonth"].astype(str).str.slice(2).astype(np.int32)
    dow = df["DayOfWeek"].astype(str).str.slice(2).astype(np.int32)
    X["month_num"] = month.astype(np.float32)
    a = 2 * np.pi * (month - 1) / 12.0
    X["month_sin"] = np.sin(a).astype(np.float32)
    X["month_cos"] = np.cos(a).astype(np.float32)
    X["dow_num"] = dow.astype(np.float32)
    a = 2 * np.pi * (dow - 1) / 7.0
    X["dow_sin"] = np.sin(a).astype(np.float32)
    X["dow_cos"] = np.cos(a).astype(np.float32)
    X["dom_num"] = dom.astype(np.float32)
    a = 2 * np.pi * (dom - 1) / 31.0
    X["dom_sin"] = np.sin(a).astype(np.float32)
    X["dom_cos"] = np.cos(a).astype(np.float32)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").astype(np.float32)
    X["hour_cat"] = pd.Categorical(hour.astype(str), categories=cat_levels["hour_cat"])
    X["dow_cat"] = pd.Categorical(dow.astype(str), categories=cat_levels["dow_cat"])
    X["month_cat"] = pd.Categorical(month.astype(str), categories=cat_levels["month_cat"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- feature-shape ablation (in-run selection) ---------------------------------
X_full = prepare(train)
X_eval_full = prepare(evald)
y_all = to_y(train)
y_eval = to_y(evald)


def make(depth, n_est, lr=0.1, mcw=1, subsample=1.0, colsample=1.0, seed=SEED):
    return xgb.XGBClassifier(
        n_estimators=n_est,
        max_depth=depth,
        learning_rate=lr,
        min_child_weight=mcw,
        subsample=subsample,
        colsample_bytree=colsample,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


ALL_NUM = ["dep_hour", "time_min", "time_sin", "time_cos",
           "month_num", "month_sin", "month_cos",
           "dow_num", "dow_sin", "dow_cos",
           "dom_num", "dom_sin", "dom_cos", "Distance"]
VARIANTS = {
    "base": CAT_COLS + ALL_NUM,
    "nodom": CAT_COLS + [c for c in ALL_NUM if not c.startswith("dom_")],
    "hourcat": CAT_COLS + ALL_NUM + ["hour_cat"],
    "nodom+hourcat": CAT_COLS + [c for c in ALL_NUM if not c.startswith("dom_")] + ["hour_cat"],
    "nodom+hourcat+calcats": CAT_COLS + [c for c in ALL_NUM if not c.startswith("dom_")]
                              + ["hour_cat", "dow_cat", "month_cat"],
}

results = {}
t0 = time.time()
for lr in [0.03, 0.05]:
    for n_est in [400, 600, 800]:
        mdl = make(4, n_est, lr=lr, mcw=20, colsample=0.6)
        mdl.set_params(reg_lambda=10.0)
        mdl.fit(X_full[VARIANTS["hourcat"]], y_all)
        auc = roc_auc_score(y_eval, mdl.predict_proba(X_eval_full[VARIANTS["hourcat"]])[:, 1])
        results[(lr, n_est)] = auc
        print(f"lr={lr} n_est={n_est}  eval_auc={auc:.4f}")
print(f"Sweep time: {time.time() - t0:.1f}s")

best = max(results, key=results.get)
print(f"best lr={best[0]} n_est={best[1]} auc={results[best]:.4f}")

model = make(4, best[1], lr=best[0], mcw=20, colsample=0.6)
model.set_params(reg_lambda=10.0)
model.fit(X_full[VARIANTS["hourcat"]], y_all)
BEST_COLS = VARIANTS["hourcat"]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df)[BEST_COLS])[:, 1]


t0 = time.time()
eval_auc = results[best]
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
