"""Synthesis 5 (after exp 33; 7 experiments left)

Best: exp27 = 0.7185, commit f59af1d. Composition: ddr(Month,Dow,Carrier,Origin,Dest;
smooth=50, unseen->prior) + hetero bag 36 members (12 cfg families x 3 seeds),
subsample .9 colsample .85, probability mean.
Rejected since: dist interactions (0.7184), median blend (0.7177), NaN-unseen-ddr
(0.7184), re-seed (0.7182), bag 48 (0.7184), slow-lr members (0.7184). All within
noise of 0.7185 — the configuration is at a local optimum w.r.t. everything tried.

What has NOT been tried, ranked by (probability of gain x size of gain):
1. Feature-set ablation under the bag: does the bag want Distance at all? Is
   DayofMonth pulling weight or is it pure noise the bag averages out? A leaner,
   more physical feature set sometimes survives year-drift better.
2. Monotone constraint on ddr features is impossible in the sklearn API with
   categorical splits; skip.
3. Two-stage stacking: OOF predictions of a few configs as meta-features for a
   final xgboost (fits contract: still XGBoost end-to-end). Costs OOF compute;
   risk of stacking onto eval-noise.
Plan: exp34 = lean feature set ablation (drop DayofMonth category, keep ddr);
exp35 = stacking if 34 fails; keep last 3 experiments as buffer for the final
validate + one robustness re-check of the winner.
"""
import warnings

warnings.filterwarnings("ignore")

import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

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
# exp34/36: raw categorical copies of weak/drifting features let the model memorize
# noise; keep only their smoothed ddr encodings (DayofMonth dropped in exp34: +0.0015)
feature_cols = [c for c in feature_cols if c not in ("DayofMonth", "Month")]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

DDR_COLS = ["Month", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "DayofMonth"]
SMOOTH = 50.0


def fit_ddr_maps(df: pd.DataFrame):
    y = (df[TARGET] == POSITIVE).astype(float)
    prior = float(y.mean())
    maps = {}
    for c in DDR_COLS:
        g = y.groupby(df[c]).agg(["mean", "count"])
        sm = (g["mean"] * g["count"] + prior * SMOOTH) / (g["count"] + SMOOTH)
        maps[c] = sm.to_dict()
    return maps, prior


def prepare(df: pd.DataFrame, ddr_maps=None, prior=None) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    if ddr_maps is not None:
        for c in DDR_COLS:
            X["ddr_" + c] = df[c].astype(object).map(ddr_maps[c]).astype(float).fillna(prior).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


y_full = to_y(train)

# --- OOF diagnostic -------------------------------------------------------------
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
for tr_idx, va_idx in skf.split(train, y_full):
    mp, pr = fit_ddr_maps(train.iloc[tr_idx])
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED,
                          n_jobs=N_JOBS, n_estimators=60, learning_rate=0.05, max_depth=6,
                          subsample=0.9, colsample_bytree=0.85)
    m.fit(prepare(train.iloc[tr_idx], mp, pr), y_full[tr_idx], verbose=False)
    oof[va_idx] = m.predict_proba(prepare(train.iloc[va_idx], mp, pr))[:, 1]
print(f"OOF AUC: {roc_auc_score(y_full, oof):.4f}")

# --- model: heterogeneous bag (best known composition) ---------------------------
CFGS = [(30, 0.1, 3), (60, 0.05, 3), (30, 0.1, 4), (60, 0.05, 4), (30, 0.1, 5), (60, 0.05, 5),
        (30, 0.1, 6), (60, 0.05, 6), (30, 0.1, 8), (60, 0.05, 8), (30, 0.1, 10), (60, 0.05, 10)]
MEMBERS = [c for c in CFGS for _ in range(3)]
CONFIG = dict(subsample=0.9, colsample_bytree=0.85)

mp, prior = fit_ddr_maps(train)
X = prepare(train, mp, prior)

t0 = time.time()
models = []
for k, (ne, lr, md) in enumerate(MEMBERS):
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED + k,
                          n_jobs=N_JOBS, n_estimators=ne, learning_rate=lr, max_depth=md, **CONFIG)
    m.fit(X, y_full, verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df, mp, prior)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
