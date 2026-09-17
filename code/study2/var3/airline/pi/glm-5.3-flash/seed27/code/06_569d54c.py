"""Airline delay XGBoost classifier. ONLY FILE THE AGENT EDITS.

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

# --- encoders fitted on TRAIN only --------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
y_full = (train[TARGET] == POSITIVE).astype(int)
PRIOR = y_full.mean()
TE_SMOOTH = 30.0


def _te_map(keys: pd.Series) -> dict:
    """Smoothed target-encoding map: (sum + prior*m) / (count + m)."""
    g = pd.DataFrame({"k": keys.values, "y": y_full.values}).groupby("k")["y"].agg(["sum", "count"])
    return ((g["sum"] + PRIOR * TE_SMOOTH) / (g["count"] + TE_SMOOTH)).to_dict()


def _cnt_map(keys: pd.Series) -> dict:
    return keys.value_counts().to_dict()


TE_MAPS = {
    "Origin": _te_map(train["Origin"]),
    "Dest": _te_map(train["Dest"]),
    "UniqueCarrier": _te_map(train["UniqueCarrier"]),
}
CNT_MAPS = {
    "Origin": _cnt_map(train["Origin"]),
    "Dest": _cnt_map(train["Dest"]),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    # calendar features: c-<n> strings -> ints
    X["month"] = df["Month"].str[2:].astype(int)
    X["dom"] = df["DayofMonth"].str[2:].astype(int)
    X["dow"] = df["DayOfWeek"].str[2:].astype(int)
    # scheduled departure time
    dep = df["DepTime"]
    X["deptime"] = dep
    X["hour"] = dep // 100
    minutes = (dep // 100) * 60 + dep % 100
    ang = 2 * np.pi * minutes / 1440.0
    X["dep_sin"] = np.sin(ang)
    X["dep_cos"] = np.cos(ang)
    # distance
    X["distance"] = df["Distance"]
    X["dist_log"] = np.log1p(df["Distance"])
    # native categoricals
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    # target encodings (maps fitted on train only)
    for key, series in [("Origin", df["Origin"]), ("Dest", df["Dest"]),
                        ("UniqueCarrier", df["UniqueCarrier"])]:
        X["te_" + key] = series.map(TE_MAPS[key]).astype(float)
        X["te_" + key] = X["te_" + key].fillna(PRIOR)
    for key, series in [("Origin", df["Origin"]), ("Dest", df["Dest"])]:
        X["cnt_" + key] = series.map(CNT_MAPS[key]).astype(float)
        X["cnt_" + key] = X["cnt_" + key].fillna(0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
rng = np.random.RandomState(SEED)
val_idx = rng.choice(len(train), size=len(train) // 10, replace=False)
fit_mask = np.ones(len(train), dtype=bool)
fit_mask[val_idx] = False

model = xgb.XGBClassifier(
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(
    prepare(train[fit_mask == 1]), to_y(train[fit_mask == 1]),
    eval_set=[(prepare(train[fit_mask == 0]), to_y(train[fit_mask == 0]))],
    verbose=False,
)
best_it = int(model.best_iteration)
print(f"ES fit: {time.time() - t0:.1f}s  best_iter: {best_it}  val_auc: {model.best_score:.4f}")

# refit an ensemble of diverse bagged models on the FULL training set
SPECS = [  # (depth, tree-count scale vs ES best_iter, colsample_bytree)
    (6, 1.3, 0.8),
    (7, 1.2, 0.8),
    (8, 1.1, 0.8),
    (8, 1.1, 0.6),
    (9, 1.0, 0.8),
    (10, 0.9, 0.8),
]
models = []
t0 = time.time()
for s, (d, scale, cs) in enumerate(SPECS):
    n_trees = int(best_it * scale) + 1
    m = xgb.XGBClassifier(
        n_estimators=n_trees,
        learning_rate=0.05,
        max_depth=d,
        subsample=0.85,
        colsample_bytree=cs,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + s * 1000,
        n_jobs=N_JOBS,
    )
    m.fit(prepare(train), to_y(train))
    models.append(m)
print(f"Bag refit x{len(SPECS)}: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Ps = [m.predict_proba(prepare(df))[:, 1] for m in models]
    return np.mean(Ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
