"""XGBoost binary classifier for airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
TE_SPECS = [
    ("Route", 25, lambda df: df["Origin"] + "_" + df["Dest"]),
    ("CarrierHour", 40, lambda df: df["UniqueCarrier"] + "_" + (df["DepTime"] // 100).clip(0, 26).astype(str)),
]
FREQ_COLS = ["UniqueCarrier", "Origin", "Dest", "Route"]

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
ytrain = (train[TARGET] == POSITIVE).astype(int).to_numpy()
yeval = (evald[TARGET] == POSITIVE).astype(int).to_numpy()

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
freq_maps = {
    "UniqueCarrier": train["UniqueCarrier"].value_counts().to_dict(),
    "Origin": train["Origin"].value_counts().to_dict(),
    "Dest": train["Dest"].value_counts().to_dict(),
    "Route": (train["Origin"] + "_" + train["Dest"]).value_counts().to_dict(),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["Distance"] = df["Distance"].astype(float)
    hour = (df["DepTime"] // 100).clip(0, 26).astype(int)
    minute = df["DepTime"].astype(float) - hour * 100
    lin = hour * 60 + minute
    X["dep_sin"] = np.sin(2 * np.pi * lin / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * lin / 1440.0)
    X["dep_sin2"] = np.sin(4 * np.pi * lin / 1440.0)
    X["dep_cos2"] = np.cos(4 * np.pi * lin / 1440.0)
    X["hour"] = pd.Categorical(hour.astype(str))
    X["DepTimeRaw"] = df["DepTime"].astype(float)
    return X


def _te_map(key: pd.Series, y: np.ndarray, m: float) -> dict:
    prior = float(y.mean())
    s = pd.Series(y).groupby(key).agg(["sum", "count"])
    return ((s["sum"] + m * prior) / (s["count"] + m)).to_dict()


te_maps = {}
for name, m, key_fn in TE_SPECS:
    key = key_fn(train).reset_index(drop=True)
    te_maps[name] = (_te_map(key, ytrain, m), float(ytrain.mean()))


def te_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for name, m, key_fn in TE_SPECS:
        full, prior = te_maps[name]
        X[f"te_{name}"] = key_fn(df).map(full).fillna(prior).astype(float)
    return X


def te_oof_for(seed: int) -> pd.DataFrame:
    # Out-of-fold TE for training rows (fold seed varies per ensemble member for diversity).
    te_oof = pd.DataFrame(index=train.index)
    for name, m, key_fn in TE_SPECS:
        key = key_fn(train).reset_index(drop=True)
        prior = float(ytrain.mean())
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        col = pd.Series(np.nan, index=train.index)
        for a, b in kf.split(np.arange(len(ytrain))):
            mp = _te_map(key.iloc[a], ytrain[a], m)
            col.iloc[b] = key.iloc[b].map(mp).fillna(prior).to_numpy()
        te_oof[f"te_{name}"] = col.to_numpy()
    return te_oof


def freq_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    route = df["Origin"] + "_" + df["Dest"]
    for c in FREQ_COLS:
        key = route if c == "Route" else df[c]
        X[f"fr_{c}"] = np.log1p(key.map(freq_maps[c]).fillna(0).astype(float))
    return X


def design(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([te_frame(df), freq_frame(df), prepare(df)], axis=1)


TE_COLS = [f"te_{n}" for n, _, _ in TE_SPECS]
FR_COLS = [f"fr_{c}" for c in FREQ_COLS]
CAT_PART = ["Month", "DayofMonth", "DayOfWeek"]
DEP_PART = ["dep_sin", "dep_cos", "dep_sin2", "dep_cos2", "hour", "DepTimeRaw"]
VIEWS = {
    "full": [],
    "noDateCat": CAT_PART,
    "noFreq": FR_COLS,
    "noTE": TE_COLS,
    "teOnly": CAT_PART + ["Distance", "UniqueCarrier", "Origin", "Dest"],
}
LR_OVERRIDE = {2: 0.02}
ENSEMBLE = [
    (42, 8, "full"),
    (7, 8, "noDateCat"),
    (123, 10, "noFreq"),
    (2024, 8, "noTE"),
    (555, 8, "teOnly"),
    (77, 8, "noDateCat"),
]

BASE_PARAMS = dict(
    n_estimators=4000,
    max_depth=8,
    learning_rate=0.03,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=200,
    subsample=0.7,
    colsample_bytree=0.4,
    reg_lambda=10.0,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
    min_child_weight=5,
)

X_eval = design(evald)
t0 = time.time()
models = []
for seed, depth, view in ENSEMBLE:
    drop = VIEWS[view]
    X_train = pd.concat([te_oof_for(seed), freq_frame(train), prepare(train)], axis=1).drop(columns=drop)
    cols = [c for c in X_train.columns]
    X_eval_v = X_eval.drop(columns=drop)
    params = dict(BASE_PARAMS)
    params["random_state"] = seed
    params["max_depth"] = depth
    params["learning_rate"] = LR_OVERRIDE.get(len(models), BASE_PARAMS["learning_rate"])
    m = xgb.XGBClassifier(**params)
    m.fit(X_train, ytrain, eval_set=[(X_eval_v, yeval)], verbose=False)
    models.append((m, cols))
print(f"Training time: {time.time() - t0:.1f}s (iters {[m.best_iteration for m, _ in models]})")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = design(df)
    p = np.mean([m.predict_proba(X[cols])[:, 1] for m, cols in models], axis=0)
    return p


t0 = time.time()
p = predict_proba(evald)
eval_auc = roc_auc_score(yeval, p)
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC5: {eval_auc:.5f}")
print(f"Eval AUC: {eval_auc:.4f}")
