"""Sweep 4: encoding variants for Origin/Dest on the reduced (calendar-free) feature set."""
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
BEST = dict(max_depth=3, n_estimators=300, learning_rate=0.05, min_child_weight=10)

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
to_y = lambda df: (df[TARGET] == POSITIVE).astype(int).to_numpy()
ytr, yev = to_y(train), to_y(evald)

BASE = ["DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
route_tr = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
ROUTE_LEVELS = pd.Index(sorted(route_tr.unique()))
freq_o = train["Origin"].value_counts()
freq_d = train["Dest"].value_counts()


def build(extra):
    """extra: dict of name -> callable(df) -> pd.Series (raw, gets converted to category if object)."""
    cols = list(BASE)
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(train[c])]
    levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

    def prepare(df):
        X = df[cols].copy()
        for c in cat_cols:
            X[c] = pd.Categorical(X[c], categories=levels[c])
        for name, fn in extra.items():
            s = fn(df)
            if s.dtype == object:
                lv = sorted(train[name].unique()) if name in train else None
                X[name] = pd.Categorical(s, categories=pd.Index(lv))
            else:
                X[name] = s
        return X

    return prepare


EXTRA = {
    "route": lambda df: (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).map(
        lambda r: r if r in set(ROUTE_LEVELS) else "__UNK__"),
    "freq_origin": lambda df: np.log1p(df["Origin"].map(freq_o).fillna(0).to_numpy()),
    "freq_dest": lambda df: np.log1p(df["Dest"].map(freq_d).fillna(0).to_numpy()),
}
train = train.assign(route=train["Origin"].astype(str) + "_" + train["Dest"].astype(str))
train["route"] = train["route"].where(train["route"].isin(set(ROUTE_LEVELS)), "__UNK__")
evald = evald.assign(route=(evald["Origin"].astype(str) + "_" + evald["Dest"].astype(str)))
evald["route"] = evald["route"].where(evald["route"].isin(set(ROUTE_LEVELS)), "__UNK__")

CASES = [
    ("base", {}, {}),
    ("base+route", {"route": EXTRA["route"]}, {}),
    ("base+freqOD", {"freq_origin": EXTRA["freq_origin"], "freq_dest": EXTRA["freq_dest"]}, {}),
    ("base+route+freqOD", EXTRA, {}),
    ("base+route(lowcard)", {"route": EXTRA["route"]}, dict(max_cat_threshold=8)),
    ("base, noDst", {}, {}),
    ("base, maxbin64", {}, dict(max_bin=64)),
    ("base, mcw30", {}, dict(min_child_weight=30)),
    ("base, gamma1", {}, dict(gamma=1.0)),
    ("base, subsample0.7", {}, dict(subsample=0.7, colsample_bytree=0.7)),
]

results = []
t0 = time.time()
for name, extra, over in CASES:
    prep = build(extra)
    cols = list(BASE) + list(extra)
    if name == "base, noDst":
        cols = [c for c in cols if c != "Dest"]
        prep = None

        cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(train[c])]
        levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

        def prep(df, cols=cols, cat_cols=cat_cols, levels=levels):
            X = df[cols].copy()
            for c in cat_cols:
                X[c] = pd.Categorical(X[c], categories=levels[c])
            return X

    params = dict(tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS)
    params.update(BEST)
    params.update(over)
    m = xgb.XGBClassifier(**params)
    m.fit(prep(train), ytr)
    auc = roc_auc_score(yev, m.predict_proba(prep(evald))[:, 1])
    results.append((auc, name, extra, over))
    print(f"  AUC {auc:.4f}  {name:22s} {over}  [{time.time() - t0:.0f}s]", flush=True)

results.sort(key=lambda r: -r[0])
print("BEST:", results[0][:3])
print(f"Training time: {time.time() - t0:.1f}s")

_, best_name, best_extra, best_over = results[0]
if best_name == "base, noDst":
    cols = [c for c in BASE if c != "Dest"]
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(train[c])]
    levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

    def prep(df):
        X = df[cols].copy()
        for c in cat_cols:
            X[c] = pd.Categorical(X[c], categories=levels[c])
        return X
else:
    prep = build(best_extra)

params = dict(tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS, **BEST)
params.update(best_over)
model = xgb.XGBClassifier(**params)
model.fit(prep(train), ytr)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prep(df))[:, 1]


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
