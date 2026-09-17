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

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42
N_FOLDS = 5
TE_SMOOTH = 20.0

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

TE_COLS = ["UniqueCarrier", "Origin", "Dest", "route"]


def _add_route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# --- statistics fit on TRAIN ONLY ---------------------------------------------
train_y = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(train_y.mean())

route_levels = pd.Index(sorted(_add_route(train).unique()))
freq_maps = {}
for c in ["UniqueCarrier", "Origin", "Dest"]:
    vc = train[c].value_counts()
    freq_maps[c] = (vc / len(train)).to_dict()
freq_maps["route"] = (_add_route(train).value_counts() / len(train)).to_dict()


def _te_map(levels: pd.Series, y: np.ndarray, m: float = TE_SMOOTH):
    """level -> smoothed mean of y; fit on given rows only."""
    g = pd.DataFrame({"k": levels.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    te = (g["sum"] + m * PRIOR) / (g["count"] + m)
    return te.to_dict()


# full-train maps (used by prepare() for any new data)
te_full = {
    c: _te_map((_add_route(train) if c == "route" else train[c]), train_y) for c in TE_COLS
}
# out-of-fold maps for training rows (avoid leakage into the training features)
rng = np.random.RandomState(SEED)
fold_ids = rng.randint(0, N_FOLDS, size=len(train))
te_oof = []
for f in range(N_FOLDS):
    mask = fold_ids != f
    te_oof.append({c: _te_map((_add_route(train) if c == "route" else train[c])[mask], train_y[mask]) for c in TE_COLS})


def _base_prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # time-of-day features from DepTime (hhmm)
    dep = _to_int(df["DepTime"])
    hour = (dep // 100).astype(float)
    dep_min = hour * 60 + (dep % 100).astype(float)
    dep_min_mod = dep_min % 1440
    X["dep_hour"] = hour
    X["dep_min_mod"] = dep_min_mod
    ang = 2 * np.pi * dep_min_mod / 1440.0
    X["dep_sin"] = np.sin(ang)
    X["dep_cos"] = np.cos(ang)
    dow = _to_int(df["DayOfWeek"].str.replace("c-", "", regex=False))
    mon = _to_int(df["Month"].str.replace("c-", "", regex=False))
    dom = _to_int(df["DayofMonth"].str.replace("c-", "", regex=False))
    doy = (mon - 1) * 31 + dom
    for name, v, period in (("dow", dow, 7.0), ("doy", doy, 372.0)):
        a = 2 * np.pi * v / period
        X[f"{name}_sin"] = np.sin(a)
        X[f"{name}_cos"] = np.cos(a)
    X["distance_log"] = np.log1p(_to_int(df["Distance"]))
    return X


def _apply_freq_te(X: pd.DataFrame, df: pd.DataFrame, te_maps, fold_ids=None) -> pd.DataFrame:
    route = _add_route(df)
    for c in ["UniqueCarrier", "Origin", "Dest"]:
        X[c + "_freq"] = df[c].map(freq_maps[c]).astype(float).fillna(0.0)
    X["route_freq"] = route.map(freq_maps["route"]).astype(float).fillna(0.0)
    src = {"UniqueCarrier": df["UniqueCarrier"], "Origin": df["Origin"], "Dest": df["Dest"], "route": route}
    if fold_ids is None:
        for c in TE_COLS:
            X[c + "_te"] = src[c].map(te_maps[c]).astype(float).fillna(PRIOR)
    else:
        for c in TE_COLS:
            X[c + "_te"] = PRIOR
            for f in range(N_FOLDS):
                mask = fold_ids == f
                if mask.any():
                    X.loc[mask, c + "_te"] = src[c][mask].map(te_maps[f][c]).astype(float).fillna(PRIOR).to_numpy()
    return X


def build_X(df: pd.DataFrame, te_maps, fold_ids=None) -> pd.DataFrame:
    """Full feature engineering; same column order for every caller."""
    X = _base_prepare(df)
    route = _add_route(df)
    X = _apply_freq_te(X, df, te_maps, fold_ids=fold_ids)
    X["route"] = pd.Categorical(route, categories=route_levels)
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Contract entry point: full feature engineering for ANY new dataframe (full-train statistics)."""
    return build_X(df, te_full)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    max_depth=6,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
# training rows get OUT-OF-FOLD TE (leak-free); new data via prepare() gets full-train TE
X_train = build_X(train, te_oof, fold_ids=fold_ids)
y_train = train_y

idx = np.random.RandomState(SEED).permutation(len(X_train))
cut = int(0.8 * len(X_train))
tr_idx, va_idx = idx[:cut], idx[cut:]

es_model = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=50, **PARAMS)
es_model.fit(X_train.iloc[tr_idx], y_train[tr_idx], eval_set=[(X_train.iloc[va_idx], y_train[va_idx])], verbose=False)
best_n = int(es_model.best_iteration) + 1
print(f"best rounds: {best_n}")

model = xgb.XGBClassifier(n_estimators=best_n, **PARAMS)
model.fit(X_train, y_train)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
