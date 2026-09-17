"""XGBoost binary classifier with out-of-fold target/frequency encodings.

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
SMOOTH = 20.0
# higher-cardinality keys need more smoothing to stay stable across years
SMOOTH_BY_KEY = {"route_tod30": 50.0, "origin_tod30": 50.0, "dest_tod30": 50.0, "route_tod15": 50.0, "distb_tod30": 50.0}

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
raw_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in raw_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# baseline: treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
num_cols = [c for c in raw_cols if c not in obj_cols]
feature_cols = num_cols + cat_cols
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
KEY_COLS = ["origin", "hour", "month", "origin_hour", "dest_hour", "route_hour", "route_month", "route_tod30", "route_tod15", "origin_tod30", "dest_tod30", "distb", "distb_tod30"]
_DIST_EDGES = np.array([250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000])


def _components(df: pd.DataFrame) -> dict:
    origin = df["Origin"].astype(str).to_numpy()
    dest = df["Dest"].astype(str).to_numpy()
    dep = df["DepTime"].to_numpy(dtype=np.int64)
    dep = np.where((dep < 0) | (dep > 2400), 0, dep)
    tod30 = (((dep // 100) % 24) * 60 + (dep % 100)) // 30
    return {
        "carrier": df["UniqueCarrier"].astype(str).to_numpy(),
        "origin": origin,
        "dest": dest,
        "route": np.char.add(np.char.add(origin, "_"), dest),
        "hour": ((dep // 100) % 24).astype(str),
        "tod30": tod30.astype(str),
        "tod15": ((((dep // 100) % 24) * 60 + (dep % 100)) // 15).astype(str),
        "month": df["Month"].astype(str).to_numpy(),
        "dow": df["DayOfWeek"].astype(str).to_numpy(),
        "distb": np.digitize(df["Distance"].to_numpy(), _DIST_EDGES).astype(str),
    }


KEY_PARTS = {k: tuple(k.split("_")) for k in KEY_COLS}


def key_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Grouping keys used for target/frequency encoding."""
    comp = _components(df)
    out = {}
    for k, parts in KEY_PARTS.items():
        s = comp[parts[0]]
        for p in parts[1:]:
            s = np.char.add(np.char.add(s, "_"), comp[p])
        out[k] = s
    return pd.DataFrame(out)


def fit_stats(keys: pd.DataFrame, y: np.ndarray, rows: np.ndarray | None = None):
    """Target-mean (smoothed) and frequency tables fitted on `rows` of the training set."""
    if rows is not None:
        kf, yy = keys.iloc[rows], y[rows]
    else:
        kf, yy = keys, y
    prior = float(yy.mean())
    te, freq = {}, {}
    for c in kf.columns:
        g = pd.DataFrame({"k": kf[c].to_numpy(), "y": yy}).groupby("k")["y"].agg(["sum", "count"])
        sm = SMOOTH_BY_KEY.get(c, SMOOTH)
        te[c] = (g["sum"] + prior * sm) / (g["count"] + sm)
        freq[c] = g["count"]
    return te, freq


# slot encodings carry the signal; explicit *contrasts* between a fine view and a coarser one
# (which a tree cannot form by itself) may add more
DIFF_PAIRS = [
    ("route_tod30", "route_hour"),
    ("route_tod30", "route_month"),
    ("route_tod15", "route_tod30"),
    ("route_tod30", "origin_tod30"),
    ("origin_tod30", "origin_hour"),
    ("dest_tod30", "dest_hour"),
    ("origin_tod30", "hour"),
]


RATIO_PAIRS = [
    ("origin_tod30", "origin"),
    ("dest_tod30", "dest_hour"),
    ("route_tod30", "route_hour"),
    ("origin_tod30", "origin_hour"),
]


def encode(keys: pd.DataFrame, stats) -> pd.DataFrame:
    te, freq = stats
    out = {}
    for c in keys.columns:
        out["te_" + c] = keys[c].map(te[c]).astype(float)
        out["freq_" + c] = keys[c].map(freq[c]).fillna(0.0).astype(float)
    for a, b in DIFF_PAIRS:
        out[f"dev_{a}_{b}"] = out["te_" + a] - out["te_" + b]
    # normalised schedule-share: how much of the parent group's traffic sits in this finer bucket
    for a, b in RATIO_PAIRS:
        out[f"shr_{a}_{b}"] = (out["freq_" + a] / out["freq_" + b].clip(lower=1.0)).astype(float)
    return pd.DataFrame(out, index=keys.index)


_DAYS_BEFORE = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])


def _doy(df: pd.DataFrame) -> np.ndarray:
    """Day-of-year from the month/day columns (this dataset carries no year)."""
    m = df["Month"].astype(str).str.slice(2).astype(int).to_numpy()
    d = df["DayofMonth"].astype(str).str.slice(2).astype(int).to_numpy()
    return _DAYS_BEFORE[np.clip(m, 1, 12) - 1] + d


# US holiday travel windows, as day-of-year ranges (recur every year, so they transfer to new data)
_PEAK_WINDOWS = ((1, 3), (60, 75), (145, 152), (180, 188), (240, 247), (322, 333), (350, 365))


def calendar_frame(df: pd.DataFrame) -> pd.DataFrame:
    doy = _doy(df)
    peak = np.zeros(len(doy), dtype=np.int8)
    for lo, hi in _PEAK_WINDOWS:
        peak |= ((doy >= lo) & (doy <= hi)).astype(np.int8)
    return pd.DataFrame(
        {
            "doy": doy.astype(float),
            "holiday_peak": peak.astype(float),
            "dow_num": (df["DayOfWeek"].astype(str).str.slice(2).astype(int).to_numpy() - 1).astype(float),
            # clock-face structure of the schedule (banks, round departure times)
            "min_of_hour": (df["DepTime"].to_numpy() % 100).astype(float),
            "round5": ((df["DepTime"].to_numpy() % 100) % 5 == 0).astype(float),
            "round10": ((df["DepTime"].to_numpy() % 100) % 10 == 0).astype(float),
            "round15": ((df["DepTime"].to_numpy() % 100) % 15 == 0).astype(float),
            "round30": ((df["DepTime"].to_numpy() % 100) % 30 == 0).astype(float),
            "min_sin": np.sin(2 * np.pi * (df["DepTime"].to_numpy() % 100) / 60.0),
            "min_cos": np.cos(2 * np.pi * (df["DepTime"].to_numpy() % 100) / 60.0),
            "min_05": ((df["DepTime"].to_numpy() % 100) // 5).astype(float),
        },
        index=df.index,
    )


def base_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raw/categorical features. Only training-set-independent transforms here."""
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return pd.concat([X, calendar_frame(df)], axis=1)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = base_frame(df)
    return pd.concat([X, encode(key_frame(df), FULL_STATS)], axis=1)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- encoders: fitted on training data only ------------------------------------
y_all = to_y(train)
keys_all = key_frame(train)
FULL_STATS = fit_stats(keys_all, y_all)
ENC_COLS = list(encode(keys_all.iloc[:2], FULL_STATS).columns)

# out-of-fold encodings for the training rows, so the model does not see its own label
rng = np.random.default_rng(SEED)
fold = rng.integers(0, N_FOLDS, len(train))
oof = np.empty((len(train), len(ENC_COLS)))
for f in range(N_FOLDS):
    va = np.where(fold == f)[0]
    oof[va] = encode(keys_all.iloc[va], fit_stats(keys_all, y_all, np.where(fold != f)[0])).to_numpy()
X_train = pd.concat([base_frame(train), pd.DataFrame(oof, columns=ENC_COLS, index=train.index)], axis=1)

# --- model --------------------------------------------------------------------
BASE_PARAMS = dict(
    learning_rate=0.03,
    subsample=0.8,
    reg_lambda=20.0,
    reg_alpha=1.0,
    gamma=1.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
# a little structural diversity across the ensemble members
PARAM_SETS = [
    dict(n_estimators=600, max_depth=5, min_child_weight=30, colsample_bytree=0.7),
    dict(n_estimators=600, max_depth=6, min_child_weight=50, colsample_bytree=0.6),
    dict(n_estimators=500, max_depth=7, min_child_weight=80, colsample_bytree=0.5),
]
SEEDS = (42, 7, 2024)

t0 = time.time()
models = [
    xgb.XGBClassifier(**BASE_PARAMS, **p, random_state=s).fit(X_train, y_all)
    for p in PARAM_SETS
    for s in SEEDS
]
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
