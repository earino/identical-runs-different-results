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
SMOOTH = 20.0
N_FOLDS = 5

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


def key_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Grouping keys used for target/frequency encoding."""
    origin = df["Origin"].astype(str).to_numpy()
    dest = df["Dest"].astype(str).to_numpy()
    return pd.DataFrame(
        {
            "carrier": df["UniqueCarrier"].astype(str).to_numpy(),
            "origin": origin,
            "dest": dest,
            "route": np.char.add(np.char.add(origin, "_"), dest),
        }
    )


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
        te[c] = (g["sum"] + prior * SMOOTH) / (g["count"] + SMOOTH)
        freq[c] = g["count"]
    return te, freq


def encode(keys: pd.DataFrame, stats) -> pd.DataFrame:
    te, freq = stats
    out = {}
    for c in keys.columns:
        out["te_" + c] = keys[c].map(te[c]).astype(float)
        out["freq_" + c] = keys[c].map(freq[c]).fillna(0.0).astype(float)
    return pd.DataFrame(out, index=keys.index)


def base_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raw/categorical features. Only training-set-independent transforms here."""
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


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

# out-of-fold encodings for the training rows, so the model does not see its own label
rng = np.random.default_rng(SEED)
fold = rng.integers(0, N_FOLDS, len(train))
oof = np.empty((len(train), 2 * keys_all.shape[1]))
for f in range(N_FOLDS):
    tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
    oof[va] = encode(keys_all.iloc[va], fit_stats(keys_all, y_all, tr)).to_numpy()
X_train = pd.concat(
    [base_frame(train), pd.DataFrame(oof, columns=encode(keys_all.iloc[:1], FULL_STATS).columns, index=train.index)],
    axis=1,
)

# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(X_train, y_all)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
