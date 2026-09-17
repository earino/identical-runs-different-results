"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS."""
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
SMOOTH = 20.0
N_FOLDS = 5

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
TE_COLS = list(cat_cols)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Y = to_y(train)
PRIOR = float(Y.mean())


def te_maps(df: pd.DataFrame, y: np.ndarray, k: float = SMOOTH):
    maps = {}
    for c in TE_COLS:
        g = pd.DataFrame({"c": df[c].astype(str).to_numpy(), "y": y}).groupby("c")["y"].agg(["sum", "count"])
        maps[c] = (g["sum"] + PRIOR * k) / (g["count"] + k)
    return maps


FULL_MAPS = te_maps(train, Y)

MONTH_START = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype="float32")


def add_calendar(X: pd.DataFrame, df: pd.DataFrame) -> None:
    mon = pd.to_numeric(df["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dom = pd.to_numeric(df["DayofMonth"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    doy = dom + MONTH_START[(mon - 1).fillna(0).clip(0, 11).astype(int)]
    dep = pd.to_numeric(df["DepTime"], errors="coerce").astype("float32")
    mins = ((dep // 100).clip(0, 23) * 60 + (dep % 100).clip(0, 59)).astype("float32")
    X["doy"] = doy.astype("float32")
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.0).astype("float32")
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.0).astype("float32")
    X["tod_sin"] = np.sin(2 * np.pi * mins / 1440.0).astype("float32")
    X["tod_cos"] = np.cos(2 * np.pi * mins / 1440.0).astype("float32")
    X["is_weekend"] = df["DayOfWeek"].astype(str).isin(["c-1", "c-7"]).astype("int8").to_numpy()


def encode(df: pd.DataFrame, maps) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    for c in TE_COLS:
        X["te_" + c] = df[c].astype(str).map(maps[c]).fillna(PRIOR).astype("float32")
    add_calendar(X, df)
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    return encode(df, FULL_MAPS)


# out-of-fold target encoding for the training matrix (avoids leakage into the fit)
def oof_te(df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    oof = pd.DataFrame(index=df.index)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in kf.split(df):
        m = te_maps(df.iloc[tr_idx], y[tr_idx])
        sub = df.iloc[va_idx]
        for c in TE_COLS:
            oof.loc[sub.index, "te_" + c] = sub[c].astype(str).map(m[c]).fillna(PRIOR).to_numpy()
    return oof


def build_train_matrix() -> pd.DataFrame:
    X = train[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    oof = oof_te(train, Y)
    for c in TE_COLS:
        X["te_" + c] = oof["te_" + c].astype("float32").values
    add_calendar(X, train)
    return X


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(build_train_matrix(), Y)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
