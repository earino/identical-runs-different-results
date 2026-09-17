"""XGBoost airline delay classifier (agent-edited). See program.md for the contract.

predict_proba(df) -> P(dep_delayed_15min == 'Y'). All feature engineering lives in prepare(df)
so the hidden holdout gets identical treatment. Encoders/statistics are fitted on train only.

This run: dep_delay_rate encoding (target statistics fitted inside CV folds), raw
DepTime kept as-is; model = CV grid winner from the previous experiment.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

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
# frozen category levels, fitted on TRAIN only (unseen levels -> NaN in predict_proba)
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame, ddr_maps: dict | None = None, default_ddr: float | None = None) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    if ddr_maps is not None:
        for c, mp in ddr_maps.items():
            X["ddr_" + c] = pd.Categorical(X[c].astype(object)).map(mp).astype(float)
            X["ddr_" + c] = X["ddr_" + c].fillna(default_ddr)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def make_model(**kw):
    params = dict(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    params.update(kw)
    return xgb.XGBClassifier(**params)


DDR_COLS = ["Month", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]


def fit_ddr_maps(df: pd.DataFrame):
    y = (df[TARGET] == POSITIVE).astype(float)
    prior = float(y.mean())
    maps = {}
    for c in DDR_COLS:
        maps[c] = y.groupby(df[c]).mean().to_dict()
    return maps, prior


X_full = train  # placeholder to keep names
y_full = to_y(train)

t0 = time.time()
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
splits = list(kf.split(train, y_full))
oof = np.zeros(len(train))
for tr_idx, va_idx in splits:
    tr_df, va_df = train.iloc[tr_idx], train.iloc[va_idx]
    mp, prior = fit_ddr_maps(tr_df)
    Xtr = prepare(tr_df, mp, prior)
    Xva = prepare(va_df, mp, prior)
    m = make_model()
    m.fit(Xtr, y_full[tr_idx], verbose=False)
    oof[va_idx] = m.predict_proba(Xva)[:, 1]
oof_auc = roc_auc_score(y_full, oof)
print(f"OOF AUC (ddr encoding): {oof_auc:.4f}  time={time.time() - t0:.1f}s")

# final: fit maps on full train, refit model on full train
t0 = time.time()
mp, prior = fit_ddr_maps(train)
X_final = prepare(train, mp, prior)
model = make_model()
model.fit(X_final, y_full, verbose=False)
print(f"Final fit time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, mp, prior))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
