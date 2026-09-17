"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     ALL feature engineering lives inside prepare() (target-encoding maps are fitted on train only, at module
     level, and are applied inside prepare() to whatever dataframe comes in).
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
TE_M = 50  # smoothing count for target encoding
N_FOLDS = 5

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
ytr = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(ytr.mean())


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def _cnum(s: pd.Series) -> pd.Series:
    """'c-7' -> 7 (float, NaN stays NaN)."""
    return s.astype(str).str.extract(r"(\d+)")[0].astype(float)


def _hour(df: pd.DataFrame) -> pd.Series:
    return (df["DepTime"].astype(float) // 100).clip(0, 24).astype(int)


def _keys_hour(df):          return _hour(df).astype(str)
def _keys_origin_hour(df):   return df["Origin"].astype(str) + "_" + _keys_hour(df)
def _keys_route_hour(df):    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str) + "_" + _keys_hour(df)
def _keys_dest_hour(df):     return df["Dest"].astype(str) + "_" + _keys_hour(df)
def _keys_carrier_hour(df):  return df["UniqueCarrier"].astype(str) + "_" + _keys_hour(df)


TE_SPECS = {
    "te_hour": _keys_hour,
    "te_origin_hour": _keys_origin_hour,
    "te_route_hour": _keys_route_hour,
    "te_dest_hour": _keys_dest_hour,
    "te_carrier_hour": _keys_carrier_hour,
}

# --- fit encoders on TRAIN ONLY ------------------------------------------------


def _smooth(keys: np.ndarray, y: np.ndarray, m: float, prior_per_cell=None) -> pd.Series:
    g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    if prior_per_cell is None:
        pr = pd.Series(PRIOR, index=g.index)
    else:
        pr = pd.Series(prior_per_cell).groupby(level=0).first().reindex(g.index)
    return (g["sum"] + m * pr) / (g["count"] + m)


TE_MAPS = {}
OOF_TE = {}
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
for name, keyfn in TE_SPECS.items():
    keys = keyfn(train).to_numpy()
    TE_MAPS[name] = _smooth(keys, ytr, TE_M)
    oof = np.full(len(keys), PRIOR)
    for tri, tei in kf.split(keys):
        oof[tei] = pd.Series(keys[tei]).map(_smooth(keys[tri], ytr[tri], TE_M)).fillna(PRIOR).to_numpy()
    OOF_TE[name] = oof

# hierarchical route x hour TE: shrink the cell mean toward the hour-level mean,
# i.e. encode each route's deviation from the global daily delay curve (more year-robust).
_h_keys = _keys_route_hour(train).to_numpy()
_h_hours = _keys_hour(train).to_numpy()
_g_hour = pd.DataFrame({"h": _h_hours, "y": ytr}).groupby("h")["y"].agg(["sum", "count"])
HOUR_PRIOR = ((_g_hour["sum"] + 20 * PRIOR) / (_g_hour["count"] + 20))
_hr_prior = pd.Series(_h_keys).groupby(_h_keys).first().map(HOUR_PRIOR)
_g_rh = pd.DataFrame({"k": _h_keys, "y": ytr}).groupby("k")["y"].agg(["sum", "count"])
H_ROUTE_HOUR = (_g_rh["sum"] + TE_M * _hr_prior.reindex(_g_rh.index).fillna(PRIOR)) / (_g_rh["count"] + TE_M)
H_OOF = np.full(len(_h_keys), PRIOR)
for tri, tei in kf.split(_h_keys):
    gh = pd.DataFrame({"h": _h_hours[tri], "y": ytr[tri]}).groupby("h")["y"].agg(["sum", "count"])
    hp = (gh["sum"] + 20 * PRIOR) / (gh["count"] + 20)
    pr = pd.Series(_h_keys[tri]).groupby(_h_keys[tri]).first().map(hp).fillna(PRIOR)
    gg = pd.DataFrame({"k": _h_keys[tri], "y": ytr[tri]}).groupby("k")["y"].agg(["sum", "count"])
    mp = (gg["sum"] + TE_M * pr.reindex(gg.index).fillna(PRIOR)) / (gg["count"] + TE_M)
    H_OOF[tei] = pd.Series(_h_keys[tei]).map(mp).fillna(PRIOR).to_numpy()

# hierarchical origin x carrier x hour TE (sparse cells shrunk toward the hour-level mean)
_oc_keys = (train["Origin"].astype(str) + "_" + train["UniqueCarrier"].astype(str) + "_" + _keys_hour(train)).to_numpy()
_oc_prior = pd.Series(_oc_keys).groupby(_oc_keys).first().map(HOUR_PRIOR)
_g_oc = pd.DataFrame({"k": _oc_keys, "y": ytr}).groupby("k")["y"].agg(["sum", "count"])
H_OCARRIER_HOUR = (_g_oc["sum"] + TE_M * _oc_prior.reindex(_g_oc.index).fillna(PRIOR)) / (_g_oc["count"] + TE_M)
_OC_OOF = np.full(len(_oc_keys), PRIOR)
for tri, tei in kf.split(_oc_keys):
    pr = pd.Series(_oc_keys[tri]).groupby(_oc_keys[tri]).first().map(HOUR_PRIOR).fillna(PRIOR)
    gg = pd.DataFrame({"k": _oc_keys[tri], "y": ytr[tri]}).groupby("k")["y"].agg(["sum", "count"])
    mp = (gg["sum"] + TE_M * pr.reindex(gg.index).fillna(PRIOR)) / (gg["count"] + TE_M)
    _OC_OOF[tei] = pd.Series(_oc_keys[tei]).map(mp).fillna(PRIOR).to_numpy()

# --- categorical levels from train only ----------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Month", "DayofMonth", "DayOfWeek", "Route"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS if c != "Route"}
CAT_LEVELS["Route"] = pd.Index(sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(float)
    hour = (dep // 100).clip(0, 24)
    tod = hour + (dep % 100) / 60.0
    X["hour"] = hour
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)
    X["minute"] = dep % 100
    X["month"], X["day"], X["dow"] = _cnum(df["Month"]), _cnum(df["DayofMonth"]), _cnum(df["DayOfWeek"])
    X["distance"] = df["Distance"].astype(float)
    X["Route"] = (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).values
    for c in CAT_COLS:
        vals = X[c] if c == "Route" else df[c].values
        X[c] = pd.Categorical(vals, categories=CAT_LEVELS[c])  # unseen levels -> NaN
    for name, keyfn in TE_SPECS.items():
        X[name] = keyfn(df).map(TE_MAPS[name]).astype(float).fillna(PRIOR).to_numpy()
    cells = _keys_route_hour(df)
    X["h_route_hour"] = (cells.map(H_ROUTE_HOUR).astype(float)
                          .fillna(_keys_hour(df).map(HOUR_PRIOR).astype(float))
                          .fillna(PRIOR).to_numpy())
    oc_cells = (df["Origin"].astype(str) + "_" + df["UniqueCarrier"].astype(str) + "_" + _keys_hour(df))
    X["h_ocarrier_hour"] = (oc_cells.map(H_OCARRIER_HOUR).astype(float)
                             .fillna(_keys_hour(df).map(HOUR_PRIOR).astype(float))
                             .fillna(PRIOR).to_numpy())
    return X


def prepare_train(df: pd.DataFrame) -> pd.DataFrame:
    """Training frame: same as prepare() but with out-of-fold TE values (no self-leakage)."""
    X = prepare(df)
    for name in OOF_TE:
        X[name] = OOF_TE[name]
    X["h_route_hour"] = H_OOF
    X["h_ocarrier_hour"] = _OC_OOF
    return X


# --- model --------------------------------------------------------------------
# two decorrelated XGB models; their probabilities are averaged (contract-safe ensemble)
CFG_A = dict(max_depth=5, learning_rate=0.03, min_child_weight=1, colsample_bytree=0.8,
             reg_alpha=10, max_bin=1024, random_state=42)
CFG_B = dict(max_depth=4, learning_rate=0.04, min_child_weight=2, colsample_bytree=0.7,
             reg_alpha=16, max_bin=512, random_state=7)
CFG_C = dict(max_depth=5, learning_rate=0.035, min_child_weight=1, colsample_bytree=0.85,
             reg_alpha=8, max_bin=512, random_state=2024)
models = []
t0 = time.time()
_month_num = train["Month"].astype(str).str.extract(r"(\d+)")[0].astype(int).to_numpy()
_sample_w = 1 + ((_month_num - 1) / 11.0) ** 2  # upweight late-2005 rows (closer to 2006)
X_train_prepared = prepare_train(train)
X_eval_prepared = prepare(evald)
y_eval = to_y(evald)
for cfg in (CFG_A, CFG_B, CFG_C):
    m = xgb.XGBClassifier(
        n_estimators=4000,
        tree_method="hist",
        enable_categorical=True,
        n_jobs=N_JOBS,
        eval_metric="auc",
        early_stopping_rounds=100,
        **cfg,
    )
    m.fit(X_train_prepared, ytr, sample_weight=_sample_w,
          eval_set=[(X_eval_prepared, y_eval)], verbose=False)
    models.append(m)
    print(f"  cfg a={cfg['reg_alpha']} d={cfg['max_depth']}: best_iter={m.best_iteration} val={m.best_score:.4f}")
print(f"Training time: {time.time() - t0:.1f}s  ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
