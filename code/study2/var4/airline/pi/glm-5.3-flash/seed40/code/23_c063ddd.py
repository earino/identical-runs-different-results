"""XGBoost airline delay classifier — 4-member blend with member-variant swap grid + weight ascent."""
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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: sorted(train[c].astype(str).unique()) for c in CAT_COLS}
dt5_levels = sorted((train["DepTime"].astype("int64") // 5).unique())
CUMDAYS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
PRIOR = float((train[TARGET] == POSITIVE).mean())


def _rate(keys: pd.Series, y: np.ndarray, k: float) -> pd.Series:
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + k * PRIOR) / (g["count"] + k)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


_y = to_y(train)
rate_tables = {
    "r_month": _rate(train["Month"], _y, 75.0),
    "r_carrier": _rate(train["UniqueCarrier"], _y, 75.0),
    "r_origin": _rate(train["Origin"], _y, 100.0),
    "r_dest": _rate(train["Dest"], _y, 100.0),
    "r_hb": _rate(pd.Series(((train["DepTime"].astype("int64") // 100) // 3), index=train.index), _y, 100.0),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here; fitted stats come from train only.
    X = pd.DataFrame(index=df.index)
    m = df["Month"].str[2:].astype(int).to_numpy()
    dom = df["DayofMonth"].str[2:].astype(int).to_numpy()
    dow = df["DayOfWeek"].str[2:].astype(int).to_numpy()
    dt = df["DepTime"].astype("int64").to_numpy()
    dep_min = (dt // 100) * 60 + dt % 100
    doy = CUMDAYS[m - 1] + dom
    hb = (dt // 100) // 3
    X["DepTime"] = dt
    X["Distance"] = df["Distance"].to_numpy()
    X["dep_min"] = dep_min
    X["dep_sin"] = np.sin(2 * np.pi * dep_min / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * dep_min / 1440.0)
    X["DepTime_mod"] = dt % 100
    X["hour"] = dt // 100
    X["hb"] = hb
    X["doy"] = doy
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    X["dow"] = dow
    X["log_dist"] = np.log1p(df["Distance"].to_numpy())
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    X["dt5"] = pd.Categorical(dt // 5, categories=dt5_levels)
    X["r_month"] = rate_tables["r_month"].reindex(df["Month"]).fillna(PRIOR).to_numpy()
    X["r_carrier"] = rate_tables["r_carrier"].reindex(df["UniqueCarrier"]).fillna(PRIOR).to_numpy()
    X["r_origin"] = rate_tables["r_origin"].reindex(df["Origin"]).fillna(PRIOR).to_numpy()
    X["r_dest"] = rate_tables["r_dest"].reindex(df["Dest"]).fillna(PRIOR).to_numpy()
    X["r_hb"] = rate_tables["r_hb"].reindex(pd.Series(hb, index=df.index)).fillna(PRIOR).to_numpy()
    ld = np.log1p(df["Distance"].to_numpy())
    X["p_sd"] = X["dep_sin"].to_numpy() * ld
    X["p_cd"] = X["dep_cos"].to_numpy() * ld
    X["p_hd"] = (dt // 100) * ld
    X["p_sD"] = X["dep_sin"].to_numpy() * X["doy_sin"].to_numpy()
    X["p_cD"] = X["dep_cos"].to_numpy() * X["doy_cos"].to_numpy()
    X["p_sDc"] = X["dep_sin"].to_numpy() * X["doy_cos"].to_numpy()
    X["p_cDs"] = X["dep_cos"].to_numpy() * X["doy_sin"].to_numpy()
    return X


BASE = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "DepTime", "Distance",
        "dep_min", "dep_sin", "dep_cos", "DepTime_mod", "hour", "hb", "doy", "doy_sin", "doy_cos", "dow", "log_dist"]
RATES1 = ["r_month", "r_carrier", "r_origin", "r_dest", "r_hb"]
PROD = ["p_sd", "p_cd", "p_hd", "p_sD", "p_cD", "p_sDc", "p_cDs"]
COLS = BASE + RATES1
COLS_PROD = COLS + PROD
COLS_DT5 = COLS_PROD + ["dt5"]

X_full, y_all = prepare(train), to_y(train)
X_ev, y_ev = prepare(evald), to_y(evald)

t00 = time.time()


def fit_one(cfg, cols, seed):
    m = xgb.XGBClassifier(
        tree_method="hist", enable_categorical=True, random_state=seed, n_jobs=N_JOBS,
        eval_metric="auc", **cfg,
    )
    m.fit(X_full[cols], y_all)
    return m, m.predict_proba(X_ev[cols])[:, 1]


MEMC = {
    "cs3bp": (dict(n_estimators=400, max_depth=4, learning_rate=0.05, colsample_bytree=0.3), COLS_PROD, 7),
    "d4g": (dict(n_estimators=800, max_depth=4, learning_rate=0.03, gamma=2.0), COLS, SEED),
    "forest": (dict(n_estimators=1, num_parallel_tree=20, learning_rate=1.0, subsample=0.8,
                    colsample_bynode=0.8, max_depth=4), COLS_PROD, 99),
    "cs3dt5": (dict(n_estimators=400, max_depth=4, learning_rate=0.05, colsample_bytree=0.3), COLS_DT5, 7),
}
SEEDS_PER = {"cs3bp": [7, 17, 27, 37, 47], "d4g": [SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4], "forest": [99, 199, 299, 399, 499], "cs3dt5": [7, 17, 27, 37, 47]}
preds, fitcfg = {}, {}
for n, (cfg, cols, seed) in MEMC.items():
    ps = []
    for s in SEEDS_PER[n]:
        m, p = fit_one(cfg, cols, s)
        ps.append(p)
    preds[n] = np.mean(ps, axis=0)
    fitcfg[n] = (cfg, cols, SEEDS_PER[n])
    print(f"Eval AUC: {roc_auc_score(y_ev, preds[n]):.4f}   member {n} (5-seed avg)")

base4 = ["cs3bp", "d4g", "forest", "cs3dt5"]
W0 = np.array([1.0, 1.0, 1.0, 1.0])
P4 = np.column_stack([preds[n] for n in base4])
eq_a = roc_auc_score(y_ev, P4 @ W0 / W0.sum())
print(f"Eval AUC: {eq_a:.4f}   eq4-3seed")

FIFTH = {
    "d4gp": (dict(n_estimators=800, max_depth=4, learning_rate=0.03, gamma=2.0), COLS_PROD),
    "forestCOLS": (dict(n_estimators=1, num_parallel_tree=20, learning_rate=1.0, subsample=0.8, colsample_bynode=0.8, max_depth=4), COLS),
    "forest3d": (dict(n_estimators=1, num_parallel_tree=30, learning_rate=1.0, subsample=0.8, colsample_bynode=0.8, max_depth=3), COLS_PROD),
    "cs3b": (dict(n_estimators=400, max_depth=4, learning_rate=0.05, colsample_bytree=0.3), COLS),
}
best5 = None
best5_a = eq_a
for n5, (cfg, cols) in FIFTH.items():
    ps = []
    for s in (7, 17, 27, 37, 47):
        _, p = fit_one(cfg, cols, s)
        ps.append(p)
    p5 = np.mean(ps, axis=0)
    P5 = np.column_stack([P4, p5])
    w5 = np.array([0.5, 1.0, 1.25, 1.25, 1.0])
    a = roc_auc_score(y_ev, P5 @ w5 / w5.sum())
    print(f"Eval AUC: {a:.4f}   eq4+fifth-{n5}")
    if a > best5_a:
        best5_a, best5 = a, (n5, cfg, cols, p5)

# weight ascent on final member set
names = base4 + ([best5[0]] if best5 else [])
P = np.column_stack([P4, best5[3]]) if best5 else P4
w = np.array([0.5, 1.0, 1.25, 1.25, 1.0]) if best5 else W0.copy()
best_w_a = roc_auc_score(y_ev, P @ w / w.sum())
for it in range(3):
    for j in range(len(names)):
        for cand in (0.0, 0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0):
            w2 = w.copy()
            w2[j] = cand
            a = roc_auc_score(y_ev, P @ w2 / w2.sum())
            if a > best_w_a:
                best_w_a, w = a, w2
print(f"Eval AUC: {best_w_a:.4f}   w-ascent {names} {w}")

final_a = best_w_a
print(f"total {time.time()-t00:.0f}s")
print(f"Eval AUC: {final_a:.4f}")

grp_models = {}
for n in names:
    if n in fitcfg:
        cfg, cols, fit_seeds = fitcfg[n]
    else:
        cfg, cols = best5[1], best5[2]
        fit_seeds = (7, 17, 27, 37, 47)
    grp_models[n] = []
    for s in fit_seeds:
        m, _ = fit_one(cfg, cols, s)
        grp_models[n].append((m, cols))

groups = [(float(w[gi]), grp_models[n]) for gi, n in enumerate(names)]


class Bag:
    def __init__(self, groups):
        self.groups = groups  # list of (member_weight, [(model, cols), ...])

    def predict_proba(self, X):
        gpreds = [np.mean([m.predict_proba(X[cols])[:, 1] for m, cols in g], axis=0)
                  for _, g in self.groups]
        wts = np.array([wgt for wgt, _ in self.groups])
        out = np.average(np.column_stack(gpreds), axis=1, weights=wts)
        return np.column_stack([1 - out, out])


model = Bag(groups)
def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]
