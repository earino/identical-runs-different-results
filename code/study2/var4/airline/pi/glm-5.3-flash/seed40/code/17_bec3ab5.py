"""XGBoost — experiment 25: fold-bagged blend members; DepTime5-cat and recency-weight variants."""
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


GROUPS = {
    "BASE": ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "DepTime", "Distance",
             "dep_min", "dep_sin", "dep_cos", "DepTime_mod", "hour", "hb", "doy", "doy_sin", "doy_cos", "dow", "log_dist"],
    "RATES1": ["r_month", "r_carrier", "r_origin", "r_dest", "r_hb"],
    "PROD": ["p_sd", "p_cd", "p_hd", "p_sD", "p_cD", "p_sDc", "p_cDs"],
    "DT5": ["dt5"],
}
COLS = [c for g in ["BASE", "RATES1"] for c in GROUPS[g]]
COLS_PROD = COLS + GROUPS["PROD"]
COLS_DT5 = COLS_PROD + GROUPS["DT5"]

X_full, y_all = prepare(train), to_y(train)
X_ev_full, y_ev = prepare(evald), to_y(evald)

rng = np.random.default_rng(SEED)
_idx = rng.permutation(len(train))
_FOLDS = np.array_split(_idx, 5)


def fit(cfg, cols, seed=SEED, rows=None, w=None):
    m = xgb.XGBClassifier(
        tree_method="hist", enable_categorical=True, random_state=seed, n_jobs=N_JOBS,
        eval_metric="auc", **cfg,
    )
    Xr = X_full[cols] if rows is None else X_full[cols].iloc[rows]
    yr = y_all if rows is None else y_all[rows]
    m.fit(Xr, yr, sample_weight=w)
    return m


def evalp(p):
    return roc_auc_score(y_ev, p)


def foldbag(cfg, cols, seed=SEED, w_fn=None):
    P = np.zeros(len(X_ev_full))
    for f, va in enumerate(_FOLDS):
        tr = np.setdiff1d(_idx, va)
        w = w_fn(tr) if w_fn else None
        mm = xgb.XGBClassifier(
            tree_method="hist", enable_categorical=True, random_state=seed + f, n_jobs=N_JOBS,
            eval_metric="auc", **cfg,
        )
        mm.fit(X_full[cols].iloc[tr], y_all[tr], sample_weight=w)
        P += mm.predict_proba(X_ev_full[cols])[:, 1] / 5
    return P


t00 = time.time()
P = {}
BASE_PRUNE = {
    "ref": [],
    "no_dow": ["dow"],
    "no_doy": ["doy"],
    "no_dtmod": ["DepTime_mod"],
    "no_hour": ["hour"],
    "no_depmin": ["dep_min"],
    "no_dt": ["DepTime"],
    "no_domcat": ["DayofMonth"],
    "no_dt_hb": ["DepTime", "hb"],
    "no_dow_doy": ["dow", "doy"],
}

def cs3_cfg():
    return dict(n_estimators=400, max_depth=4, learning_rate=0.05, colsample_bytree=0.3)

for name, drop in BASE_PRUNE.items():
    cols = [c for c in COLS_PROD if c not in drop]
    mdl = fit(cs3_cfg(), cols, seed=7)
    p = mdl.predict_proba(X_ev_full[cols])[:, 1]
    P[name] = p
    print(f"Eval AUC: {evalp(p):.4f}   cs3bp-{name}")

best_single = max(P, key=lambda k: evalp(P[k]))
print(f"best single prune: {best_single}")

# foldbag the best pruned variant and compare blends
COLS_BEST = [c for c in COLS_PROD if c not in BASE_PRUNE[best_single]]
Pfb = {"cs3bp": foldbag(cs3_cfg(), COLS_PROD, 7), "best": foldbag(cs3_cfg(), COLS_BEST, 7)}
Pfb["d4g"] = foldbag(dict(n_estimators=800, max_depth=4, learning_rate=0.03, gamma=2.0), COLS, SEED)
Pfb["forest"] = foldbag(dict(n_estimators=1, num_parallel_tree=20, learning_rate=1.0, subsample=0.8, colsample_bynode=0.8, max_depth=4), COLS_PROD, 99)
for k, v in Pfb.items():
    print(f"Eval AUC: {evalp(v):.4f}   fb-{k}")
mdl5 = fit(cs3_cfg(), COLS_DT5, seed=7)
P5 = mdl5.predict_proba(X_ev_full[COLS_DT5])[:, 1]
print(f"Eval AUC: {evalp(P5):.4f}   cs3dt5")

CANDS = {
    "fb4ref": ["cs3bp", "d4g", "forest", "cs3dt5"],
    "fb4best": ["best", "d4g", "forest", "cs3dt5"],
    "fb5mix": ["best", "cs3bp", "d4g", "forest", "cs3dt5"],
}
best_name, best_a, best_set = None, -1.0, None
for cname, names in CANDS.items():
    a = evalp(np.mean([Pfb[n] if n in Pfb else P5 for n in names], axis=0))
    print(f"Eval AUC: {a:.4f}   {cname}")
    if a > best_a:
        best_name, best_a, best_set = cname, a, names
print(f"best: {best_name}  total {time.time()-t00:.0f}s")
print(f"Eval AUC: {best_a:.4f}")

_fb_cfg = {"cs3bp": cs3_cfg(), "best": cs3_cfg(),
           "d4g": dict(n_estimators=800, max_depth=4, learning_rate=0.03, gamma=2.0),
           "forest": dict(n_estimators=1, num_parallel_tree=20, learning_rate=1.0, subsample=0.8, colsample_bynode=0.8, max_depth=4)}
_fb_cols = {"cs3bp": COLS_PROD, "best": COLS_BEST, "d4g": COLS, "forest": COLS_PROD}
_fb_seed = {"cs3bp": 7, "best": 7, "d4g": SEED, "forest": 99}
_single = {"cs3dt5": (mdl5, COLS_DT5)} if "cs3dt5" in best_set else {}


class Bag:
    def __init__(self, bag_names, singles):
        self.bag_names, self.singles = bag_names, singles
        self.fb = {n: [] for n in bag_names}
        for n in bag_names:
            for f, va in enumerate(_FOLDS):
                tr = np.setdiff1d(_idx, va)
                mm = xgb.XGBClassifier(
                    tree_method="hist", enable_categorical=True, random_state=_fb_seed[n] + f,
                    n_jobs=N_JOBS, eval_metric="auc", **_fb_cfg[n],
                )
                mm.fit(X_full[_fb_cols[n]].iloc[tr], y_all[tr])
                self.fb[n].append((mm, _fb_cols[n]))

    def predict_proba(self, X):
        acc = np.zeros(len(X))
        cnt = 0
        for n in self.bag_names:
            p = np.zeros(len(X))
            for mm, cols in self.fb[n]:
                p += mm.predict_proba(X[cols])[:, 1] / len(self.fb[n])
            acc += p
            cnt += 1
        for name, (mdl, cols) in self.singles.items():
            acc += mdl.predict_proba(X[cols])[:, 1]
            cnt += 1
        out = acc / cnt
        return np.column_stack([1 - out, out])


model = Bag([n for n in best_set if n in _fb_cfg], {n: _single[n] for n in best_set if n in _single})


def predict_proba(self, X):
        acc = np.zeros(len(X))
        cnt = 0
        for n in self.bag_names:
            p = np.zeros(len(X))
            for mm, cols in self.fb[n]:
                p += mm.predict_proba(X[cols])[:, 1] / 5
            acc += p
            cnt += 1
        for name, (mdl, cols) in self.singles.items():
            acc += mdl.predict_proba(X[cols])[:, 1]
            cnt += 1
        out = acc / cnt
        return np.column_stack([1 - out, out])


model = Bag([n for n in best_set if n in _fb_cfg], {n: _single[n] for n in best_set if n in _single})


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]
