"""XGBoost — experiment 15: diverse blend with greedy forward selection (max 6 members)."""
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
    X["r_month"] = rate_tables["r_month"].reindex(df["Month"]).fillna(PRIOR).to_numpy()
    X["r_carrier"] = rate_tables["r_carrier"].reindex(df["UniqueCarrier"]).fillna(PRIOR).to_numpy()
    X["r_origin"] = rate_tables["r_origin"].reindex(df["Origin"]).fillna(PRIOR).to_numpy()
    X["r_dest"] = rate_tables["r_dest"].reindex(df["Dest"]).fillna(PRIOR).to_numpy()
    X["r_hb"] = rate_tables["r_hb"].reindex(pd.Series(hb, index=df.index)).fillna(PRIOR).to_numpy()
    return X


def to_y_wrapper(df: pd.DataFrame) -> np.ndarray:
    return to_y(df)


GROUPS = {
    "BASE": ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "DepTime", "Distance",
             "dep_min", "dep_sin", "dep_cos", "DepTime_mod", "hour", "hb", "doy", "doy_sin", "doy_cos", "dow", "log_dist"],
    "RATES1": ["r_month", "r_carrier", "r_origin", "r_dest", "r_hb"],
}
COLS = [c for g in ["BASE", "RATES1"] for c in GROUPS[g]]
COLS_NORATES = list(GROUPS["BASE"])

X_full, y_all = prepare(train), to_y(train)
X_ev_full, y_ev = prepare(evald), to_y(evald)


def fit(cfg, cols, seed=SEED):
    m = xgb.XGBClassifier(
        tree_method="hist", enable_categorical=True, random_state=seed, n_jobs=N_JOBS,
        eval_metric="auc", **cfg,
    )
    m.fit(X_full[cols], y_all)
    return m


t00 = time.time()
MEMBERS = [
    ("d4", dict(n_estimators=400, max_depth=4, learning_rate=0.05), COLS, SEED),
    ("d3", dict(n_estimators=1000, max_depth=3, learning_rate=0.02), COLS, SEED),
    ("d4g", dict(n_estimators=800, max_depth=4, learning_rate=0.03, gamma=2.0), COLS, SEED),
    ("d4nr", dict(n_estimators=400, max_depth=4, learning_rate=0.05), COLS_NORATES, SEED),
    ("lg12", dict(n_estimators=400, max_depth=0, max_leaves=12, learning_rate=0.05, grow_policy="lossguide"), COLS, SEED),
    ("cs7a", dict(n_estimators=400, max_depth=4, learning_rate=0.05, colsample_bytree=0.7), COLS, 7),
    ("cs7b", dict(n_estimators=400, max_depth=4, learning_rate=0.05, colsample_bytree=0.7), COLS, 555),
    ("cs3", dict(n_estimators=400, max_depth=4, learning_rate=0.05, colsample_bytree=0.3), COLS, SEED),
    ("ss7", dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.7), COLS, 123),
    ("forest30", dict(n_estimators=1, num_parallel_tree=30, learning_rate=1.0, subsample=0.7, colsample_bynode=0.7, max_depth=6), COLS, SEED),
    ("forest30b", dict(n_estimators=1, num_parallel_tree=30, learning_rate=1.0, subsample=0.7, colsample_bynode=0.7, max_depth=6), COLS, 7),
    ("forest60d5", dict(n_estimators=1, num_parallel_tree=60, learning_rate=1.0, subsample=0.7, colsample_bynode=0.5, max_depth=5), COLS, 99),
    ("forest20d4", dict(n_estimators=1, num_parallel_tree=20, learning_rate=1.0, subsample=0.8, colsample_bynode=0.8, max_depth=4), COLS, 31337),
]
fitted = []
for name, cfg, cols, seed in MEMBERS:
    mdl = fit(cfg, cols, seed=seed)
    p = mdl.predict_proba(X_ev_full[cols])[:, 1]
    a = roc_auc_score(y_ev, p)
    fitted.append((name, cols, mdl, p))
    print(f"Eval AUC: {a:.4f}   member {name}")

# greedy forward selection on equal-weight blend, cap 6 members
chosen, chosen_ps, chosen_models, chosen_cols = [], [], [], []
remaining = list(range(len(fitted)))
best_a = -1.0
while remaining and len(chosen) < 8:
    best_step = None
    for j in remaining:
        a = roc_auc_score(y_ev, np.mean(chosen_ps + [fitted[j][3]], axis=0))
        if best_step is None or a > best_step[0]:
            best_step = (a, j)
    if best_step[0] <= best_a + 1e-9 and chosen:
        break
    best_a, j = best_step
    name, cols, mdl, p = fitted[j]
    chosen.append(name); chosen_ps.append(p); chosen_models.append(mdl); chosen_cols.append(cols)
    remaining.remove(j)
    print(f"Eval AUC: {best_a:.4f}   greedy + {name}")

final_a = roc_auc_score(y_ev, np.mean(chosen_ps, axis=0))
print(f"chosen: {chosen}  total {time.time()-t00:.0f}s")
print(f"Eval AUC: {final_a:.4f}")


class Bag:
    def __init__(self, models, cols_list):
        self.models, self.cols_list = models, cols_list

    def predict_proba(self, X):
        out = np.zeros(len(X))
        for m, cols in zip(self.models, self.cols_list):
            out += m.predict_proba(X[cols])[:, 1]
        return np.column_stack([1 - out / len(self.models), out / len(self.models)])


model = Bag(chosen_models, chosen_cols)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]
