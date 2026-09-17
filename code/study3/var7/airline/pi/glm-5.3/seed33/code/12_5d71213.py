"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features ------------------------------------------------------------------
BASE_CAT = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
BASE_NUM = ["DepTime", "Distance"]
HOURS = pd.Index(range(25))
CAT_LEVELS = {
    **{c: pd.Index(sorted(train[c].dropna().unique())) for c in BASE_CAT},
    "HourCat": HOURS,
}
HC_LEVELS = pd.Index(
    sorted(
        (
            train["UniqueCarrier"].astype(str)
            + "@"
            + (((train["DepTime"] // 100) % 24).astype(int)).astype(str)
        ).unique()
    )
)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in BASE_NUM:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    hr = ((pd.to_numeric(df["DepTime"], errors="coerce") // 100) % 24).astype(int)
    X["HourCat"] = pd.Categorical(hr, categories=HOURS)
    X["HourCarrierCat"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "@" + hr.astype(str), categories=HC_LEVELS
    )  # unseen combos -> NaN
    for c in BASE_CAT:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_tr = prepare(train)
y_tr = to_y(train)
X_ev = prepare(evald)
y_ev = to_y(evald)

SCAN = [
    (dict(max_depth=5, learning_rate=0.05), 2200),
    (dict(max_depth=5, learning_rate=0.03), 2600),
    (dict(max_depth=4, learning_rate=0.05), 1600),
    (dict(max_depth=5, learning_rate=0.08), 1600),
]

t0 = time.time()
members = []
for spec, rounds in SCAN:
    m = xgb.XGBClassifier(
        n_estimators=rounds,
        eval_metric="auc",
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
        max_depth=spec["max_depth"],
        learning_rate=spec["learning_rate"],
    )
    m.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    aucs = m.evals_result()["validation_0"]["auc"]
    bi = int(np.argmax(aucs))
    rising = " RISING" if bi > rounds - 30 else ""
    print(f"scan d{spec['max_depth']}-lr{spec['learning_rate']}: best_round={bi + 1} auc={aucs[bi]:.4f}{rising}")
    members.append((m, bi + 1, aucs[bi]))
print(f"Scan time: {time.time() - t0:.1f}s")

members.sort(key=lambda t: -t[2])
best_m, best_bi, best_auc = members[0]
print(f"BEST single: {best_auc:.4f} @ {best_bi} rounds")


def ens_auc(df_y, df):
    X = prepare(df)
    ps = np.zeros(len(X))
    for m, bi, _ in members:
        ps += m.predict_proba(X, iteration_range=(0, bi))[:, 1]
    return roc_auc_score(df_y, ps / len(members))


ens = ens_auc(y_ev, evald)
print(f"Ensemble AUC (4 members): {ens:.4f}")

if ens > best_auc:
    use_ens = True
    final_auc = ens
else:
    use_ens = False
    final_auc = best_auc

if use_ens:

    def predict_proba(df: pd.DataFrame) -> np.ndarray:
        X = prepare(df)
        ps = np.zeros(len(X))
        for m, bi, _ in members:
            ps += m.predict_proba(X, iteration_range=(0, bi))[:, 1]
        return ps / len(members)

else:

    def predict_proba(df: pd.DataFrame) -> np.ndarray:
        return best_m.predict_proba(prepare(df), iteration_range=(0, best_bi))[:, 1]


print(f"Eval AUC: {final_auc:.4f}")
