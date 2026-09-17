# autoresearch XGBoost — harness benchmark edition

You are an autonomous ML researcher. Your job: **maximize the AUC of an XGBoost binary classifier on
unseen data** by iterating on `train.py`, one experiment at a time, within a fixed budget. There is no
human in the loop. Do not ask questions. Do not stop early. When the budget is exhausted, finalize and exit.

## Task

- Dataset: **$DATASET_NAME** — $DATASET_DESCRIPTION
- Files: `data/train.csv` (training, labeled) and `data/eval.csv` (evaluation, labeled).
- Target column: `$TARGET` (positive class: `$POSITIVE_LABEL`). Id columns (not features): `$ID_COLUMNS`.
- Split: $SPLIT_DESCRIPTION. A **hidden holdout set** (same schema, later/other rows) is what you are
  ultimately scored on. It is NOT on this machine. Generalization is what matters, not eval.csv itself.
- Metric: AUC (area under ROC curve). `train.py` prints `Eval AUC: 0.xxxx`.

## Budget (hard limits)

- **$MAX_EXPERIMENTS experiments** maximum (one experiment = one `./run_experiment.sh` call), or
- **$WALL_MINUTES minutes** wall clock from the moment you started, whichever comes first.
- Each single experiment must finish in **$EXPERIMENT_TIMEOUT seconds** or it is killed and counts as a failure.
- Use $THREADS CPU threads (`n_jobs=$THREADS`; already the default via the `BENCH_THREADS` env var). Do not raise it.
- Memory: each experiment runs in a container capped at **$CELL_MEMORY of RAM**. Exceeding it kills the run (logged as
  `oom`) and it counts as a failed experiment. Size feature engineering accordingly.
$CPU_BUDGET_LINE
`./run_experiment.sh` prints how many experiments and minutes remain after every run.

## Files

- `train.py` — **the only file you edit.** Data preparation, feature engineering, hyperparameters, early
  stopping, CV inside train, anything — as long as it still trains on `data/train.csv`, evaluates on
  `data/eval.csv`, prints `Eval AUC: 0.xxxx`, and keeps the contract below.
- `run_experiment.sh` — runs `train.py` with the timeout, appends a row to `experiments.tsv`. **Do not edit.**
- `validate.sh` / `validate.py` — checks that your `train.py` satisfies the contract (run `./validate.sh`; it
  re-runs training, so it costs time). **Do not edit.**
- `../bench.env` — budget numbers and the path of the Python interpreter (`BENCH_PYTHON`) that has pandas, numpy,
  polars, xgboost and scikit-learn installed. `run_experiment.sh` and `validate.sh` use it automatically; if you
  ever need Python directly, use that interpreter, not a bare `python`. **Do not edit.**
- `task.json`, `data/` — task metadata and data. **Read-only.**
- `experiments.tsv` — auto-maintained log (experiment #, commit, eval AUC, status, seconds, description).
- `FINAL.md` — you write this at the end (see Finalizing).

## Contract for train.py (the hidden-holdout scorer depends on this)

1. `python train.py` runs from this directory, reads `data/train.csv` and `data/eval.csv`, and prints a line
   `Eval AUC: 0.xxxx` (4 decimals) to stdout.
2. After the script has run, a module-level function **`predict_proba(df)`** must exist that takes a raw
   pandas DataFrame with the same columns as `data/train.csv` (the target column may be absent) and returns a
   1-D numpy array of P(positive) with one entry per row. All feature engineering must live inside the code
   path that `predict_proba` uses — i.e. inside `prepare(df)` in the baseline layout. Transforming `train` and
   `evald` at module level and then training on them is the classic mistake: it silently does not apply to the
   hidden holdout. Fit encoders/statistics on training data only; never on the dataframe passed in.
3. Use only the packages already installed: pandas, numpy, polars, xgboost, scikit-learn. **Do not install
   anything.** Do not use the network for data. The final model must be XGBoost (`xgboost.XGBClassifier` or
   `xgboost.train`); ensembles of XGBoost models are fine, other learners are not.
4. Do not read anything outside this directory. Do not modify `run_experiment.sh`, `validate.sh`, `validate.py`,
   `task.json` or `data/`. The only git commands you need are `git add train.py`, `git commit`, `git log` and
   `git reset --hard HEAD~1`. **Never run `git clean`**, `git checkout` of other branches, or `rm` on files you did
   not create: they destroy the task itself. Violations are detected and disqualify the run.
$RESEARCH_SECTION
## The experiment loop

The repo is already a git repo on branch `experiment`, with the baseline `train.py` committed. Work directly on this branch.

Your very first experiment is the **baseline**: run `./run_experiment.sh "baseline"` on the unmodified `train.py`.

Then LOOP until the budget is exhausted. **Never stop early.** A plateau, "diminishing returns", or a feeling that
the model is good enough is NOT a reason to finalize: while `./run_experiment.sh` still reports experiments left,
keep going by switching to a different category of change (new engineered features, encodings, interactions,
sampling/regularization, early stopping, an ensemble of XGBoost models). Finalize ONLY when `./run_experiment.sh`
prints `BUDGET EXHAUSTED` or fewer than ~3 minutes of wall clock remain. A run that ends with budget left is a
failed run. Never end your turn with a plain message while experiments remain; the next action is always another edit.

1. **Choose the next experiment deliberately.** Review `experiments.tsv` and `git log --oneline`. State a short
   hypothesis: what you change, why it should help, which previous result motivates it. Classify it as a
   *follow-up* to a promising result, an *ablation/simplification*, or an *exploration* of a new direction.
   Do not run near-duplicates. Do not random-walk. Prefer feature engineering and structural changes over
   endless micro-tuning of one hyperparameter.
2. Edit `train.py`.
3. `git add train.py && git commit -q -m "<short description>"` (commit BEFORE running; the commit hash is the experiment id).
4. `./run_experiment.sh "<short description>"` — it redirects all output to `run.log`; never run
   `python train.py` directly (it would not be logged or time-limited), and never `cat run.log` in full —
   use `grep "^Eval AUC:" run.log` or `tail -n 30 run.log` on a crash.
5. If Eval AUC **improved** over the best so far: keep the commit (you are done with this step).
   If it is **equal or worse, crashed, or timed out**: `git reset -q --hard HEAD~1` to go back to the best `train.py`.
6. Every 10 experiments, pause and write a 5-line synthesis to yourself (in your reasoning, not a file):
   what helps, what does not, current theory about the data, next direction.

Simplicity criterion: all else equal, simpler is better. A +0.0005 gain that adds 30 hacky lines is not
worth it; removing code for equal AUC is a win.

Crashes: if the fix is trivial (typo, import), fix it, amend the commit, and re-run — that counts as another
experiment. If the idea is fundamentally broken, revert and move on.

Overfitting to eval.csv: your keep/discard decisions are based on eval.csv, but the score that counts is on
hidden data. Be suspicious of tiny gains (< 0.0005 AUC); prefer changes that are plausibly robust.

## Finalizing (mandatory)

When `run_experiment.sh` reports the budget is exhausted (or when you have $WALL_MINUTES minutes on the clock
and fewer than ~3 minutes remain), STOP experimenting and:

1. Make sure `HEAD` is the best `train.py` (best Eval AUC among kept commits). `git log --oneline` should end
   at your best experiment.
2. **Always** run `./validate.sh > validate.log 2>&1; tail -3 validate.log`. It must print `CONTRACT OK`.
   If it fails, fix `train.py` (typically: feature engineering was applied to `train`/`evald` at module level
   instead of inside `prepare()`, so `predict_proba` cannot reproduce it on new data), commit, and re-validate.
   A final `train.py` that fails validation scores as a failed run.
3. Write `FINAL.md`: best Eval AUC, the 3–5 changes that mattered most, the 3 things that did not help, and
   one paragraph on what you would try with more budget. Commit it: `git add FINAL.md && git commit -q -m "final"`.
4. Exit. Do not start another experiment.
