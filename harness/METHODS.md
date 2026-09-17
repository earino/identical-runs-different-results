# Methods and statistics

Supporting document for the business briefs. It holds the detail that does not fit on two pages: the design, what
the compute meter counts, the audits applied to every run, and the statistics behind the claims.

## Design

Six coding agents took part: Claude Code, Codex, pi, OpenCode, Hermes and OpenClaw. Each was given a model, a
dataset, a written task and a compute budget, and asked to improve an XGBoost classifier by editing one file.

| | |
|:--|:--|
| Task | airline departure delay, binary classification, AUC |
| Data | train 100,000 rows from 2005; evaluation 100,000 rows from 2006; holdout 1,000,000 rows from 2006 |
| Split | fixed, identical for every run, so agents are compared on the same data |
| Model rows | GLM-5.3, GLM-5.3 Flash and DeepSeek 4.1 Flash through LunaRoute; DeepSeek V4.1 Flash, DeepSeek V4 Flash and Nemotron Super through Ollama Cloud |
| Runs | 3 per agent and model pair, 108 in total, 103 scored |
| Container | 4 cores, 8 GB, one per run, no network except the model endpoint |
| Counted experiments | 40 per run, each a call to the provided run script |
| Per-experiment limit | 120 seconds of wall clock |

The holdout is one million rows. The standard error of an AUC on a sample that size is about 0.0005. Differences
of 0.01 to 0.03 between runs are therefore not holdout sampling noise. They come from the agent's own stochastic
search. This is why repeated runs are needed, and why a larger holdout would not remove the variation.

## The compute budget

Each run had 18,000 CPU seconds of Python compute, with a container-level stop at 46,000. The budget was
calibrated by re-running the delivered code of an earlier pilot on the same hardware.

**What the meter counts.** Every Python process started inside the container, summed over user and system CPU
time, including child processes. That covers experiments run through the provided script, the agent's own scripts,
one-line commands, data loading and feature engineering. A start-up hook in the interpreter records each process
on exit and writes one line per process to a ledger in the run directory. Every model fit is separately recorded
with its row count and duration.

**What the meter does not count.** Time the model spends generating tokens, and any reasoning the agent does in
context rather than in code. This is a real confound. An agent that plans carefully and fits few models meters as
a low user of the budget, even if it searched hard. An agent with redundant fits or retry loops meters as a high
user. The brief's finding should be read as a statement about compute invested in fitting, not about search
quality in the abstract.

**Enforcement.** The hook refuses to start a new Python process once the ledger passes the budget, and a watchdog
thread stops a running process that crosses it. Because the watchdog checks every ten seconds, a run can overshoot
slightly. Two of 103 runs did, by at most 17 seconds. No run was stopped by the container-level cap. Two runs hit
the refusal and finished normally afterwards. Overruns were scored like any other run and were not penalised.

## Scoring and audits

Every run is scored by re-running its delivered code on the holdout, which the agent never sees. Two audits run
over every run, not only over suspicious ones.

**The automatic screen** flags a run whose best reported evaluation score exceeds its holdout score by more than
0.03. It caught one run. That run had concatenated the evaluation rows into its training data, reported an
evaluation AUC of 1.0, and scored 0.839 on the holdout against a baseline of 0.715.

**The code audit** reads the delivered file of every run and traces whether the evaluation frame reaches a fit
call, directly or through a concatenation. It was written to test the screen's false-negative rate, since a run
that raises both scores together would pass the screen. Results over 108 delivered files:

| Finding | Runs |
|:--|--:|
| Evaluation rows concatenated into training data | 1, the same run the screen caught |
| Evaluation set used to stop training early | 2 |
| Training on evaluation rows that the screen missed | 0 |

The two early-stopping cases are permitted by the task rules, which name early stopping as an allowed technique.
Neither inflated its score: both finished within 0.001 AUC of their holdout result. They are disclosed because the
practice makes a reported evaluation score optimistic even when the holdout score is clean.

**Exclusions.** Five runs are excluded from the tables: three Codex runs that never started because the gateway
rejected the request format, the one evaluation-row leak, and one run whose delivered file was not its own best
experiment.

## Statistics

**Resolution.** With three runs per pair, the smallest difference in agent means this design can separate from
run-to-run variation is about 0.02 AUC. Four of six model rows have agent-mean spreads below that. The design
therefore detects agents that leave the budget unused, and cannot rank the agents that use it.

**Compute against score.** Budget share used is compared with the holdout score expressed as a within-row
standard score, so the six rows share one axis. Confidence intervals come from a bootstrap over runs with 5,000
resamples. The p-values come from permutation, shuffling scores within a row 10,000 times.

| Subset | Runs | Rank correlation | 95% interval | p |
|:--|--:|--:|:--|--:|
| All runs | 103 | +0.59 | +0.45 to +0.71 | below 0.001 |
| Above a tenth of budget | 71 | +0.51 | +0.30 to +0.68 | below 0.001 |
| Above a quarter | 59 | +0.53 | +0.36 to +0.66 | below 0.001 |
| Above half | 38 | +0.47 | +0.15 to +0.68 | 0.003 |
| Above three quarters | 23 | +0.01 | −0.46 to +0.49 | 0.977 |

The relationship is not an artefact of the gap between a low-use cluster and a high-use band. It survives when
the low-use runs are removed, down to runs using more than half the budget. It disappears above three quarters.
With 23 runs there, the interval is wide, so this is a failure to detect a relationship, not evidence that none
exists. The practical reading is that the relationship saturates.

Per row, all six point the same way; four are individually significant.

| Row | Runs | Rank correlation | 95% interval | p |
|:--|--:|--:|:--|--:|
| DeepSeek V4 Flash, Ollama | 17 | +0.85 | +0.61 to +0.94 | below 0.001 |
| GLM-5.3, LunaRoute | 17 | +0.81 | +0.51 to +0.93 | below 0.001 |
| DeepSeek 4.1 Flash, LunaRoute | 15 | +0.80 | +0.49 to +0.92 | below 0.001 |
| DeepSeek V4.1 Flash, Ollama | 18 | +0.79 | +0.48 to +0.93 | below 0.001 |
| Nemotron Super, Ollama | 18 | +0.54 | +0.08 to +0.81 | 0.024 |
| GLM-5.3 Flash, LunaRoute | 18 | +0.42 | −0.22 to +0.82 | 0.086 |

**An alternative measure of effort.** The ledger also counts model fits. Fits predict the score much more weakly
than CPU seconds do: +0.25 over all runs, with an interval of +0.06 to +0.42, and +0.20 with an interval crossing
zero among runs with more than 50 fits. The number of experiments an agent runs therefore says little. The compute
those experiments consume says more. One agent ran all 40 counted experiments on one row and fit 696 models, while
using 5% of its budget, because every model was tiny.

**Causal reading.** The relationship is correlational and the direction is not established. An agent whose search
is going well may continue, which would produce the same pattern. The claim in the brief is limited to what a
buyer can check: a run that leaves most of its budget unused scored below its row's average, and the ledger makes
that visible before anyone reads the code.

## Reproducing

```sh
.venv/bin/python scripts/phase2/stage1_analysis.py results/phase2/merged/cells.csv
.venv/bin/python scripts/phase2/compute_score_sensitivity.py
EXPORT_ROOT=results/phase2/evals .venv/bin/python scripts/phase2/audit_eval_leak.py
```

Per-run records, including the CPU ledger, the fit log and the code at every commit, are under
`results/phase2/evals/`. The merged table is `results/phase2/merged/cells.csv`.

## Token accounting, and how it was validated

Each harness reports usage differently, and `bench/usage.py` reads the authoritative record for each: a final
cumulative object for Claude Code, OpenClaw and Hermes, and summed per-event records for pi, OpenCode and Codex.

Phase 2 keeps Claude Code's session transcript, which allows an independent check of the one agent whose charges
drive brief 2. `scripts/phase2/claude_request_log.py` rebuilds each run's usage request by request from that
transcript. Two cautions apply. Each assistant message is repeated in the transcript whenever the conversation is
re-serialised, so entries must be deduplicated by message id; without that the input totals inflate several-fold.
And the transcript's `cache_creation_input_tokens` field reads zero on every request, including on runs whose
cache clearly works, so this endpoint does not populate it and a zero there means nothing.

Deduplicated, the reconstruction reproduces the charge of all 14 Claude Code runs that reported one, to within ten
cents. That agreement is what licenses using it for the two runs that were stopped before reporting. Results are
in `results/phase2/claude_per_request.csv`.

**No invoice reconciliation is possible on this plan.** The runs were paid by a flat-rate subscription that
reports a weekly quota percentage rather than a currency amount, so there is no bill to compare against. The one
external check available is that modelled spend tracked the provider's own quota meter at two points in the same
week, within about two percent both times. Every charge in the briefs is a rate-card projection.

**Cache against volume.** On the DeepSeek V4 Flash row, Claude Code averaged 7.9 million fresh and 0.6 million
cached input tokens per run, against pi's 3.9 million total. Repricing Claude Code's own volume at pi's hit rate
gives $0.20 a run against its actual $1.86 and pi's $0.08. The cache therefore accounts for about a factor of
nine and the volume for about a factor of two and a half, and the two multiply to the observed 23.

**Per-request behaviour.** On the failing endpoint the cache returns nothing from the first request of the run
onward, at 14,000 tokens, and consecutive requests sharing nearly all their text still read nothing back. On
DeepSeek V4.1 Flash the same agent reads its prefix back on the following request at sizes up to 122,000 tokens.
Context growth is therefore not the explanation. A controlled test with byte-identical prefixes sent through both
endpoints from an instrumented client would separate a client-side request-packaging cause from a server-side one.
