# Identical Runs, Different Results — data, code and delivered artifacts

Everything needed to check the paper *Identical Runs, Different Results: Benchmarking AI Coding Agents on
Open-Weight Models* (Eduardo Ariño de la Rubia and Szilard Pafka, September 2026), including the code the agents
themselves wrote.

**Start with [VERIFY.md](VERIFY.md).** It maps each headline claim to the one command that reproduces it.

## Where each number in the paper comes from

Every analysis the paper quotes has its printed output committed under
[`analysis/results/`](analysis/results/), one file per table or claim, so a number can be found without running
anything. `bash analysis/make_results.sh` regenerates them all from `data/` (about ten minutes on a laptop; each file
starts with the command that wrote it).

| Paper | Script | Committed output |
|:--|:--|:--|
| Table 3, Figure 1, Figure 5 | `analysis/figures.py` | `table3_figure1_study2_pairings.txt` |
| Table 4, Figure 6 (Study 1) | `analysis/study1_agents_vs_noise.py` | `table4_figure6_study1.txt` |
| Table 5 | `analysis/sensitivity_violations.py` | `table5_exclusion_rules.txt` |
| Table 6 | `analysis/best_of_k.py` | `table6_best_of_k.txt` |
| Table 7 | `analysis/compare_arms.py` | `table7_study3_vs_study2.txt` |
| Tables 8, 13 | `analysis/policy_tables.py` | `tables8_13_yield_and_cost.txt` |
| Table 9, Table 16, run counts, three-run draws, Hanley-McNeil SE | `analysis/paper_numbers.py` | `table9_and_text_numbers.txt` |
| Table 10 | `analysis/interaction_table.py` | `table10_interactions.txt` |
| Figures 3, 4 | `analysis/figure_interaction.py` | `figures3_4_interaction.txt` |
| Figure 2 | `analysis/figure_arms.py` | `figure2_model_upgrade.txt` |
| Table 11 (2007 flights) | `analysis/analyze_2007.py` on `data/2007/scores.csv` | `table11_later_year.txt` |
| Table 14 | `analysis/table14_decisions.py` (retrains two delivered models) | `table14_decisions.txt` |
| Table 17 | `analysis/best_of_k_by_pairing.py` | `table17_best_of_k_by_pairing.txt` |
| Table 18 | `analysis/table18_repricing.py` | `table18_repricing.txt` |
| Tables 1, 15, all run counts | `analysis/ledger.py` | `counts_ledger.txt` |
| Compliance verdicts (Section 3.5) | `analysis/audit_exported_code.py` | `compliance_audit.txt` |
| Section 4.9, cache per request | `analysis/cache_requests.py` on `data/study1/claude_requests.csv` | `section4.9_cache_requests.txt` |
| Table 12, the 23-fold cost gap | `analysis/table12_study1_cache_cost.py` | `table12_study1_cache_cost.txt` |
| Appendix D token totals | `analysis/token_totals.py` | `appendixD_study2_tokens.txt` |
| Re-scoring one program end to end | `analysis/rescore.py` | `rescore_study3_pi_seed35.txt` |

## What the paper measures

Six coding agents were paired with open-weight models and given the same machine-learning task: improve a model on
a tabular dataset, inside a fixed compute budget. A holdout set the agent never sees decides the score. The same
pairing was then run many times to see how much the result varies when nothing changes.

| | Design | Runs |
|:--|:--|--:|
| **Study 1** | six agents x six model endpoints, three runs each | 116 |
| **Study 2** | three agents x two models, 52 runs each | 312 |
| **Study 3** | the same three agents on a larger model from the same family, 52 runs each | 156 |

Four results, in the order they matter:

1. **Run-to-run variation exceeded the differences between agents.** The six pairing averages span 0.0095 AUC; the
   median pairing varies by 0.0107 across its own runs. Three-run comparisons ranked pairings unreliably.
2. **The rule-breaking runs are at the top of the table.** 10 of 312 runs trained on the labelled evaluation file
   or built features from the batch they were scoring, and they include the seven highest scores. Removing them takes
   the best score from 0.8293 to 0.7695.
3. **The larger model's gain is about the size of run-to-run noise.** Moving to the larger model gained 0.0091 AUC,
   7.1 standard errors from zero, yet run each model once and the smaller one still comes out ahead 28 percent of the
   time. Study 3 is also the positive control: the same design that could not rank three agents detects a planned
   change, so the agents are close together and the measurement is not blunt.
4. **Much of the gain belongs to the year the agents tuned on.** Scored on one million flights from 2007, the same
   programs kept a third of their gain over the starting code; every finding kept its direction but shrank.

## Layout

```
paper/        the paper, as Markdown and PDF, with its figures
data/         one row per run: score, compliance, budget, tokens  <- every number in the paper comes from here
  study1/     the broad grid
  study2/     52 runs x six pairings, plus the leak audit and the frame-statistics record
  study3/     52 runs x three pairings on the larger model, plus its leak audit and final evaluation scores
  2007/       every scored program of Studies 2 and 3, rescored on 2006 and scored on 2007 flights
code/         what the agents actually delivered, per run: train.py at every commit, their own
              notes, the per-experiment record and the CPU ledger      <- 468 runs, 2,489 program versions
analysis/     the scripts that turn data/ into the paper's tables and figures
harness/      how the runs were produced and scored: the runner, the agent wrappers, the task
              rules given to the agent, the frozen configs, and METHODS.md
task/         the task definition, the train/eval split, the 1M-row holdout, and 1M flights from 2007
```

## The part worth reading even if you check nothing

`code/` holds every version of every program the agents wrote, plus the notes they kept for themselves. It is the
record of what the agents actually tried, and it is why the integrity claims in the paper can be checked rather than
believed. For example:

```bash
cat code/study3/var7/airline/pi/glm-5.3/seed35/FINAL.md
```

is one agent's own account of its work. That run posted the highest score in its box and is one of the runs that
computed features from the batch it was scoring.

## Reproducing

Python 3.13.1 with the pinned versions in `requirements.txt` (`pip install -r requirements.txt`); re-training a
delivered program reproduces its recorded score only under those versions. No API key, no cloud account, no GPU. Everything
downstream of `data/*/cells.csv` runs on a laptop in seconds — see [VERIFY.md](VERIFY.md).

Re-running the benchmark itself is a different matter: it needs containers, a model endpoint and roughly a day per
arm on four machines. `harness/` and `harness/METHODS.md` document how, and the frozen configs record exactly what
was run.

## Deliberately not included

- **The raw run trees**, 43 GB of harness logs and container state. Everything a reader needs is extracted into
  `code/`.
- **Credentials.** Nothing here contains a key.

The holdout **is** included, at `task/airline/holdout.csv`. That makes every score checkable end to end — see the
re-scoring section of [VERIFY.md](VERIFY.md) — and it means the airline task is spent for future agent evaluation
once this repository is public, since a model trained afterwards may have read it. Extending this work needs new
tasks.

## License

Different parts of this repository are under different terms:

| Part | License |
|:--|:--|
| Code: `analysis/`, `harness/` | [MIT](LICENSE) |
| The paper, its figures (`paper/`) and the result tables (`data/`) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| The agents' delivered programs and records (`code/`) | written by the AI agents in our runs; released under [MIT](LICENSE) to the extent we hold any rights in them |
| The flight data (`task/airline/`) | derived from [Data Expo 2009: Airline on time data](https://doi.org/10.7910/DVN/HG7NV7), which is [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/); our slices are released under CC0 as well |

No agent software or model weights are redistributed here.

## Citing

```bibtex
@misc{arino2026identicalruns,
  title  = {Identical Runs, Different Results: Benchmarking AI Coding Agents on Open-Weight Models},
  author = {Ari{\~n}o de la Rubia, Eduardo and Pafka, Szilard},
  year   = {2026},
  month  = {September}
}
```
