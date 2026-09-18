# Identical Runs, Different Results — data, code and delivered artifacts

Everything needed to check the paper *Identical Runs, Different Results: Benchmarking AI Coding Agents on
Open-Weight Models* (Eduardo Ariño de la Rubia and Szilard Pafka, September 2026), including the code the agents
themselves wrote.

**Start with [VERIFY.md](VERIFY.md).** It maps each headline claim to the one command that reproduces it.

## What the paper measures

Six coding agents were paired with open-weight models and given the same machine-learning task: improve a model on
a tabular dataset, inside a fixed compute budget. A holdout set the agent never sees decides the score. The same
pairing was then run many times to see how much the result varies when nothing changes.

| | Design | Runs |
|:--|:--|--:|
| **Study 1** | six agents x six model endpoints, three runs each | 116 |
| **Study 2** | three agents x two models, 52 runs each | 312 |
| **Study 3** | the same three agents on a larger model from the same family, 52 runs each | 156 |

Three results, in the order they matter:

1. **Run-to-run variation exceeded the differences between agents.** The six pairing averages span 0.0095 AUC; the
   median pairing varies by 0.0107 across its own runs. Three-run comparisons ranked pairings unreliably.
2. **The rule-breaking runs are at the top of the table.** 11 of 312 runs trained on data the rules put off limits
   or built features from the batch they were scoring. Removing them takes the best score from 0.8293 to 0.7695.
3. **The larger model's gain is about the size of run-to-run noise.** Moving to the larger model gained 0.0097 AUC,
   7.2 standard errors from zero, yet run each model once and the smaller one still comes out ahead 28 percent of the
   time. Study 3 is also the positive control: the same design that could not rank three agents detects a real change
   cleanly, so the agents are close together and the measurement is not blunt.

## Layout

```
paper/        the paper, as Markdown and PDF, with its figures
data/         one row per run: score, flags, budget, tokens  <- every number in the paper comes from here
  study1/     the broad grid
  study2/     52 runs x six pairings, plus the leak audit and the frame-statistics record
  study3/     52 runs x three pairings on the larger model
code/         what the agents actually delivered, per run: train.py at every commit, their own
              notes, the per-experiment record and the CPU ledger      <- 468 runs, 2,489 program versions
analysis/     the scripts that turn data/ into the paper's tables and figures
harness/      how the runs were produced and scored: the runner, the agent wrappers, the task
              rules given to the agent, the frozen configs, and METHODS.md
task/         the task definition, the train/eval split, and the 1M-row holdout
```

## The part worth reading even if you check nothing

`code/` holds every version of every program the agents wrote, plus the notes they kept for themselves. It is the
record of what the agents actually tried, and it is why the integrity claims in the paper can be checked rather than
believed. For example:

```bash
cat code/study3/var7/airline/pi/glm-5.3/seed35/FINAL.md
```

is one agent's own account of its work. That run posted the highest score in its box and is one of the runs the
rule screen flags.

## Reproducing

Python 3.10+; `matplotlib` and `scipy` only for the figures. No API key, no cloud account, no GPU. Everything
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

## Citing

```bibtex
@misc{arino2026identicalruns,
  title  = {Identical Runs, Different Results: Benchmarking AI Coding Agents on Open-Weight Models},
  author = {Ari{\~n}o de la Rubia, Eduardo and Pafka, Szilard},
  year   = {2026},
  month  = {September}
}
```
