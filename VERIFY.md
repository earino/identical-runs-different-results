# Verify the paper

Every headline number in the paper comes from a file in this repository and a command you can run. Nothing below
needs a cloud account, an API key, or a GPU. Python 3.10+, with `matplotlib` and `scipy` for the figures.

Run everything from the repository root.

## The claim that matters most: runs of one pairing differ

> *"the six pairing averages over compliant runs span 0.0095 AUC, while the median pairing varies by 0.0107 across
> its own compliant runs"*

```bash
python analysis/figures.py data/study2/cells.csv out/study2
```

Prints the per-pairing table and writes the spread, histogram and compute figures. `data/study2/cells.csv` has one
row per run, 312 of them, with the holdout score, the flags and the budget accounting. The table is the paper's
Study 2 table, cell for cell: compliant runs, their mean, median, SD, 95% interval, 10th and 90th percentiles, best
run and all-run mean, and how many of the pairing's 52 runs broke a task rule. Its last line is the claim above:

```
compliant pairing means span 0.0095; median pairing SD 0.0107
```

The spread figure is the paper's first figure. It plots compliant runs only; the right-hand column gives each
pairing's noncompliant share, 0 to 12 percent. In every figure, colour is the agent and marker shape is the model;
the assignments are in `analysis/identity.py`.

## What several attempts buy, chosen on one set and scored on another

> *"Three attempts move the median artifact 0.0080 AUC above one attempt, and ten attempts 0.0135. Both gains hold
> out of sample: choosing the winner on the evaluation set and scoring it on the untouched holdout gives the same
> medians."*

```bash
python analysis/best_of_k.py data/study2/cells.csv data/study2/final_eval.csv
```

Reproduces the best-of-k table exactly (fixed seed) and prints it twice on the same resampled draws: once with each
winner chosen on holdout AUC, as the table reports, and once chosen on the evaluation-set score of the code each run
delivered and scored on the holdout. That score is in `data/study2/final_eval.csv`, extracted from each run's record
in `code/study2` by `python analysis/export_final_eval.py code/study2 data/study2/final_eval.csv`. It belongs to the
run's final code; in 18 of the 312 runs that was not its best experiment, which is why the best-experiment score in
`cells.csv` is not used here.

```
best-of-k over compliant runs, 6 pairings, 20,000 draws per k (seed 0)

  k  >=1 compliant   chosen on holdout: median  p5      p95   chosen on eval: median  p5      p95    optimism   same run
  1          96.6%                      0.7413  0.7220  0.7589                   0.7413  0.7220  0.7589   +0.00000   100%
  3         100.0%                      0.7493  0.7357  0.7641                   0.7493  0.7357  0.7641   +0.00003   97%
  5         100.0%                      0.7526  0.7411  0.7645                   0.7526  0.7409  0.7645   +0.00005   95%
 10         100.0%                      0.7548  0.7463  0.7663                   0.7548  0.7460  0.7663   +0.00005   94%

median kept: three attempts +0.0080 over one, ten attempts +0.0135
one attempt to ten: 5th percentile +0.0243, 95th percentile +0.0074
optimism from choosing on the holdout (mean kept, holdout-chosen minus eval-chosen): at most +0.00005 AUC
```

The second version answers the obvious objection to the first. Choosing the best of k runs on the same holdout that
scores it can flatter the winner, since the maximum of k noisy scores picks up favourable noise. Here the noise is a
score's standard error on one million rows, about 0.0005, twenty times smaller than the spread between runs, so the
measured optimism is at most 0.00005 AUC and the two versions agree on every median.

## What the larger model buys (Study 3)

> *"The larger model scored 0.0097 AUC above Flash ... 7.2 standard errors from zero ... 0.93 times the run-to-run SD
> of a single pairing. Run each model once, and the smaller one still comes out ahead 28 percent of the time: 15
> percent with pi, 35 with OpenCode."*

```bash
python analysis/compare_arms.py data/study3/cells.csv data/study2/cells.csv
```

This prints the per-agent gains with bootstrap intervals, the pooled main effect and, under it, the head-to-head line:

```
   one run of each model: the smaller model comes out ahead 28% of the time, averaged over agents (hermes 32%, opencode 35%, pi 15%)
```

That is the share of all compliant (Flash run, GLM-5.3 run) pairs in which the Flash run scores higher, per agent,
averaged over the three. The script then prints every pairwise interaction with a "clears zero / does not clear zero"
verdict, the detectability ladder, and the two confound checks (box effect F = 1.27, time drift +0.0000).

The script applies the two rule-breaking screens **to both arms** before comparing. That matters: an earlier interim
analysis of ours compared screened Flash runs against unscreened GLM-5.3 runs and overstated the gain by about a
thousandth. The docstring records it.

The figure "The larger model wins on average, but one run of each still favours the smaller model 28% of the time":

```bash
python analysis/figure_arms.py data/study3/cells.csv data/study2/cells.csv out/study3
```

## Which agent looks best depends on the model (Studies 2 and 3)

> *"Across models, moving from GLM-5.3 Flash to DeepSeek 4.1 Flash raised pi by 0.0095, Hermes by 0.0058 and
> OpenCode by 0.0003. pi's gain exceeds OpenCode's by 0.0092, 3.1 standard errors of that difference."*
>
> *"pi leads OpenCode on each stronger model, by 0.0053 on DeepSeek 4.1 Flash and 0.0058 on GLM-5.3, at 2.4 and
> 2.6 standard errors."*

```bash
python analysis/figure_interaction.py data/study2/cells.csv data/study3/cells.csv out/interaction
```

Writes the two agent-by-model figures and prints, for each study, every number the paper quotes from them:

```
Study 2: GLM-5.3 Flash to DeepSeek 4.1 Flash, compliant runs
  pi        0.7361 -> 0.7456   change +0.0095  (SE 0.0023, 4.2 SE)
  Hermes    0.7383 -> 0.7441   change +0.0058  (SE 0.0023, 2.6 SE)
  OpenCode  0.7400 -> 0.7403   change +0.0003  (SE 0.0019, 0.2 SE)
  pi's change minus OpenCode's: +0.0092  (3.1 SE)
  on GLM-5.3 Flash      agents spread 0.0039; pi minus OpenCode -0.0039 (-2.0 SE)
  on DeepSeek 4.1 Flash agents spread 0.0053; pi minus OpenCode +0.0053 (+2.4 SE)
  every mean inside every pairing's 10th-90th percentile band: yes
```

and the same block for Study 3, whose "pi's change minus OpenCode's" line (+0.0097, 3.3 SE) is the interaction
`compare_arms.py` reports by bootstrap. The standard errors here use the normal approximation, the square root of the
summed squared standard errors; it reproduces the paper's 3.3 exactly.

Read the two studies' agreement with the caution the paper gives it. Both comparisons start from the same GLM-5.3
Flash runs, on which pi trails OpenCode by 0.0039, so that gap counts toward pi's larger gain in both. The lines
that do not share it are the two "pi minus OpenCode" lines on the stronger models.

## The integrity claims — check our work, do not trust it

> *"11 of 312 runs broke the task rules ... five computed features from the batch they were scoring"*

This is the claim you should be most suspicious of, because we are the ones who decided what counts as cheating.
So the delivered code is in this repository and you can run the screen yourself:

```bash
python analysis/audit_exported_code.py
```

It re-runs the AST taint analysis over the **last `train.py` of every one of the 468 runs** and prints the offending
source line for each hit. Expect 7 raw hits in study 2 and 5 in study 3. Two of study 2's seven are false positives
we excluded by hand after reading them; they are printed anyway so you can judge that call. Our per-run record is
`data/study2/frame_stats.csv`, and the separate eval-leak audit is `data/study2/eval_leak_audit.txt`.

To read what an agent actually wrote, including its own notes:

```bash
ls  code/study3/var7/airline/pi/glm-5.3/seed35/          # eval.json, experiments.tsv, cpu_ledger.tsv, FINAL.md
cat code/study3/var7/airline/pi/glm-5.3/seed35/FINAL.md  # the agent's own summary of what it tried
ls  code/study3/var7/airline/pi/glm-5.3/seed35/code/     # train.py at every commit, oldest first
```

`seed35` is the run flagged for computing statistics from the frame it was scoring, and it posted the highest score
in its box. That pattern — the most impressive artifact being the least compliant one — is the paper's point.

## The strongest check: re-score a delivered program yourself

The holdout ships with this repository, so you do not have to take any score on trust. Take the code an agent wrote,
run it on the 1,000,000 rows it never saw, and compare:

```bash
python analysis/rescore.py code/study3/var7/airline/pi/glm-5.3/seed35
```

It rebuilds the working directory the agent had, imports the delivered `train.py`, calls its `predict_proba` on the
holdout with the label removed, and prints the AUC. Compare that with the `holdout_auc` for the same run in
`data/study3/cells.csv`. Add `--version 03` to score an earlier commit instead of the final one.

**Expect agreement to about 0.0002, not to the last digit.** These programs retrain from scratch and tree training
is not bit-deterministic, so a re-score lands near the recorded number rather than on it. Measured on three runs:

| Run | Recorded | Re-scored | Difference |
|:--|--:|--:|--:|
| opencode seed33 | 0.747537 | 0.747368 | 0.00017 |
| opencode seed30 | 0.733199 | 0.732970 | 0.00023 |
| pi seed35 (flagged) | 0.803492 | 0.804381 | 0.00089 |

The flagged run drifts four times further than the clean ones, which is what you would expect from a program that
builds features out of the batch it is scoring: its predictions depend on the composition of that batch, so it is
less stable under retraining. The screen and the arithmetic point the same way.

Budget a few minutes per run. Some of these programs fit 16-member ensembles.

## Rebuilding the table of per-run results from scratch

`cells.csv` is derived, not hand-made. It is produced by `analysis/analyze.py` from the raw run trees. Those trees
are 43 GB and are not in this repository, so this command is documented rather than runnable here:

```bash
python analysis/analyze.py <pulls-dir> <out-dir>     # needs the raw run trees
```

What *is* runnable is everything downstream of `cells.csv`, which is every number in the paper.

## Token and cost claims

> *"1.86 billion input tokens and wrote 25.1 million, a ratio of about 74 to 1"*

```bash
python -c "
import csv
reas={(r['harness'],r['model'],int(r['seed'])):int(r['reasoning_tokens'])
      for r in csv.DictReader(open('data/study2/reasoning_tokens.csv'))}
tin=tout=tre=0
for r in csv.DictReader(open('data/study2/cells.csv')):
    if r['counted']!='True' or not r.get('tokens_in'): continue
    tin+=int(float(r['tokens_in'])); tout+=int(float(r['tokens_out'] or 0))
    tre+=reas.get((r['harness'],r['model'],int(r['seed'])),0)
print(f'input {tin/1e9:.2f}B  output {tout/1e6:.1f}M  reasoning {tre/1e6:.1f}M  generated {(tout+tre)/1e6:.1f}M  ratio {tin/(tout+tre):.0f} to 1')"
```

Prints `input 1.86B  output 16.4M  reasoning 8.7M  generated 25.1M  ratio 74 to 1`. The join matters: `tokens_out`
comes from all 312 runs, while reasoning was recovered for the 306 whose harness reported it, so the reasoning
figure is a slight undercount.

Reasoning tokens are counted separately by every harness and were missing from our own ledger until we went back for
them; `analysis/reasoning_tokens.py` is the recovery script. Neither arm set a reasoning level, and for these models
omitting it selects the maximum effort setting.

> *"The failure did repeat on one more model, DeepSeek V4 Pro ... Claude Code cached 4 and 11 percent while the other
> five agents cached 83 to 99 percent. The quota cut off both Claude Code runs, but it also cut off both of pi's,
> which still cached 91 and 98 percent."*

```bash
python -c "
import csv
for r in csv.DictReader(open('data/study1/cells.csv')):
    if r['model'].startswith('deepseek-v4-pro'):
        print(f\"{r['harness']:9s} seed {r['seed']}  cached {float(r['cached_share']):.0%}  {'cut off by the quota' if r['provider_error'] else ''}\")"
```

`data/study1/cells.csv` is Study 1: 116 runs, one row each. `data/study1/claude_per_request.csv` is Claude Code's
request log, recovered for the runs that stopped before reporting their usage; the paper's Claude Code costs use it.

## What is not here, and why

**The raw run trees**, 43 GB of harness logs and container state. The parts a reader needs — delivered code at every
commit, per-experiment records, CPU ledgers, the agents' own notes — are extracted into `code/`.

The holdout **is** included, at `task/airline/holdout.csv`: 1,000,000 rows,
sha256 `f187e71afd2eadd5121a47577d4ea95d78299009797f4943d20d42a711ab9abc`. We chose full reproducibility over
keeping the task reusable, and that choice has a cost worth stating: once this repository is public, the airline
task is contaminated for future agent evaluation, because a model trained afterwards may have read the answers.
Anyone extending this work needs new tasks, not this one.

**Credentials and machine images.** Nothing here contains a key.
