# Verify the paper

Every headline number in the paper comes from a file in this repository and a command you can run. Nothing below
needs a cloud account, an API key, or a GPU. Python 3.10+, with `matplotlib`, `numpy` and `scipy`.

Run everything from the repository root.

## The claim that matters most: runs of one pairing differ

> *"the six pairing averages over compliant runs span 0.0095 AUC, while the median pairing varies by 0.0107 across
> its own compliant runs"*

```bash
python analysis/figures.py data/study2/cells.csv out/study2
```

Prints the per-pairing table and writes the spread, histogram and compute figures. `data/study2/cells.csv` has one
row per run, 312 of them, with the holdout score, the compliance flags and the budget accounting. The table is the
paper's Study 2 table, cell for cell: compliant runs, their mean, median, SD, 95% interval, 10th and 90th
percentiles, best run and all-run mean, and how many of the pairing's 52 runs broke a task rule. Its last line is the
claim above:

```
compliant pairing means span 0.0095; median pairing SD 0.0107
```

The spread figure is the paper's first figure. It plots compliant runs only; the right-hand column gives each
pairing's noncompliant share, 0 to 10 percent. In every figure, colour is the agent and marker shape is the model;
the assignments are in `analysis/identity.py`.

## One ledger for every count

> *"116 runs in all: 113 delivered an artifact ... 103 were scored"*, *"10 of 312 runs broke the task rules"*,
> *"Of the 156 runs, four delivered code that failed when scored on the holdout and eight broke a task rule"*

```bash
python analysis/ledger.py data/study1/cells.csv data/study2/cells.csv data/study3/cells.csv
```

Prints every flow count the paper quotes, from the per-run tables, and asserts the identities the paper relies on:
compliant runs equal scored runs minus rule-breakers, no unscored run counts as compliant, and the study totals. If a
table changed and a count stopped reconciling, it would fail. Its Study 2 and Study 3 blocks:

```
STUDY 2
  runs counted 312; status {'scored': 312}
  trained on evaluation labels 5; statistics from the scored frame 5; both 0; broke a rule 10
  compliant 302 of 312 (96.8%); excluded 10 = 0 not scored + 10 broke a rule
...
STUDY 3
  runs counted 156; status {'scored': 152, 'holdout_error': 4}
  trained on evaluation labels 3; statistics from the scored frame 5; both 0; broke a rule 8
  compliant 144 of 156 (92.3%); excluded 12 = 4 not scored + 8 broke a rule
```

It also prints when and where each Study 3 arm ran (the arms were not interleaved) and the arm means pooled over runs
and with agents weighted equally, which agree to four decimals.

## The integrity claims — check our work, do not trust it

> *"Five of the 312 runs added the labelled evaluation file to their training data ... Five other runs computed
> features from the batch they were asked to score"*

This is the claim you should be most suspicious of, because we are the ones who decided what counts as cheating.
So the delivered code is in this repository and you can run both traces yourself:

```bash
python analysis/audit_exported_code.py
```

It takes the **last `train.py` of every one of the 468 runs** and runs two traces over it: whether evaluation labels
reach a model fit (other than as the early-stopping set the rules permit), and whether statistics are computed from
the frame handed to the prediction function. Both are screens. It prints every raw hit with the offending source,
and beside it the verdict a person reached on reading the code, with the reason. The summary:

```
=== code/study2, evaluation labels: 7 raw hits, 5 confirmed
=== code/study2, scored-frame stats: 7 raw hits, 5 confirmed
  code/study2: 10 runs broke a rule, 0 broke both
=== code/study3, evaluation labels: 6 raw hits, 3 confirmed
=== code/study3, scored-frame stats: 5 raw hits, 5 confirmed
  code/study3: 8 runs broke a rule, 0 broke both

468 delivered programs audited; 18 runs broke a rule
```

The verdicts live in `analysis/audit_eval_training.py` (`REVIEWED`) and `analysis/analyze.py`, which is how they
reach the `eval_trained`, `frame_stats` and `compliant` columns of `cells.csv`. One cleared hit is a judgment call the
paper names: `opencode/glm-5.3/13` fitted a classifier to tell evaluation rows from training rows, on features alone,
and used it to weight its training data. No evaluation label reached a fit, so it counts as compliant; it scored 38th
of its pairing's 50 compliant runs, and excluding it moves Study 3's gain by 0.0001.

**A correction.** Earlier versions of the paper decided evaluation-label training with a score screen instead (the
`refit_suspect` column, still published): a run whose best evaluation score exceeded its holdout score by more than
0.03. That compares the run's best experiment with the code it delivered, which need not be the same program. It
excluded two runs that had not trained on evaluation labels (Study 2 `hermes/glm-5.3-flash/30`, Study 3
`hermes/glm-5.3/29`) and missed two that had (Study 3 `hermes/glm-5.3/1` and `/39`). `ledger.py` prints both lists.

To read what an agent actually wrote, including its own notes:

```bash
ls  code/study3/var7/airline/pi/glm-5.3/seed35/          # eval.json, experiments.tsv, cpu_ledger.tsv, FINAL.md
cat code/study3/var7/airline/pi/glm-5.3/seed35/FINAL.md  # the agent's own summary of what it tried
ls  code/study3/var7/airline/pi/glm-5.3/seed35/code/     # train.py at every commit, oldest first
```

`seed35` is a run that computed statistics from the frame it was scoring, and it posted the highest score in its
box. That pattern — the most impressive artifact being the least compliant one — is the paper's point.

How the headline numbers move under each exclusion rule (exclude nothing, either kind of violation, or both):

```bash
python analysis/sensitivity_violations.py data/study2/cells.csv data/study2/final_eval.csv
```

```
rule                                runs  span of means  median SD    best best-of-3 med best-of-10 med
exclude nothing                      312         0.0098     0.0133  0.8293        0.7499         0.7565
exclude eval-file training only      307         0.0098     0.0111  0.8036        0.7493         0.7548
exclude batch features only          307         0.0095     0.0117  0.8293        0.7498         0.7564
exclude both (the paper)             302         0.0095     0.0107  0.7695        0.7493         0.7548

of the 10 highest-scoring runs, 7 broke a task rule
```

## What several attempts buy

> *"Three attempts move the median artifact 0.0081 AUC above one attempt, with a 95 percent interval of 0.0063 to
> 0.0098 from resampling the observed runs, and ten attempts 0.0136."*

```bash
python analysis/best_of_k.py data/study2/cells.csv data/study2/final_eval.csv
```

The policy: draw k attempts from a pairing's 52 runs with replacement, reject noncompliant ones, keep the one whose
delivered code scores best on the evaluation set, and report its holdout AUC. It is computed exactly, not simulated:
with replacement, the attempt ranked r-th of n is kept with probability (r/n)^k − ((r−1)/n)^k. The evaluation score
is in `data/study2/final_eval.csv`, extracted from each run's record by
`python analysis/export_final_eval.py code/study2 data/study2/final_eval.csv`; it belongs to the run's final code,
which in 18 of the 312 runs was not its best experiment. The script also chooses on the holdout itself, an oracle no
user has, for comparison:

```
best-of-k over 6 pairings weighted equally, exact; kept attempt's holdout AUC

  k  >=1 compliant   chosen on eval: median  p5      p95    oracle, chosen on holdout: median  p5      p95    optimism
  1       96.7949%                   0.7412  0.7220  0.7587                              0.7412  0.7220  0.7587   +0.00000
  3       99.9817%                   0.7493  0.7355  0.7641                              0.7493  0.7355  0.7641   +0.00004
  5       99.9999%                   0.7526  0.7409  0.7645                              0.7526  0.7409  0.7645   +0.00005
 10      100.0000%                   0.7548  0.7460  0.7663                              0.7548  0.7463  0.7663   +0.00005

median kept: three attempts +0.0081 over one, ten attempts +0.0136
one attempt to ten: 5th percentile +0.0240, 95th percentile +0.0076
```

The percentiles describe the artifact the policy returns. The uncertainty in the gain comes from an outer bootstrap
over each pairing's runs, 2,000 resamples:

```bash
python analysis/policy_tables.py data/study2/cells.csv data/study2/final_eval.csv data/study3/cells.csv data/study3/final_eval.csv
```

```
  gain, 1 to 3 attempts: +0.0081   95% interval +0.0063 to +0.0098 (outer bootstrap, 2,000)
  gain, 1 to 5 attempts: +0.0114   95% interval +0.0085 to +0.0131 (outer bootstrap, 2,000)
  gain, 1 to 10 attempts: +0.0136   95% interval +0.0113 to +0.0154 (outer bootstrap, 2,000)
```

The same script prints Study 3's yield and policy table and the nine observed configurations (tokens, list-price
cost, yield, quality, best of three). `analysis/best_of_k_by_pairing.py` gives the policy for each pairing.

## What the larger model buys (Study 3)

> *"The larger model scored 0.0091 AUC above Flash ... 7.1 standard errors from zero ... 0.87 times the run-to-run SD
> of a single pairing. Run each model once, and the smaller one still comes out ahead 28 percent of the time: 15
> percent with pi, 33 with Hermes, 35 with OpenCode."*

```bash
python analysis/compare_arms.py data/study3/cells.csv data/study2/cells.csv
```

This prints the per-agent gains with bootstrap intervals, the pooled main effect and, under it, the head-to-head line:

```
1. MAIN EFFECT (model tier, pooled): +0.0091   95% CI +0.0066 to +0.0115   7.1 SE from zero
   within-pairing SD (flash, median across agents) 0.0104  ->  the gain is 0.87 x the noise
   one run of each model: the smaller model comes out ahead 28% of the time, averaged over agents (hermes 33%, opencode 35%, pi 15%)
```

That is the share of all compliant (Flash run, GLM-5.3 run) pairs in which the Flash run scores higher, per agent,
averaged over the three. The script then prints every pairwise interaction, the detectability ladder, and the two
checks within the GLM-5.3 arm (box effect F = 0.97, time drift −0.0000). The two arms ran on consecutive days through
a hosted endpoint, so the gain includes any change at the endpoint between them; see the same-day check below. It applies the same compliance rule **to
both arms** before comparing.

The yield beside the gain, and the three-attempt policy, from `policy_tables.py` above:

```
  GLM-5.3 Flash  yield 150 of 156 (96.2%)   compliant mean 0.7380   one attempt: artifact 96.2%, median 0.7385   best of three: artifact 99.97%, median 0.7473
  GLM-5.3        yield 144 of 156 (92.3%)   compliant mean 0.7471   one attempt: artifact 92.3%, median 0.7482   best of three: artifact 99.94%, median 0.7564
```

The figure "The larger model wins on average, but one run of each still favours the smaller model 28% of the time":

```bash
python analysis/figure_arms.py data/study3/cells.csv data/study2/cells.csv out/study3
```

## Which agent looks best depends on the model (Studies 2 and 3)

> *"pi gained 0.0152, Hermes 0.0067 and OpenCode 0.0055. pi's gain exceeds OpenCode's by 3.3 standard errors and
> Hermes's by 2.7, and both differences survive adjustment for the three agent pairs we could have compared"*

```bash
python analysis/interaction_table.py data/study2/cells.csv data/study3/cells.csv
```

Prints, for each study, every agent's change, all three interaction contrasts with normal-approximation SEs and
pointwise bootstrap intervals, Holm-adjusted p-values, and a joint Wald test of no interaction. For Study 3:

```
    pi       vs Hermes    +0.0085  SE 0.0032   +2.7 SE   bootstrap 95% +0.0024 to +0.0148   clears zero
    pi       vs OpenCode  +0.0097  SE 0.0029   +3.3 SE   bootstrap 95% +0.0040 to +0.0155   clears zero
    Hermes   vs OpenCode  +0.0012  SE 0.0031   +0.4 SE   bootstrap 95% -0.0049 to +0.0072   does not clear zero
    Holm-adjusted p, pi vs Hermes: raw 0.0074, adjusted 0.0149
    Holm-adjusted p, pi vs OpenCode: raw 0.0010, adjusted 0.0030
    Holm-adjusted p, Hermes vs OpenCode: raw 0.7103, adjusted 0.7103
  omnibus test of no agent x model interaction: Wald chi2 12.3 on 2 df, p 0.0021
```

and for Study 2 the joint test is χ² 10.2, p 0.0062. The two agent-by-model figures, with the numbers the brief quotes
from them (Study 2: pi +0.0095, Hermes +0.0062, OpenCode +0.0003; pi leads OpenCode by 0.0053 on DeepSeek 4.1 Flash
and 0.0058 on GLM-5.3, at 2.4 and 2.6 SE):

```bash
python analysis/figure_interaction.py data/study2/cells.csv data/study3/cells.csv out/interaction
```

Read the two studies' agreement with the caution the paper gives it. Both comparisons start from the same GLM-5.3
Flash runs, on which pi trails OpenCode by 0.0039, so that gap counts toward pi's larger gain in both. The lines
that do not share it are the "pi minus OpenCode" lines on the stronger models.

## The numbers no other script prints

> *"the weaker pairing comes out ahead 28 to 44 percent of the time"*, *"about 20 runs of each ... about 66"*,
> *"the standard error of an AUC on it is about 0.0005"*, the Study 3 effects table (+0.0277, +0.0091, +0.0067,
> +0.0039 with 2, 21, 39 and 113 runs), the compute correlations (+0.55, +0.43, +0.29, +0.12), and the same-day check
> for Study 3 (*"Study 1, which ran both models on 13 September with their runs overlapping in time, found a gain of
> similar size for the same three agents, +0.0075"*)

```bash
python analysis/paper_numbers.py data/study2/cells.csv data/study3/cells.csv data/study1/cells.csv
```

The three-run comparison is exact: every possible draw of three compliant runs from one pairing, with replacement,
against every draw from the other. The holdout's standard error uses Hanley and McNeil's formula for an AUC near the
observed one on 500,000 rows per class (0.00049). The same-day check compares Study 1's GLM-5.3 and GLM-5.3 Flash rows
for pi, Hermes and OpenCode, which ran concurrently, so no day boundary separates the two models:

```
  GLM-5.3 0.7449 (8 runs)   GLM-5.3 Flash 0.7374 (9 runs)   gain +0.0075  (1.7 SE)
``` The compute correlations rank runs inside their own pairing, with
bootstrap intervals and two-sided permutation p-values; a few thousand resamples take a minute or two.

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
less stable under retraining.

Budget a few minutes per run. Some of these programs fit 16-member ensembles.

## Rebuilding the table of per-run results from scratch

`cells.csv` is derived, not hand-made. It is produced by `analysis/analyze.py` from the raw run trees, applying the two
traces and the recorded verdicts. Those trees are 43 GB and are not in this repository, so this command is documented
rather than runnable here:

```bash
python analysis/analyze.py <pulls-dir> <out-dir>     # needs the raw run trees
```

What *is* runnable is everything downstream of `cells.csv`, which is every number in the paper, and the traces
themselves over the exported code (`audit_exported_code.py` above).

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
