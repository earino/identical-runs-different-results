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
row per run, 312 of them, with the holdout score, the flags and the budget accounting.

## What the larger model buys (Study 3)

> *"The larger model scored 0.0097 AUC above Flash ... 7.2 standard errors from zero ... 0.93 times the run-to-run SD"*

```bash
python analysis/compare_arms.py data/study3/cells.csv data/study2/cells.csv
```

This prints the per-agent gains with bootstrap intervals, the pooled main effect, every pairwise interaction with a
"clears zero / does not clear zero" verdict, the detectability ladder, and the two confound checks (box effect
F = 1.27, time drift +0.0000).

The script applies the two rule-breaking screens **to both arms** before comparing. That matters: an earlier interim
analysis of ours compared screened Flash runs against unscreened GLM-5.3 runs and overstated the gain by about a
thousandth. The docstring records it.

Figure 3 in the paper:

```bash
python analysis/figure_arms.py data/study3/cells.csv data/study2/cells.csv out/study3
```

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

## What is not here, and why

**The holdout set.** Scores are in `cells.csv`; the holdout itself is withheld so the task stays usable for future
evaluation. Publishing it would contaminate it. The source is public — Data Expo 2009 airline on-time data, Harvard
Dataverse `doi:10.7910/DVN/HG7NV7` — and `task/airline/meta.json` states the exact slice we used
(train = 2005 slice 1, 100k rows; eval = 2006 slice 1, 100k; holdout = 2006 slice 2, 1M), so it can be rebuilt.

**The raw run trees**, 43 GB of harness logs and container state. The parts a reader needs — delivered code at every
commit, per-experiment records, CPU ledgers, the agents' own notes — are extracted into `code/`.

**Credentials and machine images.** Nothing here contains a key.
