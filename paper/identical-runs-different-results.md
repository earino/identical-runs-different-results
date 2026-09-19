---
title: "Identical Runs, Different Results: Benchmarking AI Coding Agents on Open-Weight Models"
author: "Eduardo Ariño de la Rubia and Szilard Pafka"
date: "September 2026"
---

> **If your team deploys AI agents:**
>
> - **Do this.** Evaluate the agent and the model as one system, over several attempts. Reject the runs that broke
>   your rules, rank the compliant ones on held-back data using the objective you actually care about, read the code
>   of the leader, then measure that one model on a set you have never used.
> - **Why.** On this task, repeated runs of one pairing produced materially different models, and they still did
>   after the rule-breaking runs were removed. Three-run comparisons ranked pairings unreliably, and the most
>   extreme scores were noncompliant.
> - **Then.** Price the model you chose with its predictions at your own prevalence and costs, not with its AUC.
>   Test again when the model, the agent version, the endpoint, the cache or the price changes.

## Bottom line

Three different questions get confused when agents are compared, and this paper separates them: how good a
compliant artifact is, how often an attempt produces one, and what a policy of several attempts plus selection
delivers.

We report three studies on one task. The first is broad: six coding agents on six open-weight model endpoints, three
runs each. The second is deep: three agents on two models, 52 runs each, 312 runs in all. The third is controlled:
the same three agents on a larger model from the same family, 52 runs each, with the model as the planned change. In
every study the agents improve a model that predicts whether a flight departs late, and a holdout they never see
scores it by AUC: the chance that the model ranks a late flight above an on-time one. A run is compliant when its
code obeyed the task rules, training only on the training file and computing no feature from the batch it was
scoring.

**Quality of compliant runs.** In the deep study, the six pairing averages over compliant runs span 0.0095 AUC,
while the median pairing varies by 0.0107 across its own compliant runs. The spread inside a pairing is larger than
the spread between pairings, so three-run comparisons ranked them unreliably. That is a statement about the
reliability of small comparisons, not evidence that the agents are equivalent.

**Rate of compliant runs.** 10 of 312 runs broke the task rules: five trained on the labelled evaluation file,
five computed features from the batch they were scoring. They sit at the top of the score table: the seven highest
scores in the study all broke a rule. Removing them takes the best score from 0.8293 to 0.7695, so the most
impressive artifact was the least compliant one.

**What a policy delivers.** One attempt returns a compliant artifact 97 percent of the time, with a median holdout
AUC of 0.7412. Three attempts, keeping the compliant one that scores best on the evaluation set, return one
with an estimated probability of at least 99.88 percent, with a median of 0.7493, and ten attempts reach 0.7548. Repeat runs buy a better artifact
cheaply; they do not buy a reliable ranking of vendors.

**What a real upgrade is worth.** Runs on the larger model in the same family scored 0.0091 AUC higher, seven
standard errors from zero but only 0.87 times the run-to-run SD. The benchmark detects a genuine change; the agents are
simply closer together than that. How much the upgrade bought also depended on the agent: 0.0152 for pi against
0.0055 for OpenCode, 2.8 times less from the identical change.

**What carries into a later year.** Scored again on one million flights from 2007, which nothing in the study had
used, the same artifacts kept a third of their gain over the starting code, and the larger model's advantage shrank
to 0.0027 AUC. Every finding keeps its direction, but the sizes above belong largely to the year the agents tuned on.

Running cost depended on the pairing too, by up to 23 times for the same job. We report it separately, after the
three studies.

## The task is machine learning; the lessons are about agents

We used a machine-learning task because it makes an agent's work measurable. The task: an XGBoost model predicts
whether a US airline flight will leave at least 15 minutes late from eight fields such as carrier, route and
departure time, and the agent edits its training code to make it predict better. A hidden holdout of later flights
scores the result. The departure time is the time the flight actually left, as in the public benchmark the data
come from, so this is a fixed optimisation problem for comparing agents, not a forecast one could make before
departure. Nobody grades it by hand, the score has room to rise, and each run is cheap to repeat. Most work
that companies automate gives no such score. The variation between runs still exists there, but nobody can see it.

This task exposes four risks that teams should test for in their own deployments: results vary between identical
attempts; some attempts break the rules of the task; agents differ in how much effort they spend; and whether a
prompt cache works depends on the agent and the endpoint together. We show that all four appear here. We have not
established how common or how important they are in agentic work generally.

Most comparisons of AI agents ask which one is best. On a task where that question has a measurable answer, it was
not the question that mattered.

# Study 1: six agents, six model rows, three runs each

## Run-to-run variation made three-run quality rankings unreliable

On four of six model rows, three runs per agent could not order the agents: the gap between the best and worst
agent average is about the size of the gap between two runs of a single agent. This does not show that the agents
perform equally. It shows that a three-run comparison cannot tell you which is better. On the other two rows the
gap came from a single agent that spent little of its compute budget.

| Model row | Agent gap | Run-to-run gap | Ratio |
|:--|--:|--:|--:|
| DeepSeek V4 Flash, Ollama | 0.021 | 0.021 | 1.0 |
| DeepSeek V4.1 Flash, Ollama | 0.019 | 0.018 | 1.0 |
| Nemotron Super, Ollama | 0.011 | 0.010 | 1.1 |
| DeepSeek 4.1 Flash, LunaRoute | 0.036 | 0.027 | 1.3 |
| GLM-5.3 Flash, LunaRoute | 0.022 | 0.012 | **1.9** |
| GLM-5.3, LunaRoute | 0.032 | 0.011 | **3.0** |

Table: Holdout AUC, three runs per agent on each model row.

Quality and reliability are separate outcomes, so we report them separately. Of 108 scheduled runs on the six
complete rows, three never started because the gateway rejected Codex's request format, and two more delivered
artifacts that broke the task rules. That leaves 103 scored. The three that never started are excluded from the AUC
comparison, because there is no artifact to score, but they are a property of that deployed pairing: on DeepSeek
4.1 Flash through LunaRoute, Codex completed none of its three attempts.

| Model rows | Scheduled | Started | Delivered an artifact | Compliant |
|:--|--:|--:|--:|--:|
| Five complete rows | 90 | 90 | 90 | 88 |
| DeepSeek 4.1 Flash, LunaRoute | 18 | 15 | 15 | 15 |
| DeepSeek V4 Pro, Ollama (partial) | 8 | 8 | 8 | not assessed |

Table: Study 1 run flow. The V4 Pro row was stopped at 85 percent of a weekly quota, four of its runs cut off mid-run; it appears only as a supporting observation about caching, in the cost section. 116 runs in all: 113 delivered an artifact, the three that never started left the starting code in place, and 103 were scored in the quality analysis.

# Study 2: one pairing, fifty-two times

Three runs are enough to see that the variation exists. They are not enough to measure it, and not enough to rank
anyone. So we took three open-source agents and the two fastest models of the first study and ran each pairing 52
times, holding the data, prompt, budget, machine type and parallelism fixed: 312 runs. The three agents had the
highest three-run averages on the LunaRoute rows of Study 1, which is how we chose them rather than a finding:
Study 1 could not rank them reliably.

## Another benchmark averages the spread away

A Berkeley team published a related study while this one was running. HarnessTax (Pan and colleagues, September
2026, harnesstax.github.io) paired seven models with three agents — Claude Code, Codex CLI and pi — on SWE-bench
Lite and Terminal-Bench 2.0. The designs differ where our main question lives. HarnessTax runs each pairing three
times per task and averages the attempts, so it does not report how far repeated identical runs spread. That spread
is what our second study measures, and it is wide enough to reorder agents. Their benchmarks also sit near their
ceiling: the best pairing solved 97.8 percent of its attempts on a 30-task sample of SWE-bench Lite, which leaves a
harness difference little room to show. They note the models may have seen those tasks in training. Our task is
graded on a continuous score against a holdout the agent never sees, so differences have room to appear and
rule-breaking stays detectable. HarnessTax's cost finding agrees with ours, and we return to it after the three
studies. The reliability problem we report is what you meet as soon as you try to measure any difference between
agents.

## A single run is a draw, not a result

A single run does not give you an agent's average quality. It gives you one draw from its distribution. Among
compliant runs, the six pairing averages span 0.0095 AUC while the median pairing varies by 0.0107 across its own
runs. Two runs of the same pairing differ by 0.0147 AUC on average over all runs, and by 0.0285 at the ninth
decile. One run delivered a model worse than the code it started from, and it had broken a task rule. Every
compliant run beat the starting code, though the weakest by only 0.0007.

![Compliant runs only: one mark per run, one row per pairing, in the order of the table below. Colour is the agent and shape the model, as in every figure here. Black dot and bar: the mean and its 95 percent interval. Red dots: the 10th and 90th percentiles, so the middle 80 percent of runs lie between them. Grey bar: the full range. Dotted line: the starting code. At right: the SD and number of compliant runs, and the share of the pairing's 52 runs that broke a task rule. Those 10 runs are left out of the plot; they scored 0.7116 to 0.8293 and are described below.](fig/fig10_variance_spread.png)

| Agent and model | Compliant runs | Mean | Median | SD | 95% interval of the mean | Best | All-run mean |
|:-----------------------------|----:|-------:|-------:|------:|:-----------------:|------:|------:|
| pi, GLM-5.3 Flash | 52 | 0.7361 | 0.7359 | 0.0104 | 0.7332 – 0.7389 | 0.7597 | 0.7361 |
| Hermes, GLM-5.3 Flash | 47 | 0.7379 | 0.7404 | 0.0112 | 0.7347 – 0.7411 | 0.7590 | 0.7427 |
| OpenCode, GLM-5.3 Flash | 51 | 0.7400 | 0.7402 | 0.0097 | 0.7373 – 0.7427 | 0.7596 | 0.7417 |
| OpenCode, DeepSeek 4.1 Flash | 52 | 0.7403 | 0.7418 | 0.0096 | 0.7377 – 0.7429 | 0.7578 | 0.7403 |
| Hermes, DeepSeek 4.1 Flash | 49 | 0.7441 | 0.7437 | 0.0109 | 0.7411 – 0.7472 | 0.7663 | 0.7459 |
| pi, DeepSeek 4.1 Flash | 51 | 0.7456 | 0.7462 | 0.0125 | 0.7421 – 0.7490 | 0.7695 | 0.7455 |

Table: Compliant runs only, except the last column. The starting code scores 0.7148 AUC and 0.7041 average precision. Removing the 10 noncompliant runs lowers the study's best score from 0.8293 to 0.7695 and the median pairing SD from 0.0133 to 0.0107.

## Three runs cannot tell you which agent is better

Suppose each pairing contributes three compliant runs, drawn at random, and two pairings on the same model are
compared by their averages. Counting every possible draw, the weaker pairing comes out ahead 28 to 44 percent of
the time, and the two closest comparisons are near a coin toss. A three-run report can state either order.

How many runs it would take depends on which difference you want to detect, and the answer is not portable. Under
the conventional approximation, detecting the widest gap among these six pairings, 0.0095, would take about 20 runs
of each. But that contrast changes both the agent and the model. The widest gap between two agents *on the same
model* is 0.0053, and detecting that would take about 66 runs of each. Smaller differences need more. These are
also differences we chose after seeing the data, which flatters them; a comparison fixed in advance should be sized
on the difference that would change your decision, not on the largest one observed.

## Which agent looks best depends on the model

The six pairings are three agents crossed with two models, so their means can be read both ways. Across agents, the
three sit within 0.0039 AUC of each other on GLM-5.3 Flash, with OpenCode highest and pi lowest, and within 0.0053
on DeepSeek 4.1 Flash, where the order reverses: pi highest, OpenCode lowest. Across models, moving from GLM-5.3
Flash to DeepSeek 4.1 Flash raised pi by 0.0095, Hermes by 0.0062 and OpenCode by 0.0003. pi's gain exceeds
OpenCode's by 0.0092, 3.1 standard errors of that difference, which survives adjustment for the three agent pairs
we could have compared.

![The six Study 2 means read two ways, compliant runs. Left: agents on the axis, one line per model, labelled with how far apart the agents are on it. Right: models on the axis, one line per agent, labelled with its change. Colour is the agent and shape the model. Thick bar: the 95 percent interval of the mean. Pale band: where the middle 80 percent of single runs land. The two measure different things, and overlapping bars are not a test; the text gives the tests.](fig/fig14_interaction_study2.png)

Two cautions. Neither end of the reversal is strong on its own: pi trails OpenCode by 2.0 standard errors on GLM-5.3
Flash and leads it by 2.4 on DeepSeek 4.1 Flash. And we noticed the pattern after looking at the data, across two
models from different families, so it is a pattern to test, not a finding. The pale bands make the section's
larger point again: each of the six means lies inside every pairing's middle 80 percent of runs, so a comparison
of single runs is unreliable here. A ranking of agents is a claim about agents on one model.

## The tails are method choices, and some of them break the rules

The agents do not merely jitter. They sometimes take a different approach, and a few of those approaches are not
allowed by the task. Five of the 312 runs added the labelled evaluation file to their training data; four of those
were Hermes on GLM-5.3 Flash, and the five scored 0.7855 to 0.8293, all among the six highest scores in the study.
Five other runs computed features from the batch they were asked to score, which the task also forbids, and scored
0.7116 to 0.8036.

Both are rule violations rather than contamination of the hidden test. The holdout stayed hidden in every case: no
run read it, and every score comes from re-running the delivered code against it. A run that trained on the
evaluation file had more labelled data than the rules allow, from the same year as the holdout, so its score may be
a real gain obtained with unauthorized data rather than an illusion. A run that computes features from the scoring
batch depends on batch composition at prediction time, which this task forbids; whether that is deployable in your
own system depends on whether the batch is available when you predict.

This is why selection needs an audit. The noncompliant runs sit at the top of the score ranking, so the run you
would pick on score alone is the run most likely to have broken a rule. Picking the maximum of 52 scores on one
holdout also turns that holdout into a selection set, and the winner's margin is then part real and part luck.

Detection is the weakest remedy. Two checks find these after the fact: search the delivered code for evaluation
labels reaching training data, and trace whether any feature is computed from the frame handed to the prediction
function. Prevention is better. Put the evaluation labels behind a scoring interface, so unauthorized training is
impossible rather than detectable. Test whether a row's prediction changes when the batch around it changes, which
catches batch-dependent features without reading any code.

## What several attempts buy

The advice above is to attempt more than once, reject the noncompliant runs and keep the best of the rest. That
policy can be measured on the observed runs. Draw k attempts from a pairing's 52 runs, reject the noncompliant
ones, keep the one whose delivered code scores best on the evaluation set, and score it on the holdout, which no
agent saw:

| Attempts | At least one compliant | Median kept | 5th percentile | 95th percentile |
|:--|--:|--:|--:|--:|
| 1 | 96.8% (94.2 to 98.2) | 0.7412 | 0.7220 | 0.7587 |
| 3 | at least 99.88% | 0.7493 | 0.7355 | 0.7641 |
| 5 | at least 99.99% | 0.7526 | 0.7409 | 0.7645 |
| 10 | at least 99.99% | 0.7548 | 0.7460 | 0.7663 |

Table: The best compliant artifact among k attempts, the six pairings weighted equally, computed exactly over the observed runs rather than by simulation. The first column is the chance of at least one compliant attempt: for one attempt the observed rate with its 95 percent interval, for more the 95 percent lower bound. The percentiles describe the spread of the artifact the policy returns, not uncertainty about its median. Choosing on the holdout itself instead, an oracle no user has, would raise the mean kept score by at most 0.00005 AUC: a score on one million rows has a standard error twenty times smaller than the spread between runs, and a run's evaluation and holdout scores rank a pairing's runs almost identically (rank correlation +0.99).

Three attempts move the median artifact 0.0081 AUC above one attempt, with a 95 percent interval of 0.0063 to
0.0098 from resampling the observed runs, and ten attempts 0.0136. The returns fall away quickly, and the floor
rises faster than the ceiling: from one attempt to ten, the 5th percentile improves by 0.0240 and the 95th by
0.0076. Attempts mostly buy protection against a bad draw. At the token prices of this study that protection costs
roughly \$2 for three attempts and \$8 for ten, before the audit work, which is the real cost. The analysis is
retrospective: we specified the policy after the runs, and the holdout it is scored on was hidden from the agents
but not from us.

That is the sense in which choosing one good artifact is cheaper than ranking two vendors: three to ten attempts
against roughly 66 runs of each pairing.

## Budget use predicts score inside a pairing

In both studies, the runs that spent more of their measured compute scored higher. The broad study found a rank
correlation of +0.59 across 103 runs, where it could have meant that some agents simply work harder than others.
The deep study holds the agent and the model fixed: within a pairing the correlation is **+0.55** over 312 runs,
with a 95 percent interval of +0.45 to +0.63, and the same over compliant runs alone. Every pairing points the
same way, from +0.38 to +0.70.

![Budget use against score inside each pairing, compliant runs only. Colour is the agent and shape the model. Dotted line: the starting code.](fig/fig12_variance_compute.png)

What the meter shows is CPU seconds consumed by Python inside the container. It is not a measure of search effort.
Every one of the 95 runs that spent under a quarter of the budget still ran all 40 of its counted experiments: they
did not stop early, they used cheaper methods. Nor does the meter see the model's own generation, or reasoning the
agent does in context. A low reading is a fact about resource use and a reason to read the run's experiment trace,
not evidence of carelessness.

The association is weaker and less precisely estimated among high-budget runs: +0.43 among compliant runs above a
quarter of the budget, +0.29 above half, and +0.12 above three quarters with an interval spanning zero. Narrowing
the sample also narrows the predictor and shrinks the sample, so this is not by itself a demonstration of
diminishing returns.

# Study 3: the same agents, a larger model

## The larger model's gain is about the size of run-to-run noise

A third study switched the model and measured the difference. GLM-5.3 is the larger model in the family whose Flash
version Study 2 used. We ran the same three agents against it, 52 runs each, with the same data, prompt,
budget, machine type and parallelism: 156 runs, made the day after Study 2's GLM-5.3 Flash runs, through the same
gateway.

The larger model scored **0.0091 AUC above Flash** over the three agents' compliant runs, with a 95 percent interval
of 0.0066 to 0.0115. That is 7.1 standard errors from zero. It is also **0.87 times the run-to-run SD of a single
pairing**. Run each model once, and the smaller one still comes out ahead 28 percent of the time: 15 percent with
pi, 33 with Hermes, 35 with OpenCode.

This answers a fair question about the first two studies. When a benchmark reports that agents do not separate, a
reader should ask whether it can detect anything at all. It can. One genuine change moved the score by seven
standard errors on the same design that could not rank three agents. The agents are close together; the
measurement is not blunt.

![Left: every compliant run of both models, three agents, coloured by agent; hollow circles are GLM-5.3 Flash and squares GLM-5.3. Black marks are the averages; the dotted line is the starting code. Right: the gain from the larger model with its 95 percent interval, against a shaded band one run-to-run SD wide.](fig/fig13_pro_vs_flash.png)

| Agent | GLM-5.3 | Runs | GLM-5.3 Flash | Runs | Gain | 95% interval |
|:---------|-------:|----:|-------:|----:|-------:|:---------------:|
| pi | 0.7513 | 47 | 0.7361 | 52 | +0.0152 | +0.0110 – +0.0193 |
| Hermes | 0.7446 | 47 | 0.7379 | 47 | +0.0067 | +0.0021 – +0.0113 |
| OpenCode | 0.7455 | 50 | 0.7400 | 51 | +0.0055 | +0.0016 – +0.0094 |
| **All three** | **0.7471** | 144 | **0.7380** | 150 | **+0.0091** | +0.0066 – +0.0115 |

Table: Holdout AUC by agent for the two models, compliant runs only, under the same rules for both. The last row pools the compliant runs; weighting the three agents equally gives the same means to four decimals.

Quality is half of what a deployment needs; the other half is how often an attempt yields a compliant artifact. The
larger model yielded one in 144 of its 156 runs, 92.3 percent, against 150 of 156 for Flash, 96.2 percent. The
counts are too small to say the larger model is less reliable, but they are the right thing to report beside the
gain. Under the three-attempt policy of Study 2 both models return an artifact with an estimated probability of at
least 99.6 percent, and the larger model's median artifact is 0.0091 higher, 0.7564 against 0.7473.

**How much the upgrade buys depends on the agent it is paired with.** pi gained 0.0152, Hermes 0.0067 and OpenCode
0.0055. pi's gain exceeds OpenCode's by 3.3 standard errors and Hermes's by 2.7, and both differences survive
adjustment for the three agent pairs we could have compared; Hermes and OpenCode cannot be told apart. The same
model upgrade, on the same task under the same budget, was worth more than twice as much in one agent as in either
of the others. Buying a better model is not a decision about the model alone.

![The Study 3 means read the same two ways as in Study 2, on the same scale. Left: agents on the axis, one line per model. Right: models on the axis, one line per agent, labelled with its gain. Colour is the agent and shape the model; thick bar: the 95 percent interval of the mean; pale band: the middle 80 percent of single runs.](fig/fig15_interaction_study3.png)

**Study 2 showed the same ordering, but the two are less independent than they look.** There too, pi gained most
from the stronger model and OpenCode least. Both comparisons start from the same GLM-5.3 Flash runs, on which pi
trails OpenCode by 0.0039, and that gap counts toward pi's larger gain in both studies. What the two studies add
separately is that pi leads OpenCode on each stronger model, by 0.0053 on DeepSeek 4.1 Flash and 0.0058 on
GLM-5.3, at 2.4 and 2.6 standard errors. The pale bands carry this study's point as well: pi's gain is 7.1
standard errors from zero in the means, yet pi's runs on the two models overlap across the middle of both bands.

**What the study held fixed.** Neither arm set a reasoning level. Both GLM models document the same default,
thinking enabled at the top effort setting, and a probe of the endpoint confirmed it: with no setting sent, a short
prompt spends about as much thinking as the maximum and roughly ten times what the lowest setting spends. The arms
differ in model scale, not in how hard the model was asked to think. Scores did not drift within the arm: its
first and second halves average the same to four decimals. What we could not hold fixed is the hosted endpoint, and
no one using a hosted model can. The Flash runs were made on 15 and 16 September and the GLM-5.3 runs on 16 and 17
September, so the gain includes whatever changed at the endpoint in between. Study 1, which ran both models on 13
September with their runs overlapping in time, found a gain of similar size for the same three agents, +0.0075 from
eight and nine runs, so a shift between days is an unlikely explanation. Of the 156 runs, four delivered code
that failed when scored on the holdout and eight broke a task rule, three by training on evaluation labels and five
by computing batch features; all twelve are excluded above, by the rules Study 2 uses. Two compliant runs
delivered the starting code unchanged, and they are kept at its score.

**Putting the three effects on one scale** gives the resolving power of this benchmark, in units of the run-to-run
SD and in the number of runs each effect would need:

| Effect | Size | In SDs | Runs per arm |
|:-----------------------------------------|-------:|-----:|----:|
| starting code to what an agent delivers | +0.0277 | 2.7 | 2 |
| GLM-5.3 Flash to GLM-5.3 | +0.0091 | 0.9 | 21 |
| widest gap between two agents, on GLM-5.3 | +0.0067 | 0.6 | 39 |
| widest gap between two agents, on GLM-5.3 Flash | +0.0039 | 0.4 | 113 |

Table: Effect sizes against the median within-pairing SD of 0.0104, with the run count each would need under the
approximation used above. These are the contrasts we observed, not general requirements. The first row compares
against a fixed number rather than a second noisy arm, so its count is a lower bound.

Three runs answer one question reliably: did the agent improve on the code it started from. The larger model's gain
takes about 21. Telling two agents apart on a fixed model takes 39 to 113 for the gaps observed here. Published agent
comparisons typically repeat each task one to three times, which suits averages over many tasks but cannot resolve
differences this small on any one of them.

# A later year

## A third of the gain carries forward

Every score above comes from a holdout drawn from 2006, the year of the evaluation set the agents tuned on. To see
how much of what they delivered carries forward, we retrained all 464 scored programs of Studies 2 and 3 exactly as
the scorer does and applied each to one million flights from 2007, prepared the same way and used by no earlier
analysis. The same fits reproduced the recorded 2006 scores (median difference 0.0002 among compliant runs), and the
starting code scores 0.7175 on 2007, slightly above its 0.7148 on 2006, so the later year is not harder in itself.

| | 2006 holdout | 2007 flights |
|:----------------------------------------------------|---------------:|---------------:|
| Gain over the starting code, 446 compliant runs | +0.0280 | +0.0092 |
| Compliant runs below the starting code | 0 | 33 |
| Spread of the six Study 2 pairing means | 0.0095 | 0.0021 |
| Median run-to-run SD, Study 2 | 0.0107 | 0.0060 |
| Best of three attempts over one | +0.0081 | +0.0022 |
| Larger model's gain | +0.0091 (7.1 SE) | +0.0027 (3.8 SE) |
| pi's gain minus OpenCode's | +0.0097 (3.3 SE) | +0.0023 (1.3 SE) |
| Rule-breakers among the ten highest, Studies 2 and 3 | 7 and 6 | 7 and 6 |

Table: The same delivered programs on the 2006 holdout and on 2007 flights. The attempts policy is the one above, chosen on the evaluation set; its 2007 gain has a 95 percent interval of +0.0010 to +0.0032.

A third of the agents' gain survives. The rest was specific to 2006, the year both the evaluation set and the holdout
come from, and a run's two scores are only loosely related: inside a pairing their rank correlation is +0.38. Every
finding keeps its direction but shrinks. Runs of one pairing still vary more than the pairings differ. The
rule-breaking runs still hold the top, and those that trained on the evaluation labels stay well above the compliant
runs, consistent with a real gain from more recent data. Three attempts still buy a better artifact, by about a
quarter as much. The larger model's gain remains detectable but falls to 0.44 run-to-run SDs, about 82 runs per arm
to detect, and pi's larger share of it no longer separates from the other agents'.

So the sizes in this paper describe the year the agents tuned on. That is what a team measures when it validates on
the period it selected on, and it is why confirming a chosen model on later, untouched data matters more than any
single number here. We did not test why the gain shrinks.

# What a difference is worth

## Price it with predictions, not with AUC

Do not convert an AUC gap into money. AUC summarises the ranking across every threshold, so a 0.05 gap does not
tell you what changes at the one threshold you use. Score the candidate models on held-back data, take their
predictions, apply the decision rule you actually run, and count true positives, false positives and false
negatives at your own prevalence and costs.

Here is that calculation on two of our own models, the best compliant run of a pairing and its median run, 0.7695
and 0.7471 AUC. Suppose one million decisions a month, one in a hundred of them a true positive, a team that can
review a fixed number of cases, \$10 for a false alarm and \$100 for a miss. We assume that only the prevalence
differs from the benchmark and that the score distribution within each class is unchanged; under that assumption
the true-positive and false-positive rates measured here carry over, and the counts below follow.

| Cases reviewed each month | Caught by the 0.7695 model | Caught by the 0.7471 model | Difference |
|:----------|-------:|-------:|-------:|
| 5,000 | 562 | 580 | −\$1,920 |
| 20,000 | 1,351 | 1,279 | \$7,955 |
| 100,000 | 3,714 | 3,421 | \$32,156 |

Table: Illustrative, not a forecast. Both models were selected and scored on the same holdout, with no second untouched evaluation, so these differences are not validated prospective savings. The point is the crossing: the better model by AUC is worse at the tightest capacity.

That crossing is also an instruction. Rank your candidates on the objective you will actually use, on a selection
set, and then measure the one you chose on untouched data. Selecting on AUC and pricing afterwards can hand you the
artifact that loses money at your operating point.

Read our average precision figures the same way. AP depends on how common the positive class is, and this holdout
was built with the two classes in equal numbers, while real departure delays are far rarer. So these AP values
compare runs on this benchmark fairly and are not the precision you would see in production. You cannot rescale
them either: true-positive and false-positive rates carry across to a different prevalence under the assumption
above, precision does not, so recompute it from the predictions at your own rate.

# What it costs

Cost is a separate outcome from quality, and it also depended on the pairing.

## Cost moved with the pairing

In Study 1, on DeepSeek V4 Flash, rate-card cost followed the prompt cache. The endpoint served 92 to 98 percent of
pi's and Codex's input from cache, and 2 to 11 percent of Claude Code's. Two effects combined. Repricing Claude
Code's own token volume at pi's hit rate gives \$0.20 a run, against its actual \$1.86. So the cache accounts for
about nine times, and Claude Code's larger token volume for about two and a half.

| Agent | Run 1 | Run 2 | Run 3 | Mean cost |
|:-------------|-------------:|-------------:|-------------:|-----------:|
| pi | 98%, \$0.08 | 98%, \$0.11 | 97%, \$0.05 | \$0.08 |
| OpenClaw | no record | 98%, \$0.11 | 97%, \$0.07 | \$0.09 |
| Codex | 96%, \$0.14 | 96%, \$0.19 | 92%, \$0.13 | \$0.15 |
| OpenCode | 80%, \$0.24 | 64%, \$0.19 | 66%, \$0.49 | \$0.31 |
| Hermes | 56%, \$1.09 | 89%, \$0.47 | 74%, \$0.24 | \$0.60 |
| Claude Code | 11%, \$1.41 | 9%, \$2.95 | 2%, \$1.21 | \$1.86 |

Table: Cached share of input and rate-card cost per run, DeepSeek V4 Flash, Ollama Cloud. No money changed hands: a flat-rate subscription paid for the runs, and every cost is a rate-card projection.

The cache failure belonged to the pairing, not to either half. Claude Code cached 96 to 97 percent of its input on
DeepSeek V4.1 Flash and 73 to 84 percent on Nemotron Super. Every agent on those two models cached 70 to 99 percent.
The failure did repeat on one more model, DeepSeek V4 Pro, whose row the quota stopped after eight runs: Claude Code
cached 4 and 11 percent while the other five agents cached 83 to 99 percent. The quota cut off both Claude Code
runs, but it also cut off both of pi's, which still cached 91 and 98 percent. On V4 Flash, Claude Code's own request
log shows the cache returning almost nothing from the first request, at 14,000 tokens, and never recovering, while
on V4.1 Flash the same agent reads its prompt back at up to 122,000 tokens. So context size does not explain it.

Lawrence (June 2026) saw Claude Code's cache share collapse through another gateway, OpenRouter, and traced it to the
gateway's handling of Claude Code's Anthropic-style request format; he concluded that cache share belongs to the
whole path from agent through gateway to provider. Claude Code reached Ollama Cloud through the same kind of
Anthropic-style endpoint, so our case fits that conclusion. It does not fit a translation that fails everywhere:
through that same endpoint Claude Code cached normally on V4.1 Flash and Nemotron Super. Whatever failed depends on
the model behind the endpoint as well. A test with byte-identical prompts through both request formats would
separate the client from the server.

Rate-card cost also moved between identical runs. In Study 1, Hermes cost \$0.24, \$0.47 and \$1.09 for the same
work on the same pairing, a swing of 4.6 times. Across the 25 pairings with three complete costs, the median swing
was 2.2 times.

## Another group found the same cost gap

HarnessTax, the Berkeley study described in Study 2, reached the same cost finding on different tasks. Agent choice
barely moved task success there, but it moved cost by as much as five times: across shared models, Claude Code cost
about twice what pi cost on SWE-bench Lite and one and a half times as much on Terminal-Bench 2.0. Two studies with
different tasks, different models and a different metric reached the same conclusion, which makes it firmer than
either study alone: the cost gap is real and shows up on public coding benchmarks.

## What the repeats cost

The 312 runs of Study 2 read 1.86 billion input tokens and wrote 25.1 million, a ratio of about 74 to 1, because
the agent re-reads a growing transcript at every step. Flat-rate subscriptions paid for them. Repricing that same
token ledger at OpenRouter list prices on 16 September 2026 gives about \$244 on the two models we ran.

**A third of what these agents wrote was thinking nobody asked for.** Of the 25.1 million written tokens, 8.7 million
are reasoning tokens. We never set a reasoning level: the configuration sends no `reasoning_effort`, no thinking
budget, and one agent declares that it does not support the field at all. Omitting it is not a neutral choice. Both
GLM models document `max`, the top of their scale, as the default, and a probe of the endpoint confirms it: with no
field set, a short prompt spends about as much thinking as `max` and roughly ten times what `low` spends. Our own
token accounting missed all of it, because every one of these agents reports reasoning separately from output and
we recorded only the output. Check whether yours does the same before you trust a bill or a budget.

| Model | Price per M tokens, in / out | Repriced ledger | If 90 percent of input were cached |
|:--------------------------------|:------------------|--------:|--------:|
| GLM-5.3 Flash and DeepSeek 4.1 Flash, as run | \$0.10 / \$0.33 and \$0.15 / \$0.60 | \$244 | \$54 |
| GLM-5.3 | \$1.40 / \$4.40 | \$2,717 | \$807 |
| Gemini 3.1 Pro | \$2.00 / \$12.00 | \$4,026 | \$1,009 |
| Kimi K3 | \$2.65 / \$13.28 | \$5,265 | \$1,334 |
| Claude Opus 5 | \$5.00 / \$25.00 | \$9,939 | \$2,397 |
| GPT-5.5 Pro | \$30.00 / \$180.00 | \$60,385 | N/A |

Table: Hypothetical repricing of the observed token ledger, reasoning tokens included and billed at the output rate. This is what our tokens would have cost at other prices, not what those models would cost to do this task: another model would produce a different transcript length, completion rate and caching profile, in either direction. Cached-input rates for the last column are listed in appendix D. Token cost is also not the whole bill, which includes the machines and the audit work.

**The pairing drives both the variation and the bill.** Hermes read 8.65 million input tokens per run and pi read
3.93 million, for the same task, the same budget, and compliant averages within 0.006 of each other. Caching then
matters more than the rate card, and on the Study 1 pairing above the hit rate collapsed to 2 percent.

# What to do

On tasks like this one, the evidence supports six practices.

- **Evaluate the pairing, not the parts.** Testing an agent on one model, or a model through one agent, can badly
  mislead you about the combination you will deploy. Record whether attempts started and completed, not only how
  the finished ones scored.
- **Attempt the job several times.** One attempt is a draw. On this task three attempts moved the median artifact
  by 0.008 AUC and raised the worst case by more; ten attempts added little beyond three.
- **Reject first, then rank.** Check every attempt against your rules before you compare scores. Better, make the
  violation impossible: put evaluation labels behind a scoring interface, and test whether predictions change with
  batch composition.
- **Rank on your objective, then confirm on later, untouched data.** Rank compliant candidates on a selection set
  using the decision rule you will run, read the leader's code, then measure that one model once on data you have
  never used. The set you select on always flatters the winner: here a later year kept a third of the gain.
- **Read the experiment trace of any run that used little compute.** It used cheaper methods, which may or may not
  be what you want. It is not necessarily a run that gave up.
- **Instrument the parts nobody looks at.** Record the cached share of input and the compute used on every run. A
  pairing whose cache fails can cost about nine times more on the cache alone.
- **Price the winner with its predictions**, at your prevalence and your costs, and price the failures too: one
  stalled run still cost \$1.21 at rate-card prices and delivered a weaker result.

\newpage

# Appendix A: design

This benchmark builds on our earlier study, published in April 2026, which showed that AI coding agents can automate
XGBoost feature engineering and hyperparameter tuning
([xgboost-autoresearch](https://szilard.github.io/xgboost-autoresearch/)). The task, the data slices and the
starting code come from that study; the repeated-run experiments, audits and analyses here are new.

Each agent was given a model, a dataset, a written task and a compute budget, and asked to improve an XGBoost
classifier by editing one file.

| | Study 1 | Study 2 | Study 3 |
|:----------|:-------------------------|:-------------------------|:-------------------------|
| Question | does the pairing matter? | how much does one run vary? | what does a larger model buy? |
| Agents | Claude Code, Codex, pi, OpenCode, Hermes, OpenClaw | pi, OpenCode, Hermes | pi, OpenCode, Hermes |
| Model rows | six, on Ollama Cloud and LunaRoute | GLM-5.3 Flash and DeepSeek 4.1 Flash, LunaRoute | GLM-5.3, LunaRoute |
| Runs | 3 per pairing: 108 scheduled on the six complete rows, plus an 8-run partial row; 113 delivered an artifact; 103 scored | 52 per pairing; 312, all scored | 52 per pairing; 156, 152 scored |
| Dates | 13 to 15 September 2026 | 15 to 16 September 2026 | 16 to 17 September 2026 |
| Machines | 4 cores, 8 GB per run | four identical 16-core machines, 4 runs each | four more of the same machines, 4 runs each |

All three studies share the task: airline departure delay, binary classification, scored by AUC on a hidden holdout of
1,000,000 rows from 2006; training 100,000 rows from 2005 and evaluation 100,000 rows from 2006. The split is fixed
and identical for every run. A further 1,000,000 flights from 2007, prepared the same way, serve only the later-year
check. Every run had the same prompt, task files, time limits, compute budget and scoring.
Web search was off by instruction. Counted experiments were 40 per run, each a call to the provided run script; 244
of Study 2's 312 runs used all 40, and the median run used 40.

The eight fields are month, day of month, day of week, departure time, carrier, origin, destination and distance.
The departure time is the actual time the flight left, as in the public benchmark the data slices come from,
although the task description given to the agents called it the scheduled time. It carries information a forecast
made before departure would not have. Every agent had the same data and description, so comparisons between runs
are unaffected, but the AUCs here measure improvement on a fixed benchmark, not how predictable delays are.

The holdout has one million rows, half of each class, so the standard error of an AUC on it is about 0.0005, by
Hanley and McNeil's formula with the rows treated as independent draws from 2006's flights. That is a statement
about scoring precision, not about other years or airports. Flights share carriers, airports and days, so the
figure is a lower bound, while runs compared on the same holdout are paired, which makes their differences more
precise. Differences of 0.01 to 0.04 between runs are not measurement error. The holdout's two classes were built in equal numbers, which is not the rate at which flights
are late, so AP figures rank runs here and do not predict production precision.

In Study 2 the seed number is only a label: nothing in the code reads it, and the starting code always uses the
same random seed, so the differences come from the deployed agent, model and endpoint together. Retraining adds
little: re-running a compliant delivered file from scratch reproduced its recorded score to within about 0.0002 in
the runs we checked. Every machine ran all six pairings, so the machine and the pairing cannot be confused; the
estimated machine offsets were all within 0.002 AUC of the pairing means (F = 0.86).

**Exclusions differ between the studies, on purpose.** Study 1 asks which pairing is better, so five runs are
excluded from its AUC tables: three Codex runs that never started because the gateway rejected the request format,
one that concatenated the evaluation rows into its training data, and one whose delivered code computed count
features on the data being scored. The three that never started still appear in the run-flow table, because
failing to start is a property of the pairing. Study 2 asks what the distribution looks like, so every run is
counted, compliant results are reported separately from all-run results, and the figures plot compliant runs only.

**Reproduction.** The data repository,
[github.com/earino/identical-runs-different-results](https://github.com/earino/identical-runs-different-results),
holds the task rules and prompts, agent versions and configurations, model and endpoint identifiers, one row per run
for all three studies, and the holdout. For Studies 2 and 3 it adds every version of the code each agent delivered,
the audit records, and the scoring, statistics and figure code. Its VERIFY.md maps each headline claim to the
command that reproduces it. The full run trees, 43 GB of logs and container state, stay out of it. The flight
data are public; the repository holds our code, the task data and the agents' delivered outputs, and redistributes
no agent software or model weights.

**Built with Claude.** Claude, working in Claude Code, helped build the benchmark, operate the runs and draft this
paper. It also read the code the compliance traces flagged and drafted each verdict, which is published with its
reason. The authors chose the studies and the analyses and are responsible for the verdicts and the text. Claude
Code is also one of the six agents tested in Study 1. Every agent was scored the same way, on the
same hidden holdout.

# Appendix B: the compute budget

Each run had 18,000 CPU seconds of Python compute, with a container-level stop at 46,000. The budget was calibrated
by re-running earlier delivered code on the same hardware.

**What the meter counts.** Every Python process started inside the container, summed over user and system CPU time,
including child processes. A hook in the interpreter records each process on exit. Every model fit is logged
separately with its row count and duration.

**What it does not count.** Time the model spends generating tokens, and reasoning the agent does in context rather
than in code. An agent that plans carefully and fits small models meters as a low user of the budget. Budget use is
therefore a measure of compute invested in fitting and nothing more: in Study 2, all 95 runs that used under a
quarter of the budget nonetheless ran all 40 counted experiments.

**Enforcement.** The hook refuses to start a new process once the budget is spent, and a watchdog stops a running
process that crosses it. Because the watchdog checks every ten seconds, two runs overshot, by at most 17 seconds.
No run was stopped by the container-level cap.

# Appendix C: statistics

**Agent spreads per row, Study 1.** The agent gap is the best agent's three-run average minus the worst agent's.
The run-to-run gap is the median, across a row's agents, of each agent's best run minus its worst. Six agents with
no real difference would produce a ratio of about 0.9 by chance under a normal model. For two agents chosen in
advance, that design separates a difference of about 0.02.

![Holdout AUC, one panel per model row of Study 1. Per agent: the mean of three runs (large mark), the runs (small marks) and their range (bar). pi, Hermes and OpenCode, the agents Studies 2 and 3 follow, keep their colours; the other three are grey.](fig/m_seed_ranges.png)

**Three-run draws, Study 2.** For two pairings on the same model, every possible draw of three compliant runs from
one, with replacement, was compared with every draw from the other; the weaker pairing won 28 to 44 percent of these
comparisons. Sample sizes use 16 sd² / gap²,
the usual approximation for 80 percent power at a 5 percent two-sided test, with the median compliant-run SD of
0.0107: about 20 per pairing for the widest gap among all six pairings (0.0095, which changes agent and model
together) and about 66 for the widest gap between two agents on one model (0.0053).

**Compute against score.** Study 1: rank correlation +0.59 over 103 runs, bootstrap interval +0.45 to +0.71,
permutation p below 0.001. Study 2, within pairings (runs ranked inside their own pairing, then pooled, so
pairing-level differences cannot drive it):

| Subset | Runs | Rank correlation | 95% interval | p |
|:--|--:|--:|:--|--:|
| All runs | 312 | +0.55 | +0.45 to +0.63 | below 0.001 |
| Compliant runs | 302 | +0.55 | +0.46 to +0.63 | below 0.001 |
| Compliant, above a quarter of budget | 209 | +0.43 | +0.29 to +0.54 | below 0.001 |
| Compliant, above half | 110 | +0.29 | +0.10 to +0.45 | 0.001 |
| Compliant, above three quarters | 70 | +0.12 | −0.15 to +0.41 | 0.342 |

Intervals are bootstrap over runs with 2,000 resamples; p-values are two-sided and come from permutation, shuffling
scores within each pairing. The subsets count compliant runs, which is why they sum against 302 rather than 312: 93
compliant runs used a quarter of the budget or less, and 95 of all 312 did.

**An alternative measure of effort.** Fits predict the score far more weakly than CPU seconds: +0.25 over Study 1's
runs and +0.09 within Study 2's pairings. The number of fits says little; the compute those fits consume says more.

**Best-of-k.** The six pairings count equally. Within a pairing, k attempts are drawn from its 52 observed runs
with replacement, noncompliant attempts are rejected, and the compliant attempt whose delivered code scores best on
the evaluation set is kept. With replacement, the attempt ranked r-th of n by that score is kept with probability
(r/n)^k^ − ((r−1)/n)^k^, so the distribution over the observed runs is computed exactly, and the chance of at least one compliant attempt
is the pairings' average of 1 − f^k^, f being a pairing's noncompliant share. The interval for the gain resamples
each pairing's runs 2,000 times and recomputes it; it is conditional on the fixed evaluation and holdout sets. The
same resampling gives the lower bounds on the chance of a compliant artifact, and yields carry 95 percent Wilson
intervals: 96.8 percent (94.2 to 98.2) in Study 2, 92.3 percent (87.0 to 95.5) for GLM-5.3 and 96.2 percent (91.9
to 98.2) for GLM-5.3 Flash. Choosing on holdout AUC instead, an oracle no user has, raises the mean kept score by at most 0.00005.

**Agent-by-model contrasts.** Each difference between two agents' gains is a difference in differences of four
means, with a normal-approximation standard error; p-values are Holm-adjusted over the three agent pairs. A joint
Wald test of no agent-by-model interaction gives χ² = 10.2 on 2 degrees of freedom (p = 0.006) in Study 2 and 12.3
(p = 0.002) in Study 3. The two studies share the GLM-5.3 Flash runs, so their tests are not independent.

**Causal reading.** The compute relationship is correlational and its direction is not established. An agent whose
search is going well may continue, which would produce the same pattern.

**Conventions.** Standard deviations are sample standard deviations, reported to four decimal places.

# Appendix D: money, tokens and audits

**No metered spend occurred.** Flat-rate plans paid for these runs: an Ollama Cloud subscription and flat-rate
access to LunaRoute as a tester. Study 1 costs apply Ollama Cloud's published rate card from 2026-09-11 to the
tokens each run reported. Study 2's repricing applies OpenRouter's published prices from 2026-09-16. Neither plan
issues an invoice to reconcile against. The one external check available is that modelled spend tracked Ollama
Cloud's own weekly quota meter to within about two percent at two separate points in the same week.

**Cached-input prices** used in the last column of the repricing table, per million tokens: GLM-5.3 Flash \$0.02,
DeepSeek 4.1 Flash \$0.003, GLM-5.3 \$0.26, Gemini 3.1 Pro \$0.20, Kimi K3 \$0.303, Claude Opus 5 \$0.50. GPT-5.5
Pro lists no cached-input rate, and an unlisted price is not a zero price, so its entry is left empty. The column
assumes nine tenths of input served from cache and output priced in full.

**The 23 times figure.** It is the ratio of Claude Code's three-run mean to pi's three-run mean on DeepSeek V4
Flash. Computed from unrounded costs it is 22.5, and 23 when rounded. Two of Claude Code's three costs come from
its final usage report and one, for the stalled run, from its request log.

**Per-request validation.** Claude Code keeps a log of every request it sends. Rebuilt from that log, its usage
reproduces the cost of all 15 runs that reported one, to within ten cents. Two cautions apply. Each message is
repeated in the log whenever the conversation is saved again, so entries must be deduplicated by message id. And
the cache-creation field reads zero on every request, so a zero there means nothing.

**Audits.** Every run is scored by re-running its delivered code against the holdout, which no agent sees.
Compliance is decided on the delivered code by two traces, and every hit was read before it counted.
The first follows the evaluation labels: does any reach a model fit, other than as the early-stopping set the rules
permit? The second, added for Study 2, follows statistics computed on the frame handed to the prediction function.

Across Study 1's delivered files, the first trace finds one run, which had concatenated the evaluation rows into
training, and the second finds one, whose count features were computed on the data being scored. Both were
excluded at the time. Three runs used the evaluation set to stop training early, which the rules permit. Study 2's
312 delivered files hold five runs that trained on evaluation labels and five that computed batch features; Study
3's 156 hold three and five. No run did both.

**A screen we do not use.** A simpler check flags any run whose best evaluation score exceeds its holdout score by
more than 0.03. It compares a run's best experiment with the program it delivered, which need not be the same, and
on these runs it would have excluded two compliant runs and missed two that trained on evaluation labels. It is
published with the data but not used.

**Screens are screens.** Both traces over-flag. The frame-statistics trace raised two runs that were cleared on reading,
one that grouped rows only to index them and one that fitted a label-taking encoder which never sees a scored
frame. The label trace raised five that were cleared. Four were tracing errors: the flagged data were training
rows, or the evaluation rows served only as the early-stopping set or the matrix being scored. The fifth, an OpenCode run on GLM-5.3, fitted a classifier to tell
evaluation rows from training rows, on features alone, and used it to weight its training data. No evaluation label
reached a fit, so it counts as compliant under the rule applied here, but it is a borderline case. It scored 38th
of its pairing's 50 compliant runs, and excluding it moves Study 3's gain by 0.0001.
