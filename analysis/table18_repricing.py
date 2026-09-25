#!/usr/bin/env python3
"""Table 18: Study 2's token ledger repriced at other models' list prices (a sensitivity calculation, not a forecast).

Input tokens and generated tokens (output plus reasoning, billed at the output rate) are summed over Study 2's 312
counted runs, as in token_totals.py. "As run" prices each run at its own model's rates. The last column assumes 90
percent of input is read from cache at the model's cached-input rate; a model with no listed cached rate gets none.
Prices per million tokens: in, out, cached in.

Usage (repository root): python analysis/table18_repricing.py
"""
import csv
from collections import defaultdict

PRICES = {  # in, out, cached-in, USD per million tokens
    "glm-5.3-flash":      (0.10, 0.33, 0.02),
    "deepseek-4.1-flash": (0.15, 0.60, 0.003),
    "GLM-5.3":            (1.40, 4.40, 0.26),
    "Gemini 3.1 Pro":     (2.00, 12.00, 0.20),
    "Kimi K3":            (2.65, 13.28, 0.303),
    "Claude Opus 5":      (5.00, 25.00, 0.50),
    "GPT-5.5 Pro":        (30.00, 180.00, None),
}
reas = {(r['harness'], r['model'], int(r['seed'])): int(r['reasoning_tokens'])
        for r in csv.DictReader(open('data/study2/reasoning_tokens.csv'))}
tokens = defaultdict(lambda: [0, 0])                     # model -> [input, generated]
for r in csv.DictReader(open('data/study2/cells.csv')):
    if r['counted'] != 'True' or not r.get('tokens_in'):
        continue
    t = tokens[r['model']]
    t[0] += int(float(r['tokens_in']))
    t[1] += int(float(r['tokens_out'] or 0)) + reas.get((r['harness'], r['model'], int(r['seed'])), 0)
tin = sum(t[0] for t in tokens.values()); gen = sum(t[1] for t in tokens.values())


def cost(inp, out, p, cached):
    pin, pout, pc = p
    if cached and pc is None:
        return None
    read = 0.9 * inp if cached else 0
    return ((inp - read) * pin + read * (pc or 0) + out * pout) / 1e6


print(f"Study 2 ledger: input {tin} tokens, generated {gen} (output + reasoning)")
as_run = [sum(cost(*tokens[m], PRICES[m], c) for m in tokens) for c in (False, True)]
print(f"{'as run (each model at its own price)':38} {as_run[0]:>9.0f} {as_run[1]:>9.0f}")
for m in ("GLM-5.3", "Gemini 3.1 Pro", "Kimi K3", "Claude Opus 5", "GPT-5.5 Pro"):
    full, c90 = cost(tin, gen, PRICES[m], False), cost(tin, gen, PRICES[m], True)
    print(f"{m:38} {full:>9.0f} {('%9.0f' % c90) if c90 is not None else '      N/A'}")
