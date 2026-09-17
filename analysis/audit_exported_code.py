#!/usr/bin/env python3
"""Re-run the rule-breaking screen over the delivered code in this repository.

The paper claims a specific number of runs computed features from the batch they were scoring (contract rule 2).
That claim should not rest on our word, so this reproduces it from the code the agents actually delivered, which
ships in code/study2/ and code/study3/.

For every exported cell it takes the LAST train.py version (the one that was scored) and runs the same AST taint
analysis analyze.py used: a value derived from predict_proba's argument -- the frame being scored -- must not reach
value_counts, nunique, rank, groupby or transform. Fitting an encoder on train and applying it is fine; deriving a
feature from the scored batch is not, because it leaks information across the rows being predicted.

Usage:
    python analysis/audit_exported_code.py                 # both studies
    python analysis/audit_exported_code.py code/study3     # one study

Expected output, which is what the paper reports:

    code/study2: 7 flagged      <- the RAW screen. Two are false positives we read by hand and excluded: one fits
                                   its encoder out-of-fold at fit time, one uses groupby only to index rows. The
                                   paper therefore counts 5. See data/study2/frame_stats.csv for the per-run record.
    code/study3: 5 flagged      <- all 5 counted.
    12 flagged of 468 delivered programs audited

The raw, unfiltered screen is printed on purpose, with the offending line of source for every hit, so that you can
judge our two exclusions rather than take them on trust. A screen that hides its false positives is not a screen.
"""
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_frame_stats import audit  # noqa: E402

roots = sys.argv[1:] or ["code/study2", "code/study3"]
total = flagged = 0
for root in roots:
    if not os.path.isdir(root):
        print(f"{root}: not found (run from the repository root)")
        continue
    hits_here = []
    for commits in sorted(Path(root).rglob("commits.tsv")):
        cell = commits.parent.parent               # .../<harness>/<model>/<seed>
        versions = sorted(cell.glob("code/*.py"))
        if not versions:
            continue
        total += 1
        last = versions[-1]                        # the version that was scored
        hits, _note = audit(last.read_text(errors="replace"))
        if hits:
            flagged += 1
            hits_here.append(("/".join(cell.parts[-3:]), last.name, hits))
    print(f"\n=== {root}: {len(hits_here)} flagged")
    for cell, version, hits in sorted(hits_here):
        print(f"  {cell}  ({version})")
        for lineno, method, src, fname in hits:      # the evidence, so you can judge each hit yourself
            print(f"      line {lineno:>4}  {method:<14} in {fname or '<module>'}:  {src}")
print(f"\n{flagged} flagged of {total} delivered programs audited")
