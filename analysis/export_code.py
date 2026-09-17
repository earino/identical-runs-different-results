#!/usr/bin/env python3
"""Export each cell's delivered code history into results/evals/ so the agents' feature engineering can be read
from the GitHub repo without the 21 GB pulls/ trees.

For every cell <tree>/runs/<dataset>/<harness>/<model>/seed1 with an eval.json, writes under
results/evals/<tree>/<dataset>/<harness>/<model>/seed1/:
  code/NN_<hash>.py   train.py at every commit that touched it, oldest first (NN = order; 00 = the baseline)
  code/commits.tsv    NN, hash, unix time, subject  -> join to experiments.tsv on the hash
  FINAL.md            the agent's own final notes, when it wrote one
Never nest under a directory named runs/ (the repo .gitignore drops it). Usage:
  .venv/bin/python scripts/export_code.py pulls/bench3_ollama_final pulls/bench4_lunaroute ...
  EXPORT_OUT=results/phase2/evals .venv/bin/python scripts/export_code.py pulls/phase2c_luna_glm53   (phase 2: all seeds)
Also copies eval.json, experiments.tsv, cpu_ledger.tsv and fits.tsv when present (phase-2 ledgers).
"""
import os, subprocess, sys
from pathlib import Path

def git(wd, *args):
    return subprocess.run(["git", "-C", str(wd), *args], capture_output=True, text=True, check=True).stdout

EVIDENCE = ("eval.json", "logs/experiments.tsv", "logs/cpu_ledger.tsv", "logs/fits.tsv", "logs/cpu_kill.txt")

def export_cell(cell, out):
    out.mkdir(parents=True, exist_ok=True)
    for rel in EVIDENCE:
        src = cell / rel
        if src.exists(): (out / Path(rel).name).write_bytes(src.read_bytes())
    wd = cell / "workdir"
    if not (wd / ".git").is_dir(): return 0
    log = git(wd, "log", "--reverse", "--format=%H%x09%ct%x09%s", "--", "train.py").splitlines()
    code = out / "code"; code.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, line in enumerate(log):
        h, ts, subj = line.split("\t", 2)
        (code / f"{i:02d}_{h[:7]}.py").write_text(git(wd, "show", f"{h}:train.py"))
        rows.append(f"{i:02d}\t{h[:7]}\t{ts}\t{subj}")
    (code / "commits.tsv").write_text("n\tcommit\tunix_time\tsubject\n" + "\n".join(rows) + "\n")
    if (wd / "FINAL.md").exists(): (out / "FINAL.md").write_text((wd / "FINAL.md").read_text(errors="replace"))
    return len(rows)

def main(trees):
    total_cells = total_versions = 0
    for t in trees:
        t = Path(t); tree = t.name
        for cell in sorted((t / "runs").glob("*/*/*/seed*")):
            if not (cell / "eval.json").exists(): continue
            out = Path(os.environ.get("EXPORT_OUT", "results/evals")) / tree / cell.relative_to(t / "runs")
            n = export_cell(cell, out)
            if n: total_cells += 1; total_versions += n
    print(f"exported {total_versions} train.py versions from {total_cells} cells")

if __name__ == "__main__":
    main(sys.argv[1:])
