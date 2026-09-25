#!/usr/bin/env python3
"""Export Study 1's Claude Code usage one request at a time (token counts only, no conversation text).

claude_request_log.py sums each run; this keeps every request, in transcript order, so the paper's per-request cache
observations (Section 4.9: how large the prompt was when the cache first failed or held) can be checked from a
committed file. Same deduplication by message id, keeping each message's final (largest) usage reading.

Usage: .venv/bin/python scripts/phase2/claude_requests_export.py   -> results/phase2/claude_requests.csv
"""
import csv, json
from pathlib import Path

ROOTS = ["pulls/phase2b_ollama_stage1/runs", "pulls/phase2a_luna_stage1/runs", "pulls/phase2c_luna_glm53/runs"]
rows = []
for root in ROOTS:
    for cell in sorted(Path(root).glob("airline/claude/*/seed*")):
        ev = json.loads((cell / "eval.json").read_text()) if (cell / "eval.json").exists() else {}
        provider = "lunaroute" if "lunaroute" in (ev.get("llm_base_url") or "") else "ollama"
        order, best = [], {}
        files = sorted(cell.glob("harness_state/claude/projects/*/*.jsonl")) + sorted(cell.glob("harness_state/claude/projects/*/*/subagents/*.jsonl"))
        for p in files:
            for line in p.open(errors="replace"):
                try: d = json.loads(line)
                except Exception: continue
                m = d.get("message") or {}; u = m.get("usage")
                if not isinstance(u, dict): continue
                mid = m.get("id") or d.get("uuid")
                r = (u.get("input_tokens") or 0, u.get("cache_read_input_tokens") or 0,
                     u.get("cache_creation_input_tokens") or 0, u.get("output_tokens") or 0)
                if mid not in best: order.append(mid); best[mid] = r
                elif sum(r) > sum(best[mid]): best[mid] = r
        for i, mid in enumerate(order, 1):
            fresh, read, write, out = best[mid]
            rows.append({"provider": provider, "model": ev.get("model", ""), "seed": ev.get("seed"), "request": i,
                         "prompt_tokens": fresh + read + write, "cache_read_tokens": read, "output_tokens": out})
Path("results/phase2").mkdir(parents=True, exist_ok=True)
with open("results/phase2/claude_requests.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"wrote results/phase2/claude_requests.csv ({len(rows)} requests)")
