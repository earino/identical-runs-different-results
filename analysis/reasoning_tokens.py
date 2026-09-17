#!/usr/bin/env python3
"""Reasoning-token volume per run, recovered from the raw harness logs.

Why this exists: nothing in either arm's config sets a reasoning level. No `reasoning_effort`, no `thinking` object,
no budget — pi even declares `supportsReasoningEffort: false`. Every run therefore uses whatever the endpoint serves
by default, and `bench/usage.py` folds reasoning into `tokens_out`, so the aggregate cannot say how much thinking
happened. That matters for the GLM-5.3 vs GLM-5.3 Flash comparison: if the endpoint's default thinking budget differs
between the two, a quality difference is not purely model scale. This script measures the thinking instead of
assuming it.

Each harness accounts for reasoning differently, so each needs its own reader:
  pi       harness.stdout JSONL, `message_end` events only. Its `message_update` events repeat a CUMULATIVE usage on
           every delta (18,103 usage objects against 244 messages in one sampled cell), so summing every occurrence
           overstates the total by ~74x.
  opencode harness.stdout JSONL, `step_finish` events, `part.tokens.reasoning`. One per step, no repeats; reasoning
           is counted separately from `output`.
  hermes   logs/usage.json, `reasoning_tokens` — already a session total from its sqlite session row.

Counts are NOT comparable across harnesses (different call counts, different definitions of a step). Compare within a
harness across models: that is the flash-vs-non-flash question this was written for.

Usage: python experiments/run_variance/reasoning_tokens.py [PULLS ...] [-o OUT.csv]
       e.g. python experiments/run_variance/reasoning_tokens.py pulls/variance pulls/variance_glm53 \
            -o results/variance_glm53/reasoning_tokens.csv
"""
import csv, glob, json, os, statistics as st, sys
from collections import defaultdict

FIELDS = ["box", "harness", "model", "seed", "api_calls", "reasoning_tokens", "output_tokens", "input_tokens"]


def read_pi(logs):
    """message_end events only: message_update carries a cumulative usage and would multiply-count."""
    calls = r = o = i = 0
    for line in _jsonl(os.path.join(logs, "harness.stdout")):
        if line.get("type") != "message_end":
            continue
        u = (line.get("message") or {}).get("usage") or {}
        calls += 1
        r += u.get("reasoning") or 0
        o += u.get("output") or 0
        i += (u.get("input") or 0) + (u.get("cacheRead") or 0)
    return calls, r, o, i


def read_opencode(logs):
    calls = r = o = i = 0
    for line in _jsonl(os.path.join(logs, "harness.stdout")):
        if "step_finish" not in str(line.get("type", "")):
            continue
        t = (line.get("part") or {}).get("tokens") or {}
        cache = t.get("cache") or {}
        calls += 1
        r += t.get("reasoning") or 0
        o += t.get("output") or 0
        i += (t.get("input") or 0) + (cache.get("read") or 0)
    return calls, r, o, i


def read_hermes(logs):
    """Already a session total; there is no per-call breakdown in the sqlite row."""
    try:
        u = json.load(open(os.path.join(logs, "usage.json")))
    except Exception:
        return 0, 0, 0, 0
    if not isinstance(u, dict) or not u:
        return 0, 0, 0, 0
    return (u.get("api_calls") or 0, u.get("reasoning_tokens") or 0,
            u.get("output_tokens") or 0, (u.get("input_tokens") or 0) + (u.get("cache_read_tokens") or 0))


def _jsonl(path):
    try:
        fh = open(path, errors="replace")
    except OSError:
        return
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue          # a truncated final line on a killed cell is normal
            if isinstance(d, dict):
                yield d


READERS = {"pi": read_pi, "opencode": read_opencode, "hermes": read_hermes}


def main(argv):
    out_path = None
    if "-o" in argv:
        k = argv.index("-o"); out_path = argv[k + 1]; argv = argv[:k] + argv[k + 2:]
    pulls = argv or ["pulls/variance"]

    rows = []
    for root in pulls:
        for logs in sorted(glob.glob(os.path.join(root, "*", "runs", "*", "*", "*", "seed*", "logs"))):
            parts = logs.split(os.sep)
            box, harness, model, seed_s = parts[-6], parts[-4], parts[-3], parts[-2]
            reader = READERS.get(harness)
            if not reader:
                continue
            calls, r, o, i = reader(logs)
            if calls == 0 and r == 0:
                continue                       # unscored or empty cell: nothing to measure
            rows.append({"box": box, "harness": harness, "model": model,
                         "seed": int(seed_s.replace("seed", "").split("_")[0]),
                         "api_calls": calls, "reasoning_tokens": r, "output_tokens": o, "input_tokens": i})

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)

    by = defaultdict(list)
    for x in rows:
        by[(x["harness"], x["model"])].append(x)
    print(f"{'harness':10}{'model':22}{'n':>4}{'median reasoning':>18}{'per call':>10}{'/output':>9}{'calls':>7}")
    for k in sorted(by, key=lambda k: (k[0], k[1])):
        v = by[k]
        med = st.median([x["reasoning_tokens"] for x in v])
        calls = st.median([x["api_calls"] for x in v])
        per = st.median([x["reasoning_tokens"] / x["api_calls"] for x in v if x["api_calls"]] or [0])
        share = st.median([x["reasoning_tokens"] / x["output_tokens"] for x in v if x["output_tokens"]] or [0])
        print(f"{k[0]:10}{k[1]:22}{len(v):>4}{med:>18,.0f}{per:>10,.0f}{share:>9.2f}{calls:>7,.0f}")
    if out_path:
        print(f"\nwrote {out_path} ({len(rows)} runs)")
    print("\nCompare WITHIN a harness across models; the three harnesses count calls and reasoning differently.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
