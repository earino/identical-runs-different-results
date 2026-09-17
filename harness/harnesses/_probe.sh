#!/usr/bin/env bash
# Not a harness: prints what a cell can see. `bench probe` launches it exactly like a real cell.
source "$(dirname "$0")/_lib.sh"
echo "user: $(id)   hostname: $(hostname)   HOME=$HOME   cwd=$(pwd)"
echo "cpus visible (nproc): $(nproc)   cpuset: $(cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || echo n/a)   cpu quota: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)   mem limit: $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo n/a)"
echo "xgboost default threads (n_jobs=-1 would use): $(python -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null)   OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}   BENCH_THREADS=${BENCH_THREADS:-unset}"
echo "python: $(command -v python) -> $(python -c 'import xgboost,sys; print(sys.version.split()[0], "xgboost", xgboost.__version__)' 2>&1)"
echo "harnesses on PATH: $(for h in claude pi opencode hermes codex openclaw; do command -v $h >/dev/null && printf '%s ' $h; done)"
echo "--- run dir ($BENCH_RUN_DIR) ---"; ls "$BENCH_RUN_DIR"
echo "--- workdir ---"; ls -a "$BENCH_WORKDIR"; echo "git log: $(git -C "$BENCH_WORKDIR" log --oneline | tr '\n' ';')"
echo "--- can I see the data tree / other runs / holdout? ---"
for p in "$BENCH_ROOT/data" "$BENCH_ROOT/runs" "$BENCH_ROOT/runs_smoke" "$BENCH_ROOT/results" /app/data /app/runs "$HOME/.claude" "$HOME/.hermes/state.db" "$HOME/.local/share/opencode" "$HOME/.codex/sessions" "$HOME/.openclaw"; do
  if [ -e "$p" ]; then echo "  VISIBLE: $p ($(ls "$p" 2>/dev/null | wc -l | tr -d ' ') entries)"; else echo "  absent:  $p"; fi
done
h=$(find / -name holdout.csv -not -path '*/proc/*' 2>/dev/null | head -3 | tr '\n' ' '); echo "  holdout.csv anywhere: ${h:-none}"
o=$(find / -name experiments.tsv -not -path "$BENCH_RUN_DIR/*" -not -path '*/proc/*' 2>/dev/null | head -3 | tr '\n' ' '); echo "  other cells' experiments.tsv: ${o:-none}"
echo "--- environment (BENCH_*, keys masked) ---"; env | grep -E '^BENCH_' | sed -E 's/(API_KEY=).+/\1<masked>/' | sort
echo "--- network ---"; curl -s -o /dev/null -m 10 -w "ollama.com: HTTP %{http_code}\n" "$BENCH_LLM_BASE_URL/v1/models" -H "Authorization: Bearer $BENCH_LLM_API_KEY"; curl -s -o /dev/null -m 10 -w "example.com: HTTP %{http_code}\n" https://example.com
