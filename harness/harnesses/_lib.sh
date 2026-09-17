# Shared helpers for harness runner scripts. Sourced, not executed.
# The runner (bench/runner.py) sets these env vars before invoking harnesses/<name>.sh with cwd = workdir:
#   BENCH_WORKDIR BENCH_RUN_DIR BENCH_LOG_DIR BENCH_STATE_DIR BENCH_PROMPT_FILE
#   BENCH_MODEL BENCH_MODEL_ENDPOINT (cloud|local|<provider block name, e.g. lunaroute>) BENCH_LLM_BASE_URL BENCH_LLM_API_KEY
#   BENCH_WALL_SECONDS BENCH_DEADLINE_EPOCH BENCH_THREADS EXPERIMENT_TIMEOUT MAX_EXPERIMENTS
# Each script must: run the harness headlessly on the prompt, auto-approving tools, and exit when the agent finishes.
set -u
: "${BENCH_WORKDIR:?}" "${BENCH_LOG_DIR:?}" "${BENCH_STATE_DIR:?}" "${BENCH_PROMPT_FILE:?}" "${BENCH_MODEL:?}"
: "${BENCH_LLM_BASE_URL:?}" "${BENCH_LLM_API_KEY:=ollama}" "${BENCH_WALL_SECONDS:=3600}"
cd "$BENCH_WORKDIR"
PROMPT="$(cat "$BENCH_PROMPT_FILE")"
mkdir -p "$BENCH_LOG_DIR" "$BENCH_STATE_DIR"
# OpenAI-compatible base for harnesses that speak /v1
OPENAI_COMPAT_URL="${BENCH_LLM_BASE_URL%/}/v1"
# Wall-clock guard: the runner also kills the process group, but harnesses with a native cap get it too.
WALL="$BENCH_WALL_SECONDS"
# Temp files stay inside the run dir too.
export TMPDIR="$BENCH_STATE_DIR/tmp"; mkdir -p "$TMPDIR"
# Keep the agent from seeing the operator's own keys for other providers.
unset ANTHROPIC_API_KEY OPENAI_API_KEY GEMINI_API_KEY GOOGLE_API_KEY OPENROUTER_API_KEY 2>/dev/null || true
log() { echo "[harness $(date -u +%H:%M:%S)] $*" >&2; }
