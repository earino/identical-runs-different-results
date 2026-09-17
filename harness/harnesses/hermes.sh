#!/usr/bin/env bash
# Hermes Agent (Nous Research) -> Ollama via its first-class `ollama-cloud` provider (OpenAI-compatible /v1)
source "$(dirname "$0")/_lib.sh"
export HERMES_HOME="$BENCH_STATE_DIR/hermes"                 # fresh: no sessions, no memories, no SOUL.md from earlier runs
mkdir -p "$HERMES_HOME"
export OLLAMA_API_KEY="$BENCH_LLM_API_KEY"
export OLLAMA_BASE_URL="$OPENAI_COMPAT_URL"                 # ollama-cloud provider honors this (default https://ollama.com/v1)
export HERMES_YOLO_MODE=1
log "hermes chat --provider ollama-cloud -m $BENCH_MODEL via $OLLAMA_BASE_URL"
# --ignore-user-config: built-in defaults instead of the operator's ~/.hermes/config.yaml (AGENTS.md injection stays on).
hermes chat --oneshot -Q --yolo --ignore-user-config \
  --provider ollama-cloud -m "$BENCH_MODEL" \
  --run-budget "$WALL" --max-turns 2000 \
  --query-file "$BENCH_PROMPT_FILE"
RC=$?
# Hermes prints only the final answer + "session_id: ..." in quiet mode; token counts live in its session db.
SID=$(grep -oE 'session_id: [A-Za-z0-9_]+' "$BENCH_LOG_DIR/harness.stdout" 2>/dev/null | tail -1 | awk '{print $2}')
DB="${HERMES_HOME:-$HOME/.hermes}/state.db"
WHERE="order by started_at desc limit 1"; [ -n "$SID" ] && WHERE="where id='$SID'"
if [ -f "$DB" ] && command -v sqlite3 >/dev/null; then
  sqlite3 -json "$DB" "select id as session_id, model, message_count, tool_call_count, api_call_count as api_calls, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens, end_reason from sessions $WHERE;" \
    | jq '.[0] // {}' > "$BENCH_LOG_DIR/usage.json" 2>/dev/null || true
fi
exit $RC
