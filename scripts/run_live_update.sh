#!/usr/bin/env sh
set -eu

LIMIT="${LIMIT:-50}"
ENGINE="${ENGINE:-llm}"
LOCAL="${LOCAL:-0}"
FORCE="${FORCE:-0}"
SUMMARIZE="${SUMMARIZE:-0}"

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

set -- manage.py run_monitoring --no-bootstrap --limit "$LIMIT" --engine "$ENGINE"
if [ "$FORCE" = "1" ]; then set -- "$@" --force; fi
if [ "$SUMMARIZE" = "1" ] || [ "$ENGINE" = "llm" ]; then set -- "$@" --summarize; fi

if [ "$LOCAL" = "1" ]; then
  exec python "$@"
fi

exec docker compose exec -T web python "$@"
