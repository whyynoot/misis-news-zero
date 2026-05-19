#!/usr/bin/env sh
set -eu

DAYS="${DAYS:-365}"
FROM_DATE="${FROM_DATE:-}"
TO_DATE="${TO_DATE:-}"
SOURCES="${SOURCES:-interfax}"
LIMIT_PER_DAY="${LIMIT_PER_DAY:-8}"
CHUNK_DAYS="${CHUNK_DAYS:-1}"
LIMIT="${LIMIT:-}"
ENGINE="${ENGINE:-llm}"
DELAY_SECONDS="${DELAY_SECONDS:-0.25}"
LOCAL="${LOCAL:-0}"
CLEAN_SEED="${CLEAN_SEED:-0}"
FORCE="${FORCE:-0}"
SUMMARIZE="${SUMMARIZE:-0}"

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

set -- manage.py backfill_news_history \
  --sources "$SOURCES" \
  --limit-per-day "$LIMIT_PER_DAY" \
  --chunk-days "$CHUNK_DAYS" \
  --engine "$ENGINE" \
  --delay-seconds "$DELAY_SECONDS"

if [ -n "$FROM_DATE" ]; then
  set -- "$@" --from-date "$FROM_DATE"
else
  set -- "$@" --days "$DAYS"
fi

if [ -n "$TO_DATE" ]; then set -- "$@" --to-date "$TO_DATE"; fi
if [ -n "$LIMIT" ]; then set -- "$@" --limit "$LIMIT"; fi
if [ "$CLEAN_SEED" = "1" ]; then set -- "$@" --clean-seed; fi
if [ "$FORCE" = "1" ]; then set -- "$@" --force; fi
if [ "$SUMMARIZE" = "1" ]; then set -- "$@" --summarize; fi

if [ "$LOCAL" = "1" ]; then
  exec python "$@"
fi

exec docker compose exec -T web python "$@"
