#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ "${BUILD:-0}" = "1" ]; then
  docker compose up -d --build db web scheduler
else
  docker compose up -d db web scheduler
fi

docker compose ps
