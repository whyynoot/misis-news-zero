#!/usr/bin/env sh
set -eu

TAIL="${TAIL:-20}"
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Historical backfill log:"
if [ -f "tmp-ui-checks/history-backfill.out.log" ]; then
  tail -n "$TAIL" "tmp-ui-checks/history-backfill.out.log"
else
  echo "No history-backfill.out.log yet."
fi

echo
echo "Recent warnings/errors:"
if [ -f "tmp-ui-checks/history-backfill.err.log" ]; then
  tail -n "$TAIL" "tmp-ui-checks/history-backfill.err.log"
else
  echo "No history-backfill.err.log yet."
fi

echo
echo "Database counters:"
docker compose exec -T web python manage.py shell -c "from analyzer.models import NewsItem, NewsClassification, DailySentimentSummary; print('real_news', NewsItem.objects.filter(is_seed=False).count()); print('llm_classifications', NewsClassification.objects.filter(engine='llm').count()); print('llm_days', DailySentimentSummary.objects.filter(engine='llm').count())"
