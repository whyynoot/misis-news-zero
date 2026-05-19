import time

from decouple import config
from django.core.management.base import BaseCommand

from analyzer.monitoring_service import (
    DEFAULT_HISTORY_DAYS,
    DEFAULT_HISTORY_LIMIT_PER_DAY,
    DEFAULT_LIVE_NEWS_LIMIT,
    backfill_news_history,
    ensure_factor_catalog,
    run_monitoring_pipeline,
)


class Command(BaseCommand):
    help = "Run monitoring updates on a fixed interval. Intended for the docker-compose scheduler service."

    def add_arguments(self, parser):
        parser.add_argument(
            "--interval-hours",
            type=float,
            default=config("MONITORING_UPDATE_INTERVAL_HOURS", default=4, cast=float),
            help="Delay between live updates.",
        )
        parser.add_argument("--engine", choices=["bert", "llm"], default="llm", help="Analysis engine.")
        parser.add_argument("--limit", type=int, default=DEFAULT_LIVE_NEWS_LIMIT, help="Live news limit per run.")
        parser.add_argument("--summarize", action="store_true", help="Generate LLM summaries for LLM runs.")
        parser.add_argument("--period", choices=["day", "week", "month"], default="day", help="Summary period.")
        parser.add_argument("--once", action="store_true", help="Run one update and exit.")
        parser.add_argument(
            "--history-on-start",
            action="store_true",
            help="Run historical archive backfill once before the interval loop.",
        )
        parser.add_argument("--history-days", type=int, default=DEFAULT_HISTORY_DAYS)
        parser.add_argument("--history-sources", type=str, default=None)
        parser.add_argument("--history-limit-per-day", type=int, default=DEFAULT_HISTORY_LIMIT_PER_DAY)

    def _run_live_cycle(self, options):
        return run_monitoring_pipeline(
            engine=options.get("engine"),
            limit=options.get("limit"),
            summarize=options.get("summarize", False),
            period=options.get("period") or "day",
            bootstrap_if_empty=False,
        )

    def handle(self, *args, **options):
        ensure_factor_catalog()

        if options.get("history_on_start"):
            history_result = backfill_news_history(
                days=options.get("history_days"),
                source_names=options.get("history_sources"),
                limit_per_day=options.get("history_limit_per_day"),
                engine=options.get("engine"),
                summarize=options.get("summarize", False),
                period=options.get("period") or "day",
            )
            self.stdout.write(
                self.style.SUCCESS(
                    f"Startup history backfill: fetched={history_result.get('news_fetched', 0)}, "
                    f"stored={history_result.get('news_stored', 0)}, status={history_result.get('status', 'skipped')}."
                )
            )

        interval_seconds = max(float(options.get("interval_hours") or 4), 0.01) * 3600
        while True:
            result = self._run_live_cycle(options)
            self.stdout.write(
                self.style.SUCCESS(
                    f"Scheduled live update: status={result.get('status', 'skipped')}, "
                    f"fetched={result.get('news_fetched', 0)}, stored={result.get('news_stored', 0)}, "
                    f"classifications={result.get('classifications_created', 0)}."
                )
            )
            if options.get("once"):
                break
            time.sleep(interval_seconds)
