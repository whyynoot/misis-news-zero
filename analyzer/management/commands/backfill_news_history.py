import datetime

from django.core.management import call_command
from django.core.management.base import BaseCommand

from analyzer.monitoring_service import (
    DEFAULT_HISTORY_DAYS,
    DEFAULT_HISTORY_LIMIT_PER_DAY,
    DEFAULT_HISTORY_SOURCES,
    backfill_news_history,
    parse_date,
)


class Command(BaseCommand):
    help = "Fetch historical news from archive-capable sources, store them, classify them, and rebuild dashboard aggregates."

    def add_arguments(self, parser):
        parser.add_argument("--from-date", dest="start_date", type=str, default=None, help="Start date, YYYY-MM-DD.")
        parser.add_argument("--to-date", dest="end_date", type=str, default=None, help="End date, YYYY-MM-DD.")
        parser.add_argument(
            "--days",
            type=int,
            default=DEFAULT_HISTORY_DAYS,
            help=f"Rolling history window when --from-date is omitted. Default: {DEFAULT_HISTORY_DAYS}.",
        )
        parser.add_argument(
            "--sources",
            type=str,
            default=DEFAULT_HISTORY_SOURCES,
            help=f"Comma-separated sources. Default: {DEFAULT_HISTORY_SOURCES}.",
        )
        parser.add_argument(
            "--limit-per-day",
            type=int,
            default=DEFAULT_HISTORY_LIMIT_PER_DAY,
            help=f"Maximum relevant articles per day per source. Default: {DEFAULT_HISTORY_LIMIT_PER_DAY}.",
        )
        parser.add_argument("--limit", type=int, default=None, help="Optional global cap after collection.")
        parser.add_argument(
            "--chunk-days",
            type=int,
            default=7,
            help="Process the range in chunks and print progress after each chunk. Use 0 to process all at once.",
        )
        parser.add_argument(
            "--delay-seconds",
            type=float,
            default=0.25,
            help="Pause between archive dates to avoid hammering source sites.",
        )
        parser.add_argument("--engine", choices=["bert", "llm"], default="llm", help="Analysis engine.")
        parser.add_argument("--force", action="store_true", help="Reclassify already classified stored news.")
        parser.add_argument("--summarize", action="store_true", help="Generate LLM summaries after LLM classification.")
        parser.add_argument("--period", choices=["day", "week", "month"], default="day", help="Summary period.")
        parser.add_argument(
            "--clean-seed",
            action="store_true",
            help="Delete demo seed rows and cached summaries before real historical backfill.",
        )

    def handle(self, *args, **options):
        if options.get("clean_seed"):
            call_command("clean_monitoring_data", "--drop-seed", "--delete-summaries", stdout=self.stdout)

        result = self._run_backfill(options)

        if result.get("skipped"):
            self.stdout.write(self.style.WARNING(result.get("message", "Historical backfill skipped.")))
            return

        self.stdout.write(
            self.style.SUCCESS(
                f"Historical backfill completed. Range: {result['start_date']}..{result['end_date']}, "
                f"sources: {', '.join(result.get('sources') or [])}, engine: {result['engine']}, "
                f"fetched: {result['news_fetched']}, stored: {result['news_stored']}, "
                f"cached: {result.get('cached_count', 0)}, classifications: {result['classifications_created']}, "
                f"dates: {len(result['dates'])}."
            )
        )
        if result.get("errors"):
            self.stdout.write(self.style.WARNING("Errors: " + "; ".join(result["errors"][:3])))
        if result.get("warnings"):
            self.stdout.write(self.style.WARNING("Warnings: " + "; ".join(result["warnings"][:3])))

    def _resolve_range(self, options):
        end_date = parse_date(options.get("end_date")) if options.get("end_date") else datetime.date.today()
        if options.get("start_date"):
            start_date = parse_date(options.get("start_date"))
        else:
            start_date = end_date - datetime.timedelta(days=max(int(options.get("days") or DEFAULT_HISTORY_DAYS), 1) - 1)
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        return start_date, end_date

    def _run_backfill(self, options):
        chunk_days = max(int(options.get("chunk_days") or 0), 0)
        if chunk_days <= 0:
            return backfill_news_history(
                start_date=options.get("start_date"),
                end_date=options.get("end_date"),
                days=options.get("days"),
                source_names=options.get("sources"),
                limit_per_day=options.get("limit_per_day"),
                total_limit=options.get("limit"),
                delay_seconds=options.get("delay_seconds"),
                engine=options.get("engine"),
                force=options.get("force", False),
                summarize=options.get("summarize", False),
                period=options.get("period") or "day",
            )

        start_date, end_date = self._resolve_range(options)
        remaining_limit = options.get("limit")
        totals = {
            "skipped": True,
            "status": "skipped",
            "mode": "history_backfill",
            "engine": options.get("engine"),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "sources": [],
            "news_fetched": 0,
            "news_stored": 0,
            "cached_count": 0,
            "classifications_created": 0,
            "dates": [],
            "errors": [],
            "warnings": [],
        }

        current = start_date
        while current <= end_date:
            chunk_end = min(current + datetime.timedelta(days=chunk_days - 1), end_date)
            chunk_limit = remaining_limit if remaining_limit else None
            chunk_result = backfill_news_history(
                start_date=current,
                end_date=chunk_end,
                source_names=options.get("sources"),
                limit_per_day=options.get("limit_per_day"),
                total_limit=chunk_limit,
                delay_seconds=options.get("delay_seconds"),
                engine=options.get("engine"),
                force=options.get("force", False),
                summarize=False,
                period=options.get("period") or "day",
            )

            fetched = int(chunk_result.get("news_fetched", 0))
            totals["skipped"] = totals["skipped"] and bool(chunk_result.get("skipped"))
            totals["status"] = chunk_result.get("status") or totals["status"]
            totals["news_fetched"] += fetched
            totals["news_stored"] += int(chunk_result.get("news_stored", 0))
            totals["cached_count"] += int(chunk_result.get("cached_count", 0))
            totals["classifications_created"] += int(chunk_result.get("classifications_created", 0))
            totals["dates"].extend(chunk_result.get("dates", []))
            totals["errors"].extend(chunk_result.get("errors", []))
            totals["warnings"].extend(chunk_result.get("warnings", []))
            totals["sources"] = sorted(set(totals["sources"]) | set(chunk_result.get("sources") or []))

            self.stdout.write(
                self.style.SUCCESS(
                    f"Chunk {current.isoformat()}..{chunk_end.isoformat()}: "
                    f"status={chunk_result.get('status', 'skipped')}, fetched={fetched}, "
                    f"stored={chunk_result.get('news_stored', 0)}, "
                    f"classifications={chunk_result.get('classifications_created', 0)}."
                )
            )

            if remaining_limit:
                remaining_limit = max(int(remaining_limit) - fetched, 0)
                if remaining_limit <= 0:
                    break
            current = chunk_end + datetime.timedelta(days=1)

        totals["dates"] = sorted(set(totals["dates"]))
        if not totals["skipped"]:
            totals["status"] = "success" if not totals["errors"] else "partial_failed"
        if options.get("summarize"):
            self.stdout.write(self.style.WARNING("Summary generation is skipped for chunked history. Run summarize_monitoring after backfill if needed."))
        return totals
