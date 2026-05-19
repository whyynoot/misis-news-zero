from pathlib import Path

from django.core.management.base import BaseCommand

from analyzer.constants import ENGINE_LLM
from analyzer.llm_services import llm_summary_identity
from analyzer.monitoring_service import (
    DEFAULT_BACKFILL_SOURCE,
    DEFAULT_HISTORY_BACKFILL_LIMIT,
    backfill_existing_news,
    get_engine_identity,
    load_seed_dataset,
)
from analyzer.models import DailyFactorSentiment, DailySentimentSummary, MonitoringSummary, NewsClassification


def clear_engine_results(engine: str, source: str | None, include_seed: bool) -> int:
    identity = get_engine_identity(engine)
    queryset = NewsClassification.objects.filter(**identity)
    if source:
        queryset = queryset.filter(news_item__source=source)
    if not include_seed:
        queryset = queryset.filter(news_item__is_seed=False)
    deleted_count, _ = queryset.delete()

    DailyFactorSentiment.objects.filter(**identity).delete()
    DailySentimentSummary.objects.filter(**identity).delete()
    if engine == ENGINE_LLM:
        MonitoringSummary.objects.filter(**llm_summary_identity()).delete()
    return deleted_count


class Command(BaseCommand):
    help = "Backfill monitoring over stored real news. Demo seed loading is explicit via --seed."

    def add_arguments(self, parser):
        parser.add_argument(
            "--seed",
            action="store_true",
            help="Load the demo seed dataset instead of classifying stored real news.",
        )
        parser.add_argument(
            "--seed-file",
            type=str,
            default=None,
            help="Path to a JSON seed/archive file used only with --seed.",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Reclassify stored news even when cache is valid.",
        )
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Clear selected engine results before backfill. With --seed, clears seed target data.",
        )
        parser.add_argument(
            "--if-empty",
            action="store_true",
            help="With --seed, do nothing if seed data already exists.",
        )
        parser.add_argument(
            "--engine",
            choices=["bert", "llm"],
            default="bert",
            help="Backfill analysis engine: bert or llm.",
        )
        parser.add_argument(
            "--source",
            type=str,
            default=DEFAULT_BACKFILL_SOURCE,
            help="Stored news source to backfill. Empty default includes all sources.",
        )
        parser.add_argument(
            "--include-seed",
            action="store_true",
            help="Include seed/demo rows in real-news backfill. Disabled by default.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help=f"Limit stored news items. Defaults to MONITORING_HISTORY_BACKFILL_LIMIT ({DEFAULT_HISTORY_BACKFILL_LIMIT}).",
        )
        parser.add_argument(
            "--summarize",
            action="store_true",
            help="Generate summary after LLM backfill.",
        )
        parser.add_argument(
            "--period",
            choices=["day", "week", "month"],
            default="day",
            help="Summary period for --summarize.",
        )

    def handle(self, *args, **options):
        if options.get("seed"):
            result = load_seed_dataset(
                seed_path=Path(options["seed_file"]) if options.get("seed_file") else None,
                force=options.get("force", False),
                clear_existing=options.get("clear", False),
            )
            if result.get("skipped"):
                self.stdout.write(self.style.WARNING(result["message"]))
                return
            self.stdout.write(
                self.style.SUCCESS(
                    f"{result['mode']} seed load completed. News: {result['news_created']}, "
                    f"classifications: {result['classifications_created']}, dates: {', '.join(result['dates'])}"
                )
            )
            return

        source = (options.get("source") or "").strip() or None
        engine = options.get("engine")
        include_seed = bool(options.get("include_seed", False))
        if options.get("clear"):
            deleted_count = clear_engine_results(engine, source, include_seed)
            self.stdout.write(
                self.style.WARNING(
                    f"Cleared {deleted_count} classification rows for engine={engine}, source={source or 'all'}."
                )
            )

        result = backfill_existing_news(
            engine=engine,
            force=options.get("force", False),
            limit=options.get("limit"),
            summarize=options.get("summarize", False),
            period=options.get("period") or "day",
            source=source,
            include_seed=include_seed,
        )
        if result.get("skipped"):
            self.stdout.write(self.style.WARNING(result.get("message", "Backfill skipped.")))
            return

        self.stdout.write(
            self.style.SUCCESS(
                f"Backfill completed. Engine: {result['engine']}, source: {result.get('source') or 'all'}, "
                f"news: {result['news_count']}/{result['limit']}, cached: {result.get('cached_count', 0)}, "
                f"classifications: {result['classifications_created']}, dates: {len(result['dates'])}"
            )
        )
        if result.get("errors"):
            self.stdout.write(self.style.WARNING("Errors: " + "; ".join(result["errors"][:3])))
        if result.get("warnings"):
            self.stdout.write(self.style.WARNING("Warnings: " + "; ".join(result["warnings"][:3])))
