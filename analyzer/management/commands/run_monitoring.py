from django.core.management.base import BaseCommand

from analyzer.monitoring_service import run_monitoring_pipeline


class Command(BaseCommand):
    help = "Fetch current configured news sources and update monitoring classifications."

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Reclassify already stored live news.",
        )
        parser.add_argument(
            "--bootstrap",
            action="store_true",
            help="Explicitly load demo seed data if the database is empty.",
        )
        parser.add_argument(
            "--no-bootstrap",
            action="store_true",
            help="Do not load demo seed data if the database is empty. This is the default.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Limit current live feed news items. Defaults to MONITORING_LIVE_NEWS_LIMIT.",
        )
        parser.add_argument(
            "--engine",
            choices=["bert", "llm"],
            default="llm",
            help="Analysis engine: bert or llm.",
        )
        parser.add_argument(
            "--summarize",
            action="store_true",
            help="Generate compact LLM summaries after LLM classification.",
        )
        parser.add_argument(
            "--period",
            choices=["day", "week", "month"],
            default="day",
            help="Summary period for --summarize.",
        )

    def handle(self, *args, **options):
        result = run_monitoring_pipeline(
            force=options.get("force", False),
            bootstrap_if_empty=bool(options.get("bootstrap", False)) and not options.get("no_bootstrap", False),
            limit=options.get("limit"),
            engine=options.get("engine"),
            summarize=options.get("summarize", False),
            period=options.get("period") or "day",
        )

        if result.get("skipped"):
            self.stdout.write(self.style.WARNING(result.get("message", "Monitoring update skipped.")))
            return

        overview = result.get("overview", {})
        self.stdout.write(
            self.style.SUCCESS(
                f"Live update completed. News fetched: {result['news_fetched']}, "
                f"stored: {result['news_stored']}, classifications: {result['classifications_created']}, "
                f"engine: {result.get('engine')}, avg sentiment: {overview.get('average_sentiment', 0):.3f}, "
                f"risk index: {overview.get('risk_index', 0):.3f}"
            )
        )
        if result.get("errors"):
            self.stdout.write(self.style.WARNING("Errors: " + "; ".join(result["errors"][:3])))
        if result.get("warnings"):
            self.stdout.write(self.style.WARNING("Warnings: " + "; ".join(result["warnings"][:3])))
