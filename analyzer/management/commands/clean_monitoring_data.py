from django.core.management.base import BaseCommand

from analyzer.constants import ENGINE_BERT, ENGINE_LLM
from analyzer.monitoring_service import rebuild_sentiment_history
from analyzer.models import DailyFactorSentiment, DailySentimentSummary, MonitoringSummary, NewsItem


class Command(BaseCommand):
    help = "Clean demo/seed monitoring data and rebuild real-news aggregates."

    def add_arguments(self, parser):
        parser.add_argument(
            "--drop-seed",
            action="store_true",
            help="Delete seed/demo news rows and their classifications.",
        )
        parser.add_argument(
            "--delete-summaries",
            action="store_true",
            help="Delete cached LLM summaries before regeneration.",
        )
        parser.add_argument(
            "--rebuild",
            action="store_true",
            help="Rebuild BERT and LLM aggregate history from remaining classifications.",
        )

    def handle(self, *args, **options):
        deleted_news = 0
        if options.get("drop_seed"):
            deleted_news, _ = NewsItem.objects.filter(is_seed=True).delete()

        deleted_summaries = 0
        if options.get("delete_summaries"):
            deleted_summaries, _ = MonitoringSummary.objects.all().delete()

        rebuild_payload = {}
        if options.get("rebuild"):
            for engine in (ENGINE_BERT, ENGINE_LLM):
                DailyFactorSentiment.objects.filter(engine=engine).delete()
                DailySentimentSummary.objects.filter(engine=engine).delete()
                rebuild_payload[engine] = rebuild_sentiment_history(engine=engine)

        self.stdout.write(
            self.style.SUCCESS(
                f"Clean completed. Deleted seed rows: {deleted_news}, deleted summaries: {deleted_summaries}, "
                f"rebuilt engines: {', '.join(rebuild_payload) if rebuild_payload else 'none'}."
            )
        )
