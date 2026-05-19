from django.core.management.base import BaseCommand

from analyzer.llm_services import generate_llm_summaries


class Command(BaseCommand):
    help = "Генерирует LLM-summary по социальным факторам за выбранный период."

    def add_arguments(self, parser):
        parser.add_argument(
            "--engine",
            choices=["llm"],
            default="llm",
            help="Движок классификаций для summary. Сейчас поддержан llm.",
        )
        parser.add_argument(
            "--period",
            choices=["day", "week", "month"],
            default="day",
            help="Период summary, если не указан custom range.",
        )
        parser.add_argument("--from", dest="from_date", default=None, help="Начальная дата YYYY-MM-DD.")
        parser.add_argument("--to", dest="to_date", default=None, help="Конечная дата YYYY-MM-DD.")
        parser.add_argument("--factor", default=None, help="Ключ или алиас фактора, например medicine.")
        parser.add_argument("--force", action="store_true", help="Перегенерировать summary, игнорируя кэш.")
        parser.add_argument("--top-n", type=int, default=8, help="Сколько главных новостей передавать в LLM.")

    def handle(self, *args, **options):
        result = generate_llm_summaries(
            period=options.get("period") or "day",
            from_date=options.get("from_date"),
            to_date=options.get("to_date"),
            factor_key=options.get("factor"),
            force=options.get("force", False),
            top_n=options.get("top_n") or 8,
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"Summary готово: created={result['created_count']}, cached={result['cached_count']}, "
                f"period={result['period_start']}..{result['period_end']}"
            )
        )
        if result.get("errors"):
            self.stdout.write(self.style.WARNING("Ошибки: " + "; ".join(result["errors"][:3])))
