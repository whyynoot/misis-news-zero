from django.core.management.base import BaseCommand

from analyzer.monitoring_service import ensure_factor_catalog


class Command(BaseCommand):
    help = "Create or update the active monitoring factor catalog."

    def handle(self, *args, **options):
        factors = ensure_factor_catalog()
        self.stdout.write(self.style.SUCCESS(f"Factor catalog synced. Active factors: {len(factors)}."))
