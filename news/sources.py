import logging
import time
from datetime import date, timedelta
from functools import lru_cache
from typing import Any

from decouple import Csv, config

from news.scraper.interfax import InterfaxScraper
from news.scraper.tass import TassScraper

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_NAMES = ("tass", "interfax")


@lru_cache(maxsize=1)
def get_available_scrapers() -> dict[str, Any]:
    return {
        "tass": TassScraper(),
        "interfax": InterfaxScraper(),
    }


def configured_source_names() -> list[str]:
    names = config("NEWS_SOURCES", default=",".join(DEFAULT_SOURCE_NAMES), cast=Csv())
    return [name.strip().lower() for name in names if name.strip()]


def get_scrapers(source_names: list[str] | None = None) -> dict[str, Any]:
    available = get_available_scrapers()
    selected_names = source_names or configured_source_names()
    selected = {}
    for name in selected_names:
        if name in available:
            selected[name] = available[name]
        else:
            logger.warning("Unknown news source configured: %s", name)
    return selected


def _scraper_entries(scraper: Any, limit: int | None = None) -> list[dict[str, Any]]:
    if hasattr(scraper, "get_news_entries"):
        try:
            return list(scraper.get_news_entries(limit=limit))
        except TypeError:
            return list(scraper.get_news_entries())
    try:
        texts = scraper.get_news(limit=limit)
    except TypeError:
        texts = scraper.get_news()
    return [{"text": text, "source": getattr(scraper, "source", "")} for text in texts]


def _scraper_entries_for_date(scraper: Any, target_date: date, limit: int | None = None) -> list[dict[str, Any]]:
    if hasattr(scraper, "get_news_entries_for_date"):
        try:
            return list(scraper.get_news_entries_for_date(target_date=target_date, limit=limit))
        except TypeError:
            return list(scraper.get_news_entries_for_date(target_date, limit))
    return []


def _iter_dates(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def collect_news_entries(limit: int | None = None, source_names: list[str] | None = None) -> list[dict[str, Any]]:
    entries = []
    for source_name, scraper in get_scrapers(source_names).items():
        try:
            for entry in _scraper_entries(scraper, limit=limit):
                if isinstance(entry, dict):
                    entry.setdefault("source", source_name)
                    entries.append(entry)
                elif entry:
                    entries.append({"text": str(entry), "source": source_name})
        except Exception as exc:  # pragma: no cover - network guard
            logger.warning("Failed to fetch news source %s: %s", source_name, exc)
    return entries


def collect_historical_news_entries(
    start_date: date,
    end_date: date,
    limit_per_day: int | None = None,
    source_names: list[str] | None = None,
    delay_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    entries = []
    scrapers = get_scrapers(source_names)
    for target_date in _iter_dates(start_date, end_date):
        for source_name, scraper in scrapers.items():
            try:
                for entry in _scraper_entries_for_date(scraper, target_date=target_date, limit=limit_per_day):
                    if isinstance(entry, dict):
                        entry.setdefault("source", source_name)
                        entries.append(entry)
                    elif entry:
                        entries.append({"text": str(entry), "source": source_name, "published_at": target_date.isoformat()})
            except Exception as exc:  # pragma: no cover - network guard
                logger.warning("Failed to fetch historical news source %s for %s: %s", source_name, target_date, exc)
        if delay_seconds > 0:
            time.sleep(delay_seconds)
    return entries
