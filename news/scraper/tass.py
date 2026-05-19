import logging
import xml.etree.ElementTree as ET
from datetime import date
from email.utils import parsedate_to_datetime

import requests

from news.scraper.base import clean_text

logger = logging.getLogger(__name__)


class TassScraper:
    source = "tass"

    def __init__(self):
        self.feed_url = "https://tass.ru/rss/v2.xml"
        self.parse_pages = 10
        self.batch_size = 20
        self.relevant_categories = {
            "Общество",
            "Экономика и бизнес",
            "Наука",
            "Недвижимость",
            "Москва",
        }
        self.relevant_keywords = (
            "демограф",
            "семь",
            "рождаем",
            "жиль",
            "ипотек",
            "безработ",
            "занятост",
            "медицин",
            "здравоохр",
            "доход",
            "бедност",
            "соц",
            "населен",
            "жизн",
        )
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; SocialRiskMonitor/2.0; +https://example.com)",
            "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8",
        }

    def _parse_pub_date(self, value):
        if not value:
            return None
        try:
            return parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None

    def _is_relevant(self, category, text):
        normalized_category = (category or "").strip()
        normalized_text = (text or "").lower()
        return normalized_category in self.relevant_categories or any(
            keyword in normalized_text for keyword in self.relevant_keywords
        )

    def get_news_entries(self, limit=None):
        try:
            response = requests.get(self.feed_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except (requests.exceptions.RequestException, ET.ParseError) as exc:
            logger.warning("Error fetching news from TASS RSS feed: %s", exc)
            return []

        max_items = max(int(limit), 1) if limit else max(self.parse_pages, 1) * self.batch_size
        relevant_items = []
        fallback_items = []
        seen = set()

        for item in root.findall("./channel/item"):
            title = clean_text(item.findtext("title"))
            summary = clean_text(item.findtext("description"))
            category = clean_text(item.findtext("category"))
            url = clean_text(item.findtext("link"))
            published_at = self._parse_pub_date(item.findtext("pubDate"))
            if not title:
                continue

            text = f"{title}: {summary}" if summary else title
            dedupe_key = url or text
            if dedupe_key in seen:
                continue

            payload = {
                "source": self.source,
                "title": title,
                "summary": summary,
                "text": text,
                "url": url,
                "category": category,
                "published_at": published_at,
                "external_id": url or "",
            }

            seen.add(dedupe_key)
            fallback_items.append(payload)
            if self._is_relevant(category, text):
                relevant_items.append(payload)

            if len(fallback_items) >= max_items:
                break

        return relevant_items[:max_items] or fallback_items[:max_items]

    def get_news_entries_for_date(self, target_date: date, limit=None):
        entries = [
            entry
            for entry in self.get_news_entries(limit=limit)
            if entry.get("published_at") and entry["published_at"].date() == target_date
        ]
        return entries[: max(int(limit), 1)] if limit else entries

    def get_news(self, limit=None):
        return [entry["text"] for entry in self.get_news_entries(limit=limit)]
