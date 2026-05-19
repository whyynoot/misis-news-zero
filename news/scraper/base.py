import datetime as dt
import re
from html import unescape
from typing import Any, Protocol


class NewsScraper(Protocol):
    source: str

    def get_news_entries(self, limit: int | None = None) -> list[dict[str, Any]]:
        ...

    def get_news_entries_for_date(self, target_date: dt.date, limit: int | None = None) -> list[dict[str, Any]]:
        ...

    def get_news(self, limit: int | None = None) -> list[str]:
        ...


def clean_text(value: Any) -> str:
    if not value:
        return ""
    text = re.sub(r"<[^>]+>", "", str(value))
    text = unescape(text).replace("\xa0", " ")
    return " ".join(text.split()).strip()
