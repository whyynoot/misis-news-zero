import datetime as dt
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup, Tag
from decouple import Csv, config

from news.scraper.base import clean_text

logger = logging.getLogger(__name__)

ARTICLE_PATH_RE = re.compile(r"^/([^/?#]+)/([0-9]+)(?:[/?#].*)?$")
DEFAULT_SECTIONS = ("russia", "business", "world")
EXCLUDE_URL_PREFIXES = ("/chronicle/", "/photo/", "/story/", "/search/", "/events/")
RELEVANT_KEYWORDS = (
    "населен",
    "демограф",
    "рождаем",
    "смерт",
    "брак",
    "развод",
    "миграц",
    "доход",
    "зарплат",
    "безработ",
    "прожиточ",
    "больниц",
    "поликлиник",
    "врач",
    "пособ",
    "материнск",
    "дет",
    "сем",
    "жиль",
    "ипотек",
    "цены",
    "инфляц",
    "преступ",
    "загряз",
    "эколог",
    "производств",
    "предприяти",
    "аборт",
)


@dataclass(frozen=True)
class InterfaxArticleLink:
    url: str
    title: str
    section: str
    published_at: dt.datetime | None
    summary: str = ""


class InterfaxScraper:
    source = "interfax"

    def __init__(
        self,
        base_url: str = "https://www.interfax.ru",
        sections: tuple[str, ...] | None = None,
        timeout: int = 12,
        hydrate_articles: bool | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.sections = sections or tuple(config("INTERFAX_SECTIONS", default=",".join(DEFAULT_SECTIONS), cast=Csv()))
        self.timeout = timeout
        self.hydrate_articles = (
            config("INTERFAX_HYDRATE_ARTICLES", default=True, cast=bool)
            if hydrate_articles is None
            else hydrate_articles
        )
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (compatible; SocialRiskMonitor/2.0; +https://example.com)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "ru,en-US;q=0.9,en;q=0.8",
            }
        )

    def _fetch(self, url: str, referer: str | None = None) -> str:
        headers = {"Referer": referer} if referer else None
        response = self.session.get(url, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding or "utf-8"
        return response.text

    def _normalize_url(self, href: str) -> str:
        absolute = urljoin(self.base_url + "/", href)
        parts = urlsplit(absolute)
        path = parts.path.rstrip("/") if re.search(r"/\d+$", parts.path.rstrip("/")) else parts.path
        return urlunsplit((parts.scheme, parts.netloc, path, "", ""))

    def _is_article_href(self, href: str) -> bool:
        parts = urlsplit(href)
        base_host = urlsplit(self.base_url).netloc
        if parts.netloc and parts.netloc != base_host:
            return False
        path = parts.path
        if not path:
            return False
        if any(path.startswith(prefix) for prefix in EXCLUDE_URL_PREFIXES):
            return False
        return ARTICLE_PATH_RE.match(path) is not None

    def _parse_datetime(self, value: str | None) -> dt.datetime | None:
        text = clean_text(value)
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = dt.datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=ZoneInfo("Europe/Moscow"))
        return parsed

    def _parse_archive_datetime(self, target_date: dt.date, value: str | None) -> dt.datetime | None:
        text = clean_text(value)
        if not text:
            return dt.datetime.combine(target_date, dt.time(12, 0), tzinfo=ZoneInfo("Europe/Moscow"))
        match = re.search(r"(?P<hour>\d{1,2}):(?P<minute>\d{2})", text)
        if not match:
            return dt.datetime.combine(target_date, dt.time(12, 0), tzinfo=ZoneInfo("Europe/Moscow"))
        return dt.datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            int(match.group("hour")),
            int(match.group("minute")),
            tzinfo=ZoneInfo("Europe/Moscow"),
        )

    def _meta_one(self, soup: BeautifulSoup, key: str) -> str:
        node = soup.select_one(f'meta[property="{key}"]') or soup.select_one(f'meta[name="{key}"]')
        return clean_text(node.get("content")) if node else ""

    def _parse_section_page(self, html: str, section: str) -> list[InterfaxArticleLink]:
        soup = BeautifulSoup(html, "html.parser")
        timeline = soup.select_one(".timeline") or soup
        selectors = (
            ".timeline__smalltext",
            ".timeline__text",
            ".timeline__text-large",
            ".timeline__photo",
            ".timeline__group > div",
        )
        links = []
        seen = set()
        for node in timeline.select(", ".join(selectors)):
            if not isinstance(node, Tag):
                continue
            time_el = node.select_one("time[datetime]")
            article_link = next(
                (item for item in node.select("a[href]") if self._is_article_href(item.get("href", ""))),
                None,
            )
            if not article_link:
                continue
            url = self._normalize_url(article_link.get("href", ""))
            if url in seen:
                continue
            seen.add(url)
            title = clean_text(article_link.get("title") or article_link.get_text(" ", strip=True))
            summary = clean_text(node.select_one(".timeline__lead").get_text(" ", strip=True)) if node.select_one(".timeline__lead") else ""
            links.append(
                InterfaxArticleLink(
                    url=url,
                    title=title,
                    section=section,
                    published_at=self._parse_datetime(time_el.get("datetime") if time_el else None),
                    summary=summary,
                )
            )
        return links

    def _parse_archive_page(self, html: str, target_date: dt.date) -> list[InterfaxArticleLink]:
        soup = BeautifulSoup(html, "html.parser")
        archive = soup.select_one(".an")
        if not archive:
            return []

        links = []
        seen = set()
        for node in archive.select("div[data-id]"):
            if not isinstance(node, Tag):
                continue
            article_link = next(
                (item for item in node.select("a[href]") if self._is_article_href(item.get("href", ""))),
                None,
            )
            if not article_link:
                continue

            url = self._normalize_url(article_link.get("href", ""))
            if url in seen:
                continue
            seen.add(url)

            path_parts = [part for part in urlsplit(url).path.split("/") if part]
            section = path_parts[0] if path_parts else "news"
            title = clean_text(article_link.get("title") or article_link.get_text(" ", strip=True))
            time_text = clean_text(node.select_one("span").get_text(" ", strip=True)) if node.select_one("span") else ""
            links.append(
                InterfaxArticleLink(
                    url=url,
                    title=title,
                    section=section,
                    published_at=self._parse_archive_datetime(target_date, time_text),
                )
            )
        return links

    def _parse_article_page(self, html: str, source_url: str) -> dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        article = soup.find("article", attrs={"itemprop": "articleBody"})
        if not article:
            return {}

        canonical_node = soup.select_one('link[rel="canonical"]')
        canonical = self._normalize_url(canonical_node.get("href")) if canonical_node and canonical_node.get("href") else source_url
        title_node = article.select_one('h1[itemprop="headline"]') or soup.select_one("h1")
        title = clean_text(title_node.get_text(" ", strip=True)) if title_node else ""
        description = self._meta_one(soup, "description") or self._meta_one(soup, "og:description")
        section_title = self._meta_one(soup, "article:section")
        published_at = self._parse_datetime(
            self._meta_one(soup, "article:published_time")
            or (article.select_one('meta[itemprop="datePublished"]') or {}).get("content")
        )

        article_clone = BeautifulSoup(str(article), "html.parser")
        for selector in ("aside.textMUpdate", ".wg_news", ".ads__ban_inline", "script", "style", ".socBlock", ".article__share"):
            for node in article_clone.select(selector):
                node.decompose()

        article_root = article_clone.find("article") or article_clone
        body_blocks = []
        for child in article_root.children:
            if not isinstance(child, Tag):
                continue
            if child.name in {"p", "h2", "h3", "blockquote", "ul", "ol"}:
                text = clean_text(child.get_text(" ", strip=True))
                if text:
                    body_blocks.append(text)

        return {
            "url": canonical,
            "title": title,
            "summary": body_blocks[0] if body_blocks else description,
            "text": "\n\n".join(body_blocks) if body_blocks else description,
            "category": section_title,
            "published_at": published_at,
        }

    def _to_entry(self, link: InterfaxArticleLink, article_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        article_payload = article_payload or {}
        title = article_payload.get("title") or link.title
        summary = article_payload.get("summary") or link.summary
        text = article_payload.get("text") or f"{title}: {summary}".strip(": ")
        url = article_payload.get("url") or link.url
        return {
            "source": self.source,
            "external_id": url,
            "url": url,
            "title": title,
            "summary": summary,
            "text": text or title,
            "category": article_payload.get("category") or link.section,
            "published_at": article_payload.get("published_at") or link.published_at,
        }

    def _is_relevant(self, entry: dict[str, Any]) -> bool:
        text = f"{entry.get('title', '')} {entry.get('summary', '')} {entry.get('text', '')}".lower()
        return any(keyword in text for keyword in RELEVANT_KEYWORDS)

    def get_news_entries(self, limit: int | None = None) -> list[dict[str, Any]]:
        per_source_limit = max(int(limit), 1) if limit else 50
        links = []
        for section in self.sections:
            page_url = f"{self.base_url}/{section}/"
            try:
                links.extend(self._parse_section_page(self._fetch(page_url), section))
            except requests.RequestException as exc:
                logger.warning("Error fetching Interfax section %s: %s", section, exc)

        links = sorted(
            {link.url: link for link in links}.values(),
            key=lambda item: item.published_at or dt.datetime.min.replace(tzinfo=ZoneInfo("Europe/Moscow")),
            reverse=True,
        )[:per_source_limit]

        entries = []
        for link in links:
            article_payload = {}
            if self.hydrate_articles:
                try:
                    article_payload = self._parse_article_page(self._fetch(link.url, referer=f"{self.base_url}/{link.section}/"), link.url)
                except requests.RequestException as exc:
                    logger.warning("Error hydrating Interfax article %s: %s", link.url, exc)
            entries.append(self._to_entry(link, article_payload))

        relevant_entries = [entry for entry in entries if self._is_relevant(entry)]
        return relevant_entries[:per_source_limit] or entries[:per_source_limit]

    def get_news(self, limit: int | None = None) -> list[str]:
        return [entry["text"] for entry in self.get_news_entries(limit=limit)]

    def get_news_entries_for_date(self, target_date: dt.date, limit: int | None = None) -> list[dict[str, Any]]:
        per_day_limit = max(int(limit), 1) if limit else 20
        archive_url = f"{self.base_url}/news/{target_date:%Y/%m/%d}"
        try:
            html = self._fetch(archive_url)
        except requests.RequestException as exc:
            logger.warning("Error fetching Interfax archive %s: %s", archive_url, exc)
            return []

        links = self._parse_archive_page(html, target_date=target_date) or self._parse_section_page(html, "news")
        links = sorted(
            {link.url: link for link in links}.values(),
            key=lambda item: item.published_at or dt.datetime.min.replace(tzinfo=ZoneInfo("Europe/Moscow")),
            reverse=True,
        )

        entries = []
        for link in links:
            article_payload = {}
            if self.hydrate_articles:
                try:
                    article_payload = self._parse_article_page(self._fetch(link.url, referer=archive_url), link.url)
                except requests.RequestException as exc:
                    logger.warning("Error hydrating Interfax archive article %s: %s", link.url, exc)
            entry = self._to_entry(link, article_payload)
            if self._is_relevant(entry):
                entries.append(entry)
            if len(entries) >= per_day_limit:
                break

        return entries[:per_day_limit]
