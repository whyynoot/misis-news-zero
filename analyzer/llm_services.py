import datetime
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable, Sequence

from decouple import config
from django.db import transaction
from django.utils import timezone

from analyzer.constants import (
    ENGINE_LLM,
    FACTOR_CATALOG_VERSION,
    normalize_factor_key,
)
from analyzer.llm_client import LLMClient, LLMDisabledError, LLMError, get_llm_settings
from analyzer.llm_prompts import (
    CLASSIFICATION_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    build_classification_user_prompt,
    build_summary_user_prompt,
    compact_json,
)

logger = logging.getLogger(__name__)

VALID_LABELS = {"positive", "neutral", "negative"}
VALID_TRENDS = {"improving", "worsening", "stable", "mixed"}
VALID_RISK_LEVELS = {"low", "medium", "high"}
DEFAULT_SUMMARY_BATCH_FACTOR_SIZE = max(config("LLM_SUMMARY_BATCH_FACTOR_SIZE", default=1, cast=int), 1)
REQUIRED_CLASSIFICATION_FIELDS = {
    "factor_id",
    "relevance",
    "sentiment",
    "pressure",
    "confidence",
    "label",
    "evidence",
    "reason",
}
REQUIRED_SUMMARY_FIELDS = {
    "factor_id",
    "trend",
    "risk_level",
    "confidence",
    "summary",
    "main_drivers",
}


def clamp(value: Any, minimum: float, maximum: float, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def trim_words(value: Any, limit: int) -> str:
    words = str(value or "").strip().split()
    return " ".join(words[:limit])


def news_content_hash(news_item) -> str:
    normalized = " ".join(
        f"{news_item.title or ''} {news_item.summary or ''} {news_item.text or ''}".lower().split()
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(compact_json(value).encode("utf-8")).hexdigest()


def chunked(items: Sequence[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def factor_catalog_payload(factors) -> list[dict[str, Any]]:
    return [
        {
            "factor_id": factor.key,
            "name": factor.name,
            "meaning": factor.description,
            "positive_signal": factor.positive_label,
            "negative_signal": factor.negative_label,
        }
        for factor in factors
    ]


def news_payload(news_items) -> list[dict[str, Any]]:
    payload = []
    for item in news_items:
        text = item.text or item.summary or item.title
        payload.append(
            {
                "news_id": str(item.id),
                "date": item.published_at.date().isoformat(),
                "title": item.title,
                "summary": item.summary,
                "text": text[:1600],
                "url": item.url,
            }
        )
    return payload


def normalize_classification_factor(raw_factor: dict[str, Any] | None, factor_key: str) -> dict[str, Any]:
    raw_factor = raw_factor or {}
    relevance = clamp(raw_factor.get("relevance"), 0.0, 1.0)
    sentiment = clamp(raw_factor.get("sentiment"), -1.0, 1.0)
    pressure = clamp(raw_factor.get("pressure"), 0.0, 1.0)
    confidence = clamp(raw_factor.get("confidence"), 0.0, 1.0)
    label = str(raw_factor.get("label") or "neutral").strip().lower()
    if label not in VALID_LABELS and raw_factor:
        label = "positive" if sentiment > 0.2 else "negative" if sentiment < -0.2 else "neutral"
    if relevance <= 0:
        sentiment = 0.0
        pressure = 0.0
        label = "neutral"

    return {
        "factor_id": factor_key,
        "relevance": relevance,
        "sentiment": sentiment,
        "pressure": pressure,
        "confidence": confidence,
        "label": label,
        "evidence": trim_words(raw_factor.get("evidence"), 12),
        "reason": trim_words(raw_factor.get("reason") or "нет связи", 12),
        "raw": raw_factor,
    }


def normalize_classification_response(response: dict[str, Any], news_items, factors) -> tuple[list[dict[str, Any]], list[str]]:
    factors_by_key = {factor.key: factor for factor in factors}
    raw_items = response.get("items") or []
    raw_by_news_id = {}
    warnings = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        news_id = str(raw_item.get("news_id") or "")
        if news_id:
            raw_by_news_id[news_id] = raw_item

    normalized = []
    for news_item in news_items:
        raw_item = raw_by_news_id.get(str(news_item.id))
        raw_factor_map = {}
        if raw_item:
            for raw_factor in raw_item.get("factors") or []:
                if not isinstance(raw_factor, dict):
                    continue
                factor_key = normalize_factor_key(str(raw_factor.get("factor_id") or ""))
                if factor_key in factors_by_key:
                    raw_factor_map[factor_key] = raw_factor
        else:
            warnings.append(f"LLM response missed news_id={news_item.id}")

        normalized_factors = []
        for factor in factors:
            normalized_factors.append(normalize_classification_factor(raw_factor_map.get(factor.key), factor.key))

        normalized.append(
            {
                "news_item": news_item,
                "factors": normalized_factors,
                "raw": raw_item or {},
            }
        )

    return normalized, warnings


def llm_classification_identity(settings=None) -> dict[str, str]:
    settings = settings or get_llm_settings()
    return {
        "engine": ENGINE_LLM,
        "model_name": settings.model,
        "prompt_version": settings.classification_prompt_version,
        "factor_catalog_version": FACTOR_CATALOG_VERSION,
    }


def has_cached_llm_classification(news_item, factors, identity: dict[str, str]) -> bool:
    from analyzer.models import NewsClassification

    return (
        NewsClassification.objects.filter(
            news_item=news_item,
            factor__in=factors,
            content_hash=news_content_hash(news_item),
            **identity,
        ).count()
        >= len(factors)
    )


def classify_news_items_with_llm(
    news_items,
    factors,
    batch=None,
    force: bool = False,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    from analyzer.models import NewsClassification

    settings = get_llm_settings()
    identity = llm_classification_identity(settings)
    if client is None:
        if not settings.enabled:
            raise LLMDisabledError("LLM is disabled. Set LLM_ENABLED=true.")
        client = LLMClient(settings)

    pending_items = [
        item for item in news_items if force or not has_cached_llm_classification(item, factors, identity)
    ]
    if not pending_items:
        return {
            "classified_count": 0,
            "affected_dates": sorted({item.published_at.date() for item in news_items}),
            "errors": [],
            "cached_count": len(news_items),
        }

    classified_count = 0
    affected_dates = set()
    errors = []
    warnings = []
    catalog = factor_catalog_payload(factors)

    def request_batch(batch_items):
        batch_client = client if settings.concurrency == 1 else LLMClient(settings)
        try:
            user_prompt = build_classification_user_prompt(news_payload(batch_items), catalog)
            response, raw_text = batch_client.complete_json(CLASSIFICATION_SYSTEM_PROMPT, user_prompt)
            normalized_items, normalize_warnings = normalize_classification_response(response, batch_items, factors)
            return normalized_items, [], normalize_warnings, raw_text
        except LLMError as exc:
            logger.exception("LLM classification batch failed")
            return [], [str(exc)], [], ""

    def persist_batch(normalized_items, raw_text):
        if not normalized_items:
            return 0

        saved_count = 0
        with transaction.atomic():
            for item_payload in normalized_items:
                news_item = item_payload["news_item"]
                content_hash = news_content_hash(news_item)
                affected_dates.add(news_item.published_at.date())
                for factor_payload in item_payload["factors"]:
                    factor = next(factor for factor in factors if factor.key == factor_payload["factor_id"])
                    sentiment = factor_payload["sentiment"]
                    positive_probability = (sentiment + 1.0) / 2.0
                    negative_probability = 1.0 - positive_probability
                    is_relevant = (
                        factor_payload["relevance"] >= 0.3
                        and (
                            abs(sentiment) >= 0.2
                            or factor_payload["pressure"] >= 0.2
                            or factor_payload["label"] != "neutral"
                        )
                    )
                    NewsClassification.objects.update_or_create(
                        news_item=news_item,
                        factor=factor,
                        **identity,
                        defaults={
                            "batch": batch,
                            "positive_probability": positive_probability,
                            "negative_probability": negative_probability,
                            "sentiment_score": sentiment,
                            "relevance": factor_payload["relevance"],
                            "pressure": factor_payload["pressure"],
                            "sentiment_label": factor_payload["label"],
                            "confidence": factor_payload["confidence"],
                            "evidence": factor_payload["evidence"] if factor_payload["raw"] else "",
                            "reason": factor_payload["reason"] if factor_payload["raw"] else "",
                            "is_relevant": is_relevant,
                            "is_seed": False,
                            "content_hash": content_hash,
                            "raw_response": {
                                "item": item_payload["raw"],
                                "factor": factor_payload["raw"],
                                "raw_text": raw_text[:12000],
                            },
                        },
                    )
                    saved_count += 1
        return saved_count

    batches = list(chunked(pending_items, settings.batch_news_size))
    if settings.concurrency > 1 and len(batches) > 1:
        with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
            futures = [pool.submit(request_batch, batch_items) for batch_items in batches]
            for future in as_completed(futures):
                normalized_items, batch_errors, batch_warnings, raw_text = future.result()
                errors.extend(batch_errors)
                warnings.extend(batch_warnings)
                classified_count += persist_batch(normalized_items, raw_text)
    else:
        for batch_items in batches:
            normalized_items, batch_errors, batch_warnings, raw_text = request_batch(batch_items)
            errors.extend(batch_errors)
            warnings.extend(batch_warnings)
            classified_count += persist_batch(normalized_items, raw_text)

    return {
        "classified_count": classified_count,
        "affected_dates": sorted(affected_dates),
        "errors": errors,
        "warnings": warnings,
        "cached_count": len(news_items) - len(pending_items),
    }


def parse_date(value: Any) -> datetime.date:
    if isinstance(value, datetime.date):
        return value
    return datetime.date.fromisoformat(str(value))


def resolve_period_bounds(
    engine: str,
    period: str = "day",
    from_date: Any = None,
    to_date: Any = None,
) -> tuple[datetime.date, datetime.date]:
    from analyzer.models import DailySentimentSummary

    if from_date and to_date:
        start = parse_date(from_date)
        end = parse_date(to_date)
        return min(start, end), max(start, end)

    settings = get_llm_settings()
    filter_kwargs = llm_classification_identity(settings) if engine == ENGINE_LLM else {"engine": engine}
    latest = DailySentimentSummary.objects.filter(**filter_kwargs).order_by("-date").first()
    end = latest.date if latest else timezone.localdate()
    period = (period or "day").lower()
    if period == "week":
        return end - datetime.timedelta(days=6), end
    if period == "month":
        return end - datetime.timedelta(days=29), end
    return end, end


def top_factor_classifications(factor, start: datetime.date, end: datetime.date, identity: dict[str, str], limit: int):
    from analyzer.models import NewsClassification

    queryset = (
        NewsClassification.objects.select_related("news_item")
        .filter(
            factor=factor,
            news_item__published_at__date__gte=start,
            news_item__published_at__date__lte=end,
            is_relevant=True,
            **identity,
        )
        .order_by("-news_item__published_at")
    )
    items = list(queryset)
    items.sort(
        key=lambda item: (
            item.relevance * 1.4
            + abs(item.sentiment_score) * 1.2
            + item.pressure * 1.6
            + item.confidence * 0.2
        ),
        reverse=True,
    )
    return items[:limit]


def build_factor_summary_input(
    factor,
    start: datetime.date,
    end: datetime.date,
    identity: dict[str, str],
    top_n: int = 8,
) -> dict[str, Any]:
    top_items = top_factor_classifications(factor, start, end, identity, top_n)
    if top_items:
        average_sentiment = sum(item.sentiment_score for item in top_items) / len(top_items)
        average_pressure = sum(item.pressure for item in top_items) / len(top_items)
    else:
        average_sentiment = 0.0
        average_pressure = 0.0

    return {
        "factor_id": factor.key,
        "factor_name": factor.name,
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "metrics": {
            "top_news_count": len(top_items),
            "average_sentiment": average_sentiment,
            "average_pressure": average_pressure,
            "negative_count": len([item for item in top_items if item.sentiment_label == "negative"]),
            "positive_count": len([item for item in top_items if item.sentiment_label == "positive"]),
        },
        "top_news": [
            {
                "date": item.news_item.published_at.date().isoformat(),
                "title": item.news_item.title,
                "summary": (item.news_item.summary or item.news_item.text or "")[:600],
                "sentiment": item.sentiment_score,
                "pressure": item.pressure,
                "relevance": item.relevance,
                "evidence": item.evidence,
                "reason": item.reason,
                "url": item.news_item.url,
            }
            for item in top_items
        ],
    }


def llm_summary_identity(settings=None) -> dict[str, str]:
    settings = settings or get_llm_settings()
    return {
        "engine": ENGINE_LLM,
        "model_name": settings.model,
        "prompt_version": settings.summary_prompt_version,
        "factor_catalog_version": FACTOR_CATALOG_VERSION,
    }


def normalize_summary_factor(raw_factor: dict[str, Any] | None, factor_input: dict[str, Any]) -> dict[str, Any] | None:
    raw_factor = raw_factor or {}
    trend = str(raw_factor.get("trend") or "stable").strip().lower()
    if trend not in VALID_TRENDS:
        trend = "mixed"
    risk_level = str(raw_factor.get("risk_level") or "low").strip().lower()
    if risk_level not in VALID_RISK_LEVELS:
        risk_level = "low"

    drivers = []
    for item in raw_factor.get("main_drivers") or []:
        if not isinstance(item, dict):
            continue
        drivers.append(
            {
                "title": trim_words(item.get("title"), 18),
                "date": str(item.get("date") or factor_input["period_end"]),
                "impact": clamp(item.get("impact"), -1.0, 1.0),
                "why": trim_words(item.get("why"), 14),
            }
        )
        if len(drivers) >= 5:
            break

    summary_text = str(raw_factor.get("summary") or "").strip()
    if not summary_text:
        return None

    return {
        "factor_id": factor_input["factor_id"],
        "trend": trend,
        "risk_level": risk_level,
        "confidence": clamp(raw_factor.get("confidence"), 0.0, 1.0),
        "summary": summary_text,
        "main_drivers": drivers[:5],
        "raw": raw_factor,
    }


def generate_llm_summaries(
    period: str = "day",
    from_date: Any = None,
    to_date: Any = None,
    factor_key: str | None = None,
    force: bool = False,
    client: LLMClient | None = None,
    top_n: int = 8,
    factor_batch_size: int | None = None,
) -> dict[str, Any]:
    from analyzer.models import MonitoringSummary, RiskFactor

    settings = get_llm_settings()
    identity = llm_summary_identity(settings)
    classification_identity = llm_classification_identity(settings)
    if client is None:
        if not settings.enabled:
            raise LLMDisabledError("LLM is disabled. Set LLM_ENABLED=true.")
        client = LLMClient(settings)

    start, end = resolve_period_bounds(ENGINE_LLM, period, from_date, to_date)
    factors = RiskFactor.objects.filter(is_active=True).order_by("display_order", "name")
    normalized_factor_key = normalize_factor_key(factor_key)
    if normalized_factor_key:
        factors = factors.filter(key=normalized_factor_key)
    factors = list(factors)

    factor_inputs = [
        build_factor_summary_input(factor, start, end, classification_identity, top_n=top_n)
        for factor in factors
    ]
    input_hashes = {item["factor_id"]: stable_hash(item) for item in factor_inputs}

    cached = {}
    if not force:
        for summary in MonitoringSummary.objects.select_related("factor").filter(
            factor__in=factors,
            period_start=start,
            period_end=end,
            **identity,
        ):
            if summary.input_hash == input_hashes.get(summary.factor.key):
                cached[summary.factor.key] = summary

    pending_inputs = [item for item in factor_inputs if item["factor_id"] not in cached]
    if not pending_inputs:
        return {
            "created_count": 0,
            "cached_count": len(cached),
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "errors": [],
            "warnings": [],
        }

    created_count = 0
    errors = []
    warnings = []
    factor_by_key = {factor.key: factor for factor in factors}

    effective_batch_size = max(int(factor_batch_size or DEFAULT_SUMMARY_BATCH_FACTOR_SIZE), 1)
    for input_batch in chunked(pending_inputs, effective_batch_size):
        summary_input = {
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "engine": ENGINE_LLM,
            "factors": input_batch,
        }
        try:
            response, raw_text = client.complete_json(SUMMARY_SYSTEM_PROMPT, build_summary_user_prompt(summary_input))
        except LLMError as exc:
            logger.exception("LLM summary batch failed")
            factor_ids = ", ".join(item["factor_id"] for item in input_batch)
            errors.append(f"LLM summary failed for factors={factor_ids}: {exc}")
            continue

        raw_by_factor = {}
        for raw_factor in response.get("factors") or []:
            if not isinstance(raw_factor, dict):
                continue
            key = normalize_factor_key(raw_factor.get("factor_id"))
            if key:
                raw_by_factor[key] = raw_factor

        with transaction.atomic():
            for factor_input in input_batch:
                factor_key = factor_input["factor_id"]
                raw_factor = raw_by_factor.get(factor_key)
                if raw_factor is None:
                    errors.append(f"LLM summary missed factor={factor_key}")
                    continue
                normalized = normalize_summary_factor(raw_factor, factor_input)
                if normalized is None:
                    warnings.append(f"LLM summary returned empty summary for factor={factor_key}")
                    continue
                if not normalized["main_drivers"]:
                    warnings.append(f"LLM summary returned no main_drivers for factor={factor_key}")
                    continue
                MonitoringSummary.objects.update_or_create(
                    factor=factor_by_key[factor_key],
                    period_start=start,
                    period_end=end,
                    **identity,
                    defaults={
                        "input_hash": input_hashes[factor_key],
                        "trend": normalized["trend"],
                        "risk_level": normalized["risk_level"],
                        "confidence": normalized["confidence"],
                        "summary": normalized["summary"],
                        "main_drivers": normalized["main_drivers"],
                        "raw_response": {
                            "factor": normalized["raw"],
                            "input": factor_input,
                            "raw_text": raw_text[:12000],
                        },
                    },
                )
                created_count += 1

    return {
        "created_count": created_count,
        "cached_count": len(cached),
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "errors": errors,
        "warnings": warnings,
    }


def serialize_summary(summary) -> dict[str, Any]:
    return {
        "factor_id": summary.factor.key,
        "factor_name": summary.factor.name,
        "period_start": summary.period_start.isoformat(),
        "period_end": summary.period_end.isoformat(),
        "engine": summary.engine,
        "model_name": summary.model_name,
        "prompt_version": summary.prompt_version,
        "trend": summary.trend,
        "risk_level": summary.risk_level,
        "confidence": summary.confidence,
        "summary": summary.summary,
        "main_drivers": summary.main_drivers,
        "updated_at": timezone.localtime(summary.updated_at).isoformat(),
    }


def get_summary_feed(
    engine: str = ENGINE_LLM,
    period: str = "day",
    from_date: Any = None,
    to_date: Any = None,
    factor_key: str | None = None,
) -> dict[str, Any]:
    from analyzer.models import MonitoringSummary, RiskFactor

    settings = get_llm_settings()
    start, end = resolve_period_bounds(engine, period, from_date, to_date)
    factors = RiskFactor.objects.filter(is_active=True).order_by("display_order", "name")
    normalized_factor_key = normalize_factor_key(factor_key)
    if normalized_factor_key:
        factors = factors.filter(key=normalized_factor_key)
    factors = list(factors)

    summaries = {
        summary.factor_id: summary
        for summary in MonitoringSummary.objects.select_related("factor").filter(
            factor__in=factors,
            period_start=start,
            period_end=end,
            engine=ENGINE_LLM,
            model_name=settings.model,
            prompt_version=settings.summary_prompt_version,
            factor_catalog_version=FACTOR_CATALOG_VERSION,
        )
    }
    items = []
    for factor in factors:
        summary = summaries.get(factor.id)
        if summary:
            items.append(serialize_summary(summary))
        else:
            items.append(
                {
                    "factor_id": factor.key,
                    "factor_name": factor.name,
                    "period_start": start.isoformat(),
                    "period_end": end.isoformat(),
                    "engine": ENGINE_LLM,
                    "trend": None,
                    "risk_level": None,
                    "confidence": 0,
                    "summary": "",
                    "main_drivers": [],
                    "status": "missing",
                }
            )

    return {
        "engine": engine,
        "summary_engine": ENGINE_LLM,
        "model_name": settings.model,
        "prompt_version": settings.summary_prompt_version,
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "status": "ready" if any(item.get("summary") for item in items) else "missing",
        "llm_enabled": settings.enabled,
        "factors": items,
    }
