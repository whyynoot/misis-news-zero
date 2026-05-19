import datetime
import hashlib
import json
import logging
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from decouple import config
from django.db import transaction
from django.utils import timezone

from analyzer.analyzers import get_monitoring_analyzer
from analyzer.factors import FACTOR_CONFIG
from analyzer.constants import (
    BERT_MODEL_NAME,
    BERT_PROMPT_VERSION,
    ENGINE_BERT,
    ENGINE_LLM,
    FACTOR_CATALOG_VERSION,
    normalize_engine,
    normalize_factor_key,
)
from analyzer.llm_services import news_content_hash

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
SEED_DATA_PATH = BASE_DIR / "analyzer" / "data" / "social_risk_seed.json"
DEFAULT_NEUTRAL_THRESHOLD = 0.20
DEFAULT_WINDOW_DAYS = 3650
DEFAULT_HISTORY_BACKFILL_LIMIT = max(config("MONITORING_HISTORY_BACKFILL_LIMIT", default=5000, cast=int), 1)
DEFAULT_LIVE_NEWS_LIMIT = max(config("MONITORING_LIVE_NEWS_LIMIT", default=50, cast=int), 1)
DEFAULT_BACKFILL_SOURCE = config("MONITORING_BACKFILL_SOURCE", default="")
DEFAULT_HISTORY_DAYS = max(config("MONITORING_HISTORY_DAYS", default=365, cast=int), 1)
DEFAULT_HISTORY_LIMIT_PER_DAY = max(config("MONITORING_HISTORY_LIMIT_PER_DAY", default=8, cast=int), 1)
DEFAULT_HISTORY_SOURCES = config("MONITORING_HISTORY_SOURCES", default="interfax")
EXCLUDE_SEED_BY_DEFAULT = config("MONITORING_EXCLUDE_SEED", default=True, cast=bool)


@lru_cache(maxsize=1)
def get_classifier():
    from analyzer.zero import ZeroShotClassifier

    return ZeroShotClassifier(
        use_cuda=config("USE_CUDA", default=False, cast=bool),
        batch_size=config("MODEL_BATCH_SIZE", default=8, cast=int),
    )


def get_engine_identity(engine: str = ENGINE_BERT) -> Dict[str, str]:
    engine = normalize_engine(engine)
    if engine == ENGINE_LLM:
        from analyzer.llm_client import get_llm_settings

        settings = get_llm_settings()
        return {
            "engine": ENGINE_LLM,
            "model_name": settings.model,
            "prompt_version": settings.classification_prompt_version,
            "factor_catalog_version": FACTOR_CATALOG_VERSION,
        }
    return {
        "engine": ENGINE_BERT,
        "model_name": BERT_MODEL_NAME,
        "prompt_version": BERT_PROMPT_VERSION,
        "factor_catalog_version": FACTOR_CATALOG_VERSION,
    }


def compute_sentiment_score(positive_probability: float, negative_probability: float) -> float:
    return float(positive_probability) - float(negative_probability)


def label_from_score(score: float, threshold: float = DEFAULT_NEUTRAL_THRESHOLD) -> str:
    if score >= threshold:
        return "positive"
    if score <= -threshold:
        return "negative"
    return "neutral"


def is_relevant_signal(score: float, threshold: float = DEFAULT_NEUTRAL_THRESHOLD) -> bool:
    return abs(score) >= threshold


def parse_timestamp(value: Any) -> datetime.datetime:
    if isinstance(value, datetime.datetime):
        dt_value = value
    elif not value:
        dt_value = timezone.now()
    else:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        if "T" not in text:
            text = f"{text}T12:00:00+03:00"
        dt_value = datetime.datetime.fromisoformat(text)

    if timezone.is_naive(dt_value):
        dt_value = timezone.make_aware(dt_value, timezone.get_current_timezone())
    return dt_value


def parse_date(value: Any) -> datetime.date:
    if isinstance(value, datetime.datetime):
        return as_local_date(value)
    if isinstance(value, datetime.date):
        return value
    if not value:
        return timezone.localdate()
    return datetime.date.fromisoformat(str(value))


def normalize_source_names(value: Any = None, default: str | Sequence[str] | None = None) -> list[str]:
    selected = value if value not in (None, "") else default
    if selected in (None, ""):
        return []
    if isinstance(selected, str):
        raw_names = selected.split(",")
    else:
        raw_names = list(selected)
    return [str(name).strip().lower() for name in raw_names if str(name).strip()]


def as_local_date(dt_value: datetime.datetime) -> datetime.date:
    if timezone.is_aware(dt_value):
        return timezone.localtime(dt_value).date()
    return dt_value.date()


def build_external_id(source: str, title: str, published_at: datetime.datetime, url: str, text: str) -> str:
    if url:
        return url
    digest = hashlib.sha1(f"{source}|{title}|{published_at.isoformat()}|{text}".encode("utf-8")).hexdigest()
    return f"{source}:{digest}"


def normalize_news_entry(entry: Any, source_override: Optional[str] = None) -> Dict[str, Any]:
    if isinstance(entry, str):
        title = entry[:160]
        summary = ""
        text = entry
        source = source_override or "tass"
        published_at = timezone.now()
        url = ""
        category = ""
        external_id = None
    else:
        title = (entry.get("title") or "").strip()
        summary = (entry.get("summary") or entry.get("lead") or "").strip()
        text = (entry.get("text") or "").strip()
        source = source_override or entry.get("source") or "tass"
        published_at = parse_timestamp(entry.get("published_at"))
        url = (entry.get("url") or "").strip()
        category = (entry.get("category") or "").strip()
        external_id = (entry.get("external_id") or "").strip() or None

    if not text:
        text = f"{title}: {summary}".strip(": ").strip() or title
    if not title:
        title = text[:160]

    return {
        "source": source,
        "title": title,
        "summary": summary,
        "text": text,
        "url": url,
        "category": category,
        "published_at": published_at,
        "external_id": external_id or build_external_id(source, title, published_at, url, text),
    }


def serialize_news_item(news_item, classification=None) -> Dict[str, Any]:
    payload = {
        "id": news_item.id,
        "source": news_item.source,
        "title": news_item.title,
        "summary": news_item.summary,
        "text": news_item.text,
        "url": news_item.url,
        "category": news_item.category,
        "published_at": timezone.localtime(news_item.published_at).isoformat()
        if timezone.is_aware(news_item.published_at)
        else news_item.published_at.isoformat(),
        "is_seed": news_item.is_seed,
    }
    if classification is not None:
        payload.update(
            {
                "sentiment_score": classification.sentiment_score,
                "sentiment_label": classification.sentiment_label,
                "positive_probability": classification.positive_probability,
                "negative_probability": classification.negative_probability,
                "confidence": classification.confidence,
                "is_relevant": classification.is_relevant,
                "relevance": classification.relevance,
                "pressure": classification.pressure,
                "engine": classification.engine,
                "evidence": classification.evidence,
                "reason": classification.reason,
            }
        )
    return payload


def ensure_factor_catalog():
    from analyzer.models import RiskFactor

    ordered_factors = []
    configured_keys = []
    for item in FACTOR_CONFIG:
        configured_keys.append(item["key"])
        factor, _ = RiskFactor.objects.update_or_create(
            key=item["key"],
            defaults={
                "name": item["name"],
                "description": item["description"],
                "positive_label": item["positive_label"],
                "negative_label": item["negative_label"],
                "weight": item["weight"],
                "display_order": item["display_order"],
                "is_active": True,
            },
        )
        ordered_factors.append(factor)

    RiskFactor.objects.exclude(key__in=configured_keys).update(is_active=False)
    return ordered_factors


def dedupe_news_entries(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique_entries = []
    seen = set()
    for entry in entries:
        key = (entry.get("source"), entry.get("external_id") or entry.get("url") or entry.get("title"))
        if key in seen:
            continue
        seen.add(key)
        unique_entries.append(entry)
    return unique_entries


def source_label_from_entries(entries: Iterable[Dict[str, Any]], prefix: str = "live") -> str:
    sources = sorted({str(entry.get("source") or "").strip() for entry in entries if entry.get("source")})
    suffix = ",".join(sources) if sources else "unknown"
    return f"{prefix}:{suffix}"[:64]


def get_news_entries(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    from news.sources import collect_news_entries

    try:
        entries = collect_news_entries(limit=limit)
    except Exception as exc:  # pragma: no cover - live network guard
        logger.warning("Failed to fetch live news: %s", exc)
        return []

    normalized = dedupe_news_entries(normalize_news_entry(entry) for entry in entries)
    normalized.sort(key=lambda item: item["published_at"], reverse=True)
    if limit:
        normalized = normalized[:limit]
    return normalized


def get_historical_news_entries(
    start_date: datetime.date,
    end_date: datetime.date,
    limit_per_day: int,
    source_names: Sequence[str] | None = None,
    delay_seconds: float = 0.0,
) -> List[Dict[str, Any]]:
    from news.sources import collect_historical_news_entries

    try:
        entries = collect_historical_news_entries(
            start_date=start_date,
            end_date=end_date,
            limit_per_day=limit_per_day,
            source_names=list(source_names) if source_names else None,
            delay_seconds=delay_seconds,
        )
    except Exception as exc:  # pragma: no cover - live network guard
        logger.warning("Failed to fetch historical news: %s", exc)
        return []

    normalized = dedupe_news_entries(normalize_news_entry(entry) for entry in entries)
    normalized.sort(key=lambda item: item["published_at"], reverse=True)
    return normalized


def store_news_entries(entries: Sequence[Dict[str, Any]], is_seed: bool = False):
    from analyzer.models import NewsItem

    stored_items = []
    stored_count = 0
    for entry in entries:
        news_item, created = NewsItem.objects.update_or_create(
            source=entry["source"],
            external_id=entry["external_id"],
            defaults={
                "title": entry["title"],
                "summary": entry["summary"],
                "text": entry["text"],
                "url": entry["url"],
                "category": entry["category"],
                "published_at": entry["published_at"],
                "is_seed": is_seed,
            },
        )
        stored_items.append(news_item)
        if created:
            stored_count += 1
    return stored_items, stored_count


def resolve_live_limit(limit: Optional[int] = None) -> int:
    return max(int(limit or DEFAULT_LIVE_NEWS_LIMIT), 1)


def resolve_history_backfill_limit(limit: Optional[int] = None) -> int:
    return max(int(limit or DEFAULT_HISTORY_BACKFILL_LIMIT), 1)


def load_seed_payload(seed_path: Optional[Path] = None) -> Dict[str, Any]:
    seed_file = Path(seed_path or SEED_DATA_PATH)
    if not seed_file.exists():
        raise FileNotFoundError(f"Seed file not found: {seed_file}")
    return json.loads(seed_file.read_text(encoding="utf-8"))


@transaction.atomic
def load_seed_dataset(
    seed_path: Optional[Path] = None,
    force: bool = False,
    clear_existing: bool = False,
) -> Dict[str, Any]:
    from analyzer.models import (
        DailyFactorSentiment,
        DailySentimentSummary,
        MonitoringBatch,
        NewsClassification,
        NewsItem,
    )

    ensure_factor_catalog()
    seed_payload = load_seed_payload(seed_path=seed_path)
    factor_map = {factor.key: factor for factor in ensure_factor_catalog()}

    has_seed_data = NewsItem.objects.filter(is_seed=True).exists()
    if has_seed_data and not force and not clear_existing:
        return {
            "skipped": True,
            "message": "Seed history is already loaded",
            "seed_path": str(seed_path or SEED_DATA_PATH),
        }

    if clear_existing:
        NewsClassification.objects.all().delete()
        NewsItem.objects.all().delete()
        DailyFactorSentiment.objects.all().delete()
        DailySentimentSummary.objects.all().delete()
    elif force:
        NewsClassification.objects.filter(is_seed=True).delete()
        NewsItem.objects.filter(is_seed=True).delete()

    batch = MonitoringBatch.objects.create(
        mode="seed" if not has_seed_data or clear_existing else "backfill",
        status="success",
        engine=ENGINE_BERT,
        model_name=BERT_MODEL_NAME,
        prompt_version=BERT_PROMPT_VERSION,
        factor_catalog_version=FACTOR_CATALOG_VERSION,
        source="seed",
        details={"seed_path": str(seed_path or SEED_DATA_PATH)},
    )

    created_news = 0
    affected_dates = set()
    classifications_count = 0

    for item in seed_payload.get("items", []):
        normalized = normalize_news_entry(item, source_override=item.get("source") or "seed")
        news_item, created = NewsItem.objects.update_or_create(
            source=normalized["source"],
            external_id=normalized["external_id"],
            defaults={
                "title": normalized["title"],
                "summary": normalized["summary"],
                "text": normalized["text"],
                "url": normalized["url"],
                "category": normalized["category"],
                "published_at": normalized["published_at"],
                "is_seed": True,
            },
        )
        if created:
            created_news += 1

        affected_dates.add(as_local_date(news_item.published_at))
        for payload in item.get("classifications", []):
            factor_key = normalize_factor_key(payload.get("factor_key"))
            factor = factor_map.get(factor_key)
            if not factor:
                logger.warning("Skipping unknown seed factor: %s", payload.get("factor_key"))
                continue
            positive_probability = float(payload.get("positive_probability", 0.5))
            negative_probability = float(payload.get("negative_probability", 0.5))
            sentiment_score = float(
                payload.get(
                    "sentiment_score",
                    compute_sentiment_score(positive_probability, negative_probability),
                )
            )
            sentiment_label = payload.get("sentiment_label") or label_from_score(sentiment_score)
            relevance = abs(sentiment_score)
            NewsClassification.objects.update_or_create(
                news_item=news_item,
                factor=factor,
                engine=ENGINE_BERT,
                model_name=BERT_MODEL_NAME,
                prompt_version=BERT_PROMPT_VERSION,
                factor_catalog_version=FACTOR_CATALOG_VERSION,
                defaults={
                    "batch": batch,
                    "positive_probability": positive_probability,
                    "negative_probability": negative_probability,
                    "sentiment_score": sentiment_score,
                    "relevance": relevance,
                    "pressure": max(-sentiment_score, 0.0),
                    "sentiment_label": sentiment_label,
                    "confidence": max(positive_probability, negative_probability),
                    "is_relevant": bool(payload.get("is_relevant", is_relevant_signal(sentiment_score))),
                    "is_seed": True,
                    "content_hash": news_content_hash(news_item),
                    "evidence": "",
                    "reason": "seed оценка",
                    "raw_response": payload,
                },
            )
            classifications_count += 1

    rebuild_result = rebuild_sentiment_history(affected_dates)
    batch.news_fetched = len(seed_payload.get("items", []))
    batch.news_stored = created_news
    batch.classifications_created = classifications_count
    batch.finished_at = timezone.now()
    batch.details.update({"affected_dates": rebuild_result["dates"]})
    batch.save(
        update_fields=[
            "news_fetched",
            "news_stored",
            "classifications_created",
            "finished_at",
            "details",
        ]
    )

    return {
        "skipped": False,
        "mode": batch.mode,
        "seed_path": str(seed_path or SEED_DATA_PATH),
        "news_created": created_news,
        "classifications_created": classifications_count,
        "dates": rebuild_result["dates"],
    }


def _matrix_width(matrix: Any) -> int:
    try:
        return len(matrix[0])
    except (IndexError, TypeError):
        return 0


def _factor_probabilities_slow(classifier: Any, texts: list[str], factors: Sequence[Any]) -> Dict[int, Any]:
    probabilities_by_factor = {}
    for factor in factors:
        probabilities_by_factor[factor.id] = classifier.predict_batch(
            texts,
            [factor.positive_label, factor.negative_label],
        )
    return probabilities_by_factor


def predict_factor_probabilities(classifier: Any, texts: list[str], factors: Sequence[Any]) -> Dict[int, Any]:
    labels = []
    label_pairs = {}
    for factor in factors:
        positive_index = len(labels)
        labels.extend([factor.positive_label, factor.negative_label])
        label_pairs[factor.id] = (positive_index, positive_index + 1)

    try:
        raw_matrix = classifier.predict_batch(texts, labels, normalize=False)
    except TypeError:
        return _factor_probabilities_slow(classifier, texts, factors)

    if _matrix_width(raw_matrix) < len(labels):
        return _factor_probabilities_slow(classifier, texts, factors)

    probabilities_by_factor = {}
    for factor in factors:
        positive_index, negative_index = label_pairs[factor.id]
        rows = []
        for row in raw_matrix:
            positive = float(row[positive_index])
            negative = float(row[negative_index])
            total = positive + negative
            if total <= 0:
                rows.append([0.5, 0.5])
            else:
                rows.append([positive / total, negative / total])
        probabilities_by_factor[factor.id] = rows
    return probabilities_by_factor


def classify_live_news_items(news_items, factors, batch=None, force: bool = False) -> Dict[str, Any]:
    from analyzer.models import NewsClassification

    if not news_items or not factors:
        return {
            "classified_count": 0,
            "affected_dates": [],
            "errors": [],
            "warnings": [],
            "cached_count": 0,
        }

    identity = get_engine_identity(ENGINE_BERT)
    factor_count = len(factors)
    pending_items = []
    if force:
        pending_items = list(news_items)
    else:
        for item in news_items:
            current_count = NewsClassification.objects.filter(
                news_item=item,
                factor__in=factors,
                content_hash=news_content_hash(item),
                **identity,
            ).count()
            if current_count < factor_count:
                pending_items.append(item)

    if not pending_items:
        return {
            "classified_count": 0,
            "affected_dates": sorted({as_local_date(item.published_at) for item in news_items}),
            "errors": [],
            "warnings": [],
            "cached_count": len(news_items),
        }

    classifier = get_classifier()
    texts = [item.text for item in pending_items]
    probabilities_by_factor = predict_factor_probabilities(classifier, texts, factors)
    classified_count = 0
    affected_dates = {as_local_date(item.published_at) for item in pending_items}

    for factor in factors:
        probabilities = probabilities_by_factor[factor.id]
        for index, news_item in enumerate(pending_items):
            positive_probability = float(probabilities[index][0])
            negative_probability = float(probabilities[index][1])
            sentiment_score = compute_sentiment_score(positive_probability, negative_probability)
            NewsClassification.objects.update_or_create(
                news_item=news_item,
                factor=factor,
                **identity,
                defaults={
                    "batch": batch,
                    "positive_probability": positive_probability,
                    "negative_probability": negative_probability,
                    "sentiment_score": sentiment_score,
                    "relevance": abs(sentiment_score),
                    "pressure": max(-sentiment_score, 0.0),
                    "sentiment_label": label_from_score(sentiment_score),
                    "confidence": max(positive_probability, negative_probability),
                    "is_relevant": is_relevant_signal(sentiment_score),
                    "is_seed": False,
                    "content_hash": news_content_hash(news_item),
                    "evidence": "",
                    "reason": "bert zero-shot",
                    "raw_response": {
                        "labels": [factor.positive_label, factor.negative_label],
                        "probabilities": [positive_probability, negative_probability],
                    },
                },
            )
            classified_count += 1

    return {
        "classified_count": classified_count,
        "affected_dates": sorted(affected_dates),
        "errors": [],
        "warnings": [],
        "cached_count": len(news_items) - len(pending_items),
    }


def recompute_factor_deltas(
    factor_ids: Optional[Iterable[int]] = None,
    identity: Optional[Dict[str, str]] = None,
) -> None:
    from analyzer.models import DailyFactorSentiment

    queryset = DailyFactorSentiment.objects.all().order_by("factor_id", "date")
    if identity:
        queryset = queryset.filter(**identity)
    if factor_ids:
        queryset = queryset.filter(factor_id__in=list(factor_ids))

    updates = []
    previous_by_factor = {}
    for summary in queryset:
        previous = previous_by_factor.get(summary.factor_id)
        summary.delta_from_previous = (
            summary.average_sentiment - previous.average_sentiment if previous else 0.0
        )
        previous_by_factor[summary.factor_id] = summary
        updates.append(summary)

    if updates:
        DailyFactorSentiment.objects.bulk_update(updates, ["delta_from_previous"])


def recompute_daily_deltas(identity: Optional[Dict[str, str]] = None) -> None:
    from analyzer.models import DailySentimentSummary

    queryset = DailySentimentSummary.objects.all().order_by("date")
    if identity:
        queryset = queryset.filter(**identity)

    updates = []
    previous = None
    for summary in queryset:
        summary.delta_from_previous = summary.average_sentiment - previous.average_sentiment if previous else 0.0
        previous = summary
        updates.append(summary)

    if updates:
        DailySentimentSummary.objects.bulk_update(updates, ["delta_from_previous"])


def normalize_dates(dates: Optional[Iterable[Any]]) -> List[datetime.date]:
    normalized = set()
    for value in dates or []:
        if isinstance(value, datetime.datetime):
            normalized.add(as_local_date(value))
        elif isinstance(value, datetime.date):
            normalized.add(value)
        elif value:
            normalized.add(datetime.date.fromisoformat(str(value)))
    return sorted(normalized)


@transaction.atomic
def rebuild_sentiment_history(
    dates: Optional[Iterable[Any]] = None,
    engine: str = ENGINE_BERT,
) -> Dict[str, Any]:
    from analyzer.models import DailyFactorSentiment, DailySentimentSummary, NewsClassification, RiskFactor

    identity = get_engine_identity(engine)
    normalized_dates = normalize_dates(dates)
    queryset = NewsClassification.objects.select_related("factor", "news_item").filter(**identity)
    if normalized_dates:
        queryset = queryset.filter(news_item__published_at__date__in=normalized_dates)

    classifications = list(queryset.order_by("news_item__published_at", "factor__display_order"))
    effective_dates = normalized_dates or sorted({as_local_date(item.news_item.published_at) for item in classifications})

    if not effective_dates:
        return {"dates": [], "daily_records": 0, "factor_records": 0}

    DailyFactorSentiment.objects.filter(date__in=effective_dates, **identity).delete()
    DailySentimentSummary.objects.filter(date__in=effective_dates, **identity).delete()

    grouped = defaultdict(list)
    unique_news_by_date = defaultdict(set)
    for classification in classifications:
        day = as_local_date(classification.news_item.published_at)
        grouped[(day, classification.factor_id)].append(classification)
        unique_news_by_date[day].add(classification.news_item_id)

    active_factors = list(RiskFactor.objects.filter(is_active=True).order_by("display_order", "name"))
    factor_records = []
    factor_ids = {factor.id for factor in active_factors}
    grouped_by_date = defaultdict(list)
    for day in effective_dates:
        for factor in active_factors:
            items = grouped.get((day, factor.id), [])
            relevant_items = [item for item in items if item.is_relevant]
            positive_items = [item for item in relevant_items if item.sentiment_label == "positive"]
            negative_items = [item for item in relevant_items if item.sentiment_label == "negative"]
            neutral_items = [item for item in items if item.sentiment_label == "neutral"]
            average_sentiment = (
                sum(item.sentiment_score for item in relevant_items) / len(relevant_items) if relevant_items else 0.0
            )
            top_positive = max(positive_items, key=lambda item: item.sentiment_score) if positive_items else None
            top_negative = min(negative_items, key=lambda item: item.sentiment_score) if negative_items else None

            record = DailyFactorSentiment(
                factor=factor,
                date=day,
                **identity,
                average_sentiment=average_sentiment,
                total_news_count=len({item.news_item_id for item in items}),
                relevant_news_count=len({item.news_item_id for item in relevant_items}),
                positive_hits=len(positive_items),
                negative_hits=len(negative_items),
                neutral_hits=len(neutral_items),
                max_positive_score=top_positive.sentiment_score if top_positive else 0.0,
                max_negative_score=top_negative.sentiment_score if top_negative else 0.0,
                top_positive_news=top_positive.news_item if top_positive else None,
                top_negative_news=top_negative.news_item if top_negative else None,
            )
            factor_records.append(record)
            grouped_by_date[day].append(record)

    DailyFactorSentiment.objects.bulk_create(factor_records)

    daily_records = []
    for day in sorted(grouped_by_date):
        items = grouped_by_date[day]
        total_weight = sum(item.factor.weight for item in items) or 1.0
        average_sentiment = sum(item.average_sentiment for item in items) / len(items)
        risk_index = sum(max(-item.average_sentiment, 0.0) * item.factor.weight for item in items) / total_weight
        daily_records.append(
            DailySentimentSummary(
                date=day,
                **identity,
                average_sentiment=average_sentiment,
                risk_index=risk_index,
                news_count=len(unique_news_by_date.get(day, set())),
                factor_count=len(items),
            )
        )

    DailySentimentSummary.objects.bulk_create(daily_records)
    recompute_factor_deltas(factor_ids, identity=identity)
    recompute_daily_deltas(identity=identity)

    return {
        "dates": [item.isoformat() for item in sorted(effective_dates)],
        "daily_records": len(daily_records),
        "factor_records": len(factor_records),
    }


def get_news_classification(factor, news_item, engine: str = ENGINE_BERT):
    from analyzer.models import NewsClassification

    identity = get_engine_identity(engine)
    return (
        NewsClassification.objects.filter(factor=factor, news_item=news_item, **identity)
        .select_related("news_item", "factor")
        .first()
    )


def build_spike_payload(summary, engine: str = ENGINE_BERT) -> Dict[str, Any]:
    spotlight = summary.top_positive_news if summary.delta_from_previous >= 0 else summary.top_negative_news
    if spotlight is None:
        spotlight = summary.top_negative_news or summary.top_positive_news

    classification = get_news_classification(summary.factor, spotlight, engine=engine) if spotlight else None
    return {
        "date": summary.date.isoformat(),
        "factor_key": summary.factor.key,
        "factor_name": summary.factor.name,
        "delta_from_previous": summary.delta_from_previous,
        "average_sentiment": summary.average_sentiment,
        "relevant_news_count": summary.relevant_news_count,
        "spotlight_news": serialize_news_item(spotlight, classification) if spotlight and classification else None,
    }


def get_factor_detail(
    factor_key: str,
    days: int = DEFAULT_WINDOW_DAYS,
    news_limit: int = 20,
    engine: str = ENGINE_BERT,
) -> Dict[str, Any]:
    from analyzer.models import DailyFactorSentiment, NewsClassification, RiskFactor

    identity = get_engine_identity(engine)
    factor_key = normalize_factor_key(factor_key)
    factor = RiskFactor.objects.filter(key=factor_key, is_active=True).first()
    if not factor:
        return {}

    cutoff = timezone.localdate() - datetime.timedelta(days=max(days - 1, 0))
    summaries = list(factor.daily_summaries.filter(date__gte=cutoff, **identity).order_by("date"))
    recent_classifications = list(
        NewsClassification.objects.select_related("news_item")
        .filter(factor=factor, is_relevant=True, **identity)
        .order_by("-news_item__published_at")[:news_limit]
    )
    spike_records = sorted(summaries, key=lambda item: abs(item.delta_from_previous), reverse=True)[:8]
    latest = summaries[-1] if summaries else None

    return {
        "key": factor.key,
        "name": factor.name,
        "description": factor.description,
        "positive_label": factor.positive_label,
        "negative_label": factor.negative_label,
        "weight": factor.weight,
        "latest": {
            "date": latest.date.isoformat(),
            "average_sentiment": latest.average_sentiment,
            "delta_from_previous": latest.delta_from_previous,
            "relevant_news_count": latest.relevant_news_count,
            "total_news_count": latest.total_news_count,
            "positive_hits": latest.positive_hits,
            "negative_hits": latest.negative_hits,
            "neutral_hits": latest.neutral_hits,
        }
        if latest
        else {},
        "timeline": [
            {
                "date": item.date.isoformat(),
                "average_sentiment": item.average_sentiment,
                "delta_from_previous": item.delta_from_previous,
                "relevant_news_count": item.relevant_news_count,
                "total_news_count": item.total_news_count,
                "positive_hits": item.positive_hits,
                "negative_hits": item.negative_hits,
                "neutral_hits": item.neutral_hits,
            }
            for item in summaries
        ],
        "recent_news": [serialize_news_item(item.news_item, item) for item in recent_classifications],
        "spikes": [build_spike_payload(item, engine=engine) for item in spike_records],
    }


def get_monitoring_feed(
    days: int = DEFAULT_WINDOW_DAYS,
    factor_key: Optional[str] = None,
    engine: str = ENGINE_BERT,
) -> Dict[str, Any]:
    from analyzer.models import DailyFactorSentiment, DailySentimentSummary, MonitoringBatch, RiskFactor

    engine = normalize_engine(engine)
    identity = get_engine_identity(engine)
    cutoff = timezone.localdate() - datetime.timedelta(days=max(days - 1, 0))
    timeline_records = list(DailySentimentSummary.objects.filter(date__gte=cutoff, **identity).order_by("date"))
    active_factors = list(RiskFactor.objects.filter(is_active=True).order_by("display_order", "name"))

    factors_payload = []
    for factor in active_factors:
        latest = factor.daily_summaries.filter(**identity).order_by("-date").first()
        factors_payload.append(
            {
                "key": factor.key,
                "name": factor.name,
                "description": factor.description,
                "weight": factor.weight,
                "latest_sentiment": latest.average_sentiment if latest else 0.0,
                "delta_from_previous": latest.delta_from_previous if latest else 0.0,
                "relevant_news_count": latest.relevant_news_count if latest else 0,
                "total_news_count": latest.total_news_count if latest else 0,
                "positive_hits": latest.positive_hits if latest else 0,
                "negative_hits": latest.negative_hits if latest else 0,
                "neutral_hits": latest.neutral_hits if latest else 0,
                "last_date": latest.date.isoformat() if latest else None,
            }
        )

    spike_candidates = list(
        DailyFactorSentiment.objects.select_related("factor", "top_positive_news", "top_negative_news")
        .filter(date__gte=cutoff, **identity)
    )
    spikes = [
        build_spike_payload(item, engine=engine)
        for item in sorted(spike_candidates, key=lambda row: abs(row.delta_from_previous), reverse=True)[:10]
    ]

    default_factor_key = next(
        (item["key"] for item in factors_payload if item["relevant_news_count"] > 0),
        active_factors[0].key if active_factors else None,
    )
    selected_factor_key = normalize_factor_key(factor_key) or default_factor_key
    last_batch = MonitoringBatch.objects.filter(engine=engine, finished_at__isnull=False).order_by(
        "-finished_at",
        "-started_at",
    ).first()
    latest_record = timeline_records[-1] if timeline_records else None

    return {
        "overview": {
            "latest_date": latest_record.date.isoformat() if latest_record else None,
            "average_sentiment": latest_record.average_sentiment if latest_record else 0.0,
            "risk_index": latest_record.risk_index if latest_record else 0.0,
            "news_count": latest_record.news_count if latest_record else 0,
            "factor_count": latest_record.factor_count if latest_record else len(active_factors),
            "window_news_count": sum(item.news_count for item in timeline_records),
            "last_run": {
                "mode": last_batch.mode,
                "status": last_batch.status,
                "engine": last_batch.engine,
                "model_name": last_batch.model_name,
                "finished_at": timezone.localtime(last_batch.finished_at).isoformat()
                if last_batch and last_batch.finished_at
                else None,
                "news_fetched": last_batch.news_fetched,
                "news_stored": last_batch.news_stored,
                "classifications_created": last_batch.classifications_created,
            }
            if last_batch
            else {},
        },
        "timeline": [
            {
                "date": item.date.isoformat(),
                "average_sentiment": item.average_sentiment,
                "delta_from_previous": item.delta_from_previous,
                "risk_index": item.risk_index,
                "news_count": item.news_count,
                "factor_count": item.factor_count,
            }
            for item in timeline_records
        ],
        "factors": factors_payload,
        "spikes": spikes,
        "factor_detail": get_factor_detail(selected_factor_key, days=days, engine=engine) if selected_factor_key else {},
        "engine": engine,
        "model_name": identity["model_name"],
        "prompt_version": identity["prompt_version"],
    }


def run_monitoring_pipeline(
    news_entries: Optional[Sequence[Dict[str, Any]]] = None,
    force: bool = False,
    bootstrap_if_empty: bool = False,
    limit: Optional[int] = None,
    engine: str = ENGINE_BERT,
    summarize: bool = False,
    period: str = "day",
) -> Dict[str, Any]:
    from analyzer.models import MonitoringBatch, NewsItem

    engine = normalize_engine(engine)
    identity = get_engine_identity(engine)
    factors = ensure_factor_catalog()
    bootstrap_result = None
    if bootstrap_if_empty and not NewsItem.objects.exists():
        bootstrap_result = load_seed_dataset(force=False, clear_existing=False)

    batch = MonitoringBatch.objects.create(mode="update", status="success", source="live", **identity)
    try:
        effective_limit = resolve_live_limit(limit)
        entries = list(news_entries) if news_entries is not None else get_news_entries(limit=effective_limit)
        entries = entries[:effective_limit]

        normalized_entries = [normalize_news_entry(entry) for entry in entries]
        if not normalized_entries:
            batch.status = "skipped"
            batch.finished_at = timezone.now()
            batch.details = {"reason": "No live news entries returned"}
            batch.save(update_fields=["status", "finished_at", "details"])
            return {
                "skipped": True,
                "message": "No live news entries returned",
                "engine": engine,
                "bootstrap": bootstrap_result,
            }

        stored_items, stored_count = store_news_entries(normalized_entries, is_seed=False)

        classification_result = get_monitoring_analyzer(engine).classify(
            stored_items,
            factors,
            batch=batch,
            force=force,
        ).as_dict()

        rebuild_result = rebuild_sentiment_history(classification_result["affected_dates"], engine=engine)
        summary_result = None
        summary_errors = []
        if summarize and engine == ENGINE_LLM:
            from analyzer.llm_services import generate_llm_summaries

            try:
                summary_result = generate_llm_summaries(period=period, force=force)
                summary_errors = summary_result.get("errors", [])
                summary_warnings = summary_result.get("warnings", [])
            except Exception as exc:
                logger.exception("LLM summary generation failed")
                summary_errors = [str(exc)]
                summary_warnings = []

        source_label = source_label_from_entries(normalized_entries)
        batch.source = source_label
        batch.news_fetched = len(normalized_entries)
        batch.news_stored = stored_count
        batch.classifications_created = classification_result["classified_count"]
        batch.finished_at = timezone.now()
        batch.details = {
            "affected_dates": rebuild_result["dates"],
            "force": force,
            "engine": engine,
            "cached_count": classification_result.get("cached_count", 0),
            "limit": effective_limit,
            "classification_errors": classification_result.get("errors", []),
            "classification_warnings": classification_result.get("warnings", []),
            "summary": summary_result,
            "summary_errors": summary_errors,
            "summary_warnings": summary_warnings if summarize and engine == ENGINE_LLM else [],
            "sources": sorted({entry["source"] for entry in normalized_entries}),
        }
        if classification_result.get("errors") or summary_errors:
            batch.status = "partial_failed"
        elif stored_count == 0 and classification_result["classified_count"] == 0:
            batch.status = "skipped"
        batch.save(
            update_fields=[
                "news_fetched",
                "news_stored",
                "classifications_created",
                "finished_at",
                "status",
                "source",
                "details",
            ]
        )

        feed = get_monitoring_feed(days=DEFAULT_WINDOW_DAYS, engine=engine)
        return {
            "skipped": batch.status == "skipped",
            "status": batch.status,
            "mode": "update",
            "engine": engine,
            "model_name": identity["model_name"],
            "news_fetched": len(normalized_entries),
            "news_stored": stored_count,
            "classifications_created": classification_result["classified_count"],
            "dates": rebuild_result["dates"],
            "bootstrap": bootstrap_result,
            "summary": summary_result,
            "sources": sorted({entry["source"] for entry in normalized_entries}),
            "errors": classification_result.get("errors", []) + summary_errors,
            "warnings": classification_result.get("warnings", []),
            "overview": feed["overview"],
        }
    except Exception as exc:
        batch.status = "failed"
        batch.finished_at = timezone.now()
        batch.details = {"error": str(exc)}
        batch.save(update_fields=["status", "finished_at", "details"])
        logger.exception("Monitoring update failed")
        raise


def backfill_news_history(
    start_date: Any = None,
    end_date: Any = None,
    days: Optional[int] = None,
    source_names: Sequence[str] | str | None = None,
    limit_per_day: Optional[int] = None,
    total_limit: Optional[int] = None,
    delay_seconds: float = 0.0,
    engine: str = ENGINE_BERT,
    force: bool = False,
    summarize: bool = False,
    period: str = "day",
) -> Dict[str, Any]:
    from analyzer.models import MonitoringBatch

    engine = normalize_engine(engine)
    identity = get_engine_identity(engine)
    factors = ensure_factor_catalog()

    effective_end = parse_date(end_date) if end_date else timezone.localdate()
    effective_days = max(int(days or DEFAULT_HISTORY_DAYS), 1)
    effective_start = parse_date(start_date) if start_date else effective_end - datetime.timedelta(days=effective_days - 1)
    if effective_start > effective_end:
        effective_start, effective_end = effective_end, effective_start

    effective_limit_per_day = max(int(limit_per_day or DEFAULT_HISTORY_LIMIT_PER_DAY), 1)
    selected_sources = normalize_source_names(source_names, default=DEFAULT_HISTORY_SOURCES)
    batch_source = f"history:{','.join(selected_sources) if selected_sources else 'configured'}"[:64]
    batch = MonitoringBatch.objects.create(mode="backfill", status="success", source=batch_source, **identity)

    try:
        normalized_entries = get_historical_news_entries(
            start_date=effective_start,
            end_date=effective_end,
            limit_per_day=effective_limit_per_day,
            source_names=selected_sources,
            delay_seconds=max(float(delay_seconds or 0), 0.0),
        )
        if total_limit:
            normalized_entries = normalized_entries[: max(int(total_limit), 1)]

        if not normalized_entries:
            batch.status = "skipped"
            batch.finished_at = timezone.now()
            batch.details = {
                "reason": "No historical news entries returned",
                "engine": engine,
                "start_date": effective_start.isoformat(),
                "end_date": effective_end.isoformat(),
                "sources": selected_sources,
                "limit_per_day": effective_limit_per_day,
            }
            batch.save(update_fields=["status", "finished_at", "details"])
            return {
                "skipped": True,
                "message": "No historical news entries returned",
                "engine": engine,
                "start_date": effective_start.isoformat(),
                "end_date": effective_end.isoformat(),
                "sources": selected_sources,
            }

        stored_items, stored_count = store_news_entries(normalized_entries, is_seed=False)
        classification_result = get_monitoring_analyzer(engine).classify(
            stored_items,
            factors,
            batch=batch,
            force=force,
        ).as_dict()

        rebuild_result = rebuild_sentiment_history(classification_result["affected_dates"], engine=engine)
        summary_result = None
        summary_errors = []
        summary_warnings = []
        if summarize and engine == ENGINE_LLM:
            from analyzer.llm_services import generate_llm_summaries

            try:
                summary_result = generate_llm_summaries(period=period, force=force)
                summary_errors = summary_result.get("errors", [])
                summary_warnings = summary_result.get("warnings", [])
            except Exception as exc:
                logger.exception("LLM summary generation failed")
                summary_errors = [str(exc)]

        batch.source = source_label_from_entries(normalized_entries, prefix="history")
        batch.news_fetched = len(normalized_entries)
        batch.news_stored = stored_count
        batch.classifications_created = classification_result["classified_count"]
        batch.finished_at = timezone.now()
        batch.details = {
            "affected_dates": rebuild_result["dates"],
            "force": force,
            "engine": engine,
            "start_date": effective_start.isoformat(),
            "end_date": effective_end.isoformat(),
            "sources": sorted({entry["source"] for entry in normalized_entries}),
            "requested_sources": selected_sources,
            "limit_per_day": effective_limit_per_day,
            "total_limit": total_limit,
            "cached_count": classification_result.get("cached_count", 0),
            "classification_errors": classification_result.get("errors", []),
            "classification_warnings": classification_result.get("warnings", []),
            "summary": summary_result,
            "summary_errors": summary_errors,
            "summary_warnings": summary_warnings if summarize and engine == ENGINE_LLM else [],
        }
        if classification_result.get("errors") or summary_errors:
            batch.status = "partial_failed"
        elif stored_count == 0 and classification_result["classified_count"] == 0:
            batch.status = "skipped"
        batch.save(
            update_fields=[
                "news_fetched",
                "news_stored",
                "classifications_created",
                "finished_at",
                "status",
                "source",
                "details",
            ]
        )

        feed = get_monitoring_feed(days=DEFAULT_WINDOW_DAYS, engine=engine)
        return {
            "skipped": batch.status == "skipped",
            "status": batch.status,
            "mode": "history_backfill",
            "engine": engine,
            "model_name": identity["model_name"],
            "start_date": effective_start.isoformat(),
            "end_date": effective_end.isoformat(),
            "sources": sorted({entry["source"] for entry in normalized_entries}),
            "requested_sources": selected_sources,
            "news_fetched": len(normalized_entries),
            "news_stored": stored_count,
            "cached_count": classification_result.get("cached_count", 0),
            "classifications_created": classification_result["classified_count"],
            "dates": rebuild_result["dates"],
            "summary": summary_result,
            "errors": classification_result.get("errors", []) + summary_errors,
            "warnings": classification_result.get("warnings", []),
            "overview": feed["overview"],
        }
    except Exception as exc:
        batch.status = "failed"
        batch.finished_at = timezone.now()
        batch.details = {"error": str(exc)}
        batch.save(update_fields=["status", "finished_at", "details"])
        logger.exception("Historical monitoring backfill failed")
        raise


def backfill_existing_news(
    engine: str = ENGINE_BERT,
    force: bool = False,
    limit: Optional[int] = None,
    summarize: bool = False,
    period: str = "day",
    source: Optional[str] = DEFAULT_BACKFILL_SOURCE,
    include_seed: Optional[bool] = None,
) -> Dict[str, Any]:
    from analyzer.models import MonitoringBatch, NewsItem

    engine = normalize_engine(engine)
    identity = get_engine_identity(engine)
    factors = ensure_factor_catalog()
    source = (source or "").strip() or None
    include_seed = (not EXCLUDE_SEED_BY_DEFAULT) if include_seed is None else include_seed
    effective_limit = resolve_history_backfill_limit(limit)
    queryset = NewsItem.objects.all()
    if source:
        queryset = queryset.filter(source=source)
    if not include_seed:
        queryset = queryset.filter(is_seed=False)
    queryset = queryset.order_by("-published_at", "-id")[:effective_limit]
    news_items = list(queryset)

    batch_source = f"stored:{source or 'all'}"
    batch = MonitoringBatch.objects.create(mode="backfill", status="success", source=batch_source, **identity)
    if not news_items:
        batch.status = "skipped"
        batch.finished_at = timezone.now()
        batch.details = {
            "reason": "No stored news for backfill",
            "engine": engine,
            "source": source,
            "include_seed": include_seed,
            "limit": effective_limit,
        }
        batch.save(update_fields=["status", "finished_at", "details"])
        return {"skipped": True, "message": "No stored news for backfill", "engine": engine}

    classification_result = get_monitoring_analyzer(engine).classify(
        news_items,
        factors,
        batch=batch,
        force=force,
    ).as_dict()

    rebuild_result = rebuild_sentiment_history(classification_result["affected_dates"], engine=engine)
    summary_result = None
    summary_errors = []
    if summarize and engine == ENGINE_LLM:
        from analyzer.llm_services import generate_llm_summaries

        try:
            summary_result = generate_llm_summaries(period=period, force=force)
            summary_errors = summary_result.get("errors", [])
            summary_warnings = summary_result.get("warnings", [])
        except Exception as exc:
            logger.exception("LLM summary generation failed")
            summary_errors = [str(exc)]
            summary_warnings = []

    batch.news_fetched = len(news_items)
    batch.news_stored = 0
    batch.classifications_created = classification_result["classified_count"]
    batch.finished_at = timezone.now()
    batch.details = {
        "affected_dates": rebuild_result["dates"],
        "force": force,
        "engine": engine,
        "source": source,
        "include_seed": include_seed,
        "limit": effective_limit,
        "cached_count": classification_result.get("cached_count", 0),
        "classification_errors": classification_result.get("errors", []),
        "classification_warnings": classification_result.get("warnings", []),
        "summary": summary_result,
        "summary_errors": summary_errors,
        "summary_warnings": summary_warnings if summarize and engine == ENGINE_LLM else [],
    }
    if classification_result.get("errors") or summary_errors:
        batch.status = "partial_failed"
    elif classification_result["classified_count"] == 0 and classification_result.get("cached_count", 0) == 0:
        batch.status = "skipped"
    batch.save(
        update_fields=[
            "news_fetched",
            "news_stored",
            "classifications_created",
            "finished_at",
            "status",
            "details",
        ]
    )

    return {
        "skipped": batch.status == "skipped",
        "status": batch.status,
        "mode": "backfill",
        "engine": engine,
        "model_name": identity["model_name"],
        "news_count": len(news_items),
        "limit": effective_limit,
        "source": source,
        "include_seed": include_seed,
        "cached_count": classification_result.get("cached_count", 0),
        "classifications_created": classification_result["classified_count"],
        "dates": rebuild_result["dates"],
        "summary": summary_result,
        "errors": classification_result.get("errors", []) + summary_errors,
        "warnings": classification_result.get("warnings", []),
    }
