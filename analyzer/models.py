from django.db import models
from django.utils import timezone

from analyzer.constants import (
    BERT_MODEL_NAME,
    BERT_PROMPT_VERSION,
    ENGINE_BERT,
    ENGINE_CHOICES,
    ENGINE_LLM,
    FACTOR_CATALOG_VERSION,
)


class RiskFactor(models.Model):
    key = models.SlugField(max_length=50, unique=True)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    positive_label = models.CharField(max_length=255)
    negative_label = models.CharField(max_length=255)
    weight = models.FloatField(default=1.0)
    display_order = models.PositiveSmallIntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["display_order", "name"]

    def __str__(self) -> str:
        return self.name


class MonitoringBatch(models.Model):
    MODE_CHOICES = [
        ("seed", "Seed"),
        ("backfill", "Backfill"),
        ("update", "Update"),
    ]
    STATUS_CHOICES = [
        ("success", "Success"),
        ("skipped", "Skipped"),
        ("failed", "Failed"),
        ("partial_failed", "Partial failed"),
    ]

    mode = models.CharField(max_length=20, choices=MODE_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="success")
    engine = models.CharField(max_length=16, choices=ENGINE_CHOICES, default=ENGINE_BERT)
    model_name = models.CharField(max_length=255, default=BERT_MODEL_NAME)
    prompt_version = models.CharField(max_length=80, default=BERT_PROMPT_VERSION)
    factor_catalog_version = models.CharField(max_length=80, default=FACTOR_CATALOG_VERSION)
    source = models.CharField(max_length=64, default="tass")
    started_at = models.DateTimeField(default=timezone.now)
    finished_at = models.DateTimeField(null=True, blank=True)
    news_fetched = models.PositiveIntegerField(default=0)
    news_stored = models.PositiveIntegerField(default=0)
    classifications_created = models.PositiveIntegerField(default=0)
    details = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self) -> str:
        return f"{self.mode}:{self.status}:{self.started_at:%Y-%m-%d %H:%M}"


class NewsItem(models.Model):
    source = models.CharField(max_length=64, default="tass")
    external_id = models.CharField(max_length=512)
    url = models.URLField(max_length=500, blank=True)
    title = models.CharField(max_length=500)
    summary = models.TextField(blank=True)
    text = models.TextField()
    category = models.CharField(max_length=255, blank=True)
    published_at = models.DateTimeField(db_index=True)
    is_seed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [("source", "external_id")]
        indexes = [
            models.Index(fields=["source", "-published_at"], name="news_item_source_date_idx"),
            models.Index(fields=["is_seed", "-published_at"], name="news_item_seed_date_idx"),
        ]
        ordering = ["-published_at", "-id"]

    def __str__(self) -> str:
        return self.title


class NewsClassification(models.Model):
    SENTIMENT_CHOICES = [
        ("positive", "Positive"),
        ("neutral", "Neutral"),
        ("negative", "Negative"),
    ]

    news_item = models.ForeignKey(NewsItem, on_delete=models.CASCADE, related_name="classifications")
    factor = models.ForeignKey(RiskFactor, on_delete=models.CASCADE, related_name="classifications")
    batch = models.ForeignKey(
        MonitoringBatch,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="classifications",
    )
    positive_probability = models.FloatField()
    negative_probability = models.FloatField()
    sentiment_score = models.FloatField()
    relevance = models.FloatField(default=0)
    pressure = models.FloatField(default=0)
    sentiment_label = models.CharField(max_length=16, choices=SENTIMENT_CHOICES, default="neutral")
    confidence = models.FloatField(default=0)
    evidence = models.CharField(max_length=255, blank=True)
    reason = models.CharField(max_length=255, blank=True)
    is_relevant = models.BooleanField(default=False)
    is_seed = models.BooleanField(default=False)
    engine = models.CharField(max_length=16, choices=ENGINE_CHOICES, default=ENGINE_BERT)
    model_name = models.CharField(max_length=255, default=BERT_MODEL_NAME)
    prompt_version = models.CharField(max_length=80, default=BERT_PROMPT_VERSION)
    factor_catalog_version = models.CharField(max_length=80, default=FACTOR_CATALOG_VERSION)
    content_hash = models.CharField(max_length=64, blank=True)
    raw_response = models.JSONField(default=dict, blank=True)
    classified_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [
            (
                "news_item",
                "factor",
                "engine",
                "model_name",
                "prompt_version",
                "factor_catalog_version",
            )
        ]
        indexes = [
            models.Index(fields=["engine", "factor", "is_relevant"], name="news_class_factor_idx"),
            models.Index(fields=["engine", "classified_at"], name="news_class_engine_time_idx"),
        ]
        ordering = ["-news_item__published_at", "factor__display_order"]

    def __str__(self) -> str:
        return f"{self.factor.name}: {self.news_item.title}"


class DailySentimentSummary(models.Model):
    date = models.DateField()
    engine = models.CharField(max_length=16, choices=ENGINE_CHOICES, default=ENGINE_BERT)
    model_name = models.CharField(max_length=255, default=BERT_MODEL_NAME)
    prompt_version = models.CharField(max_length=80, default=BERT_PROMPT_VERSION)
    factor_catalog_version = models.CharField(max_length=80, default=FACTOR_CATALOG_VERSION)
    average_sentiment = models.FloatField(default=0)
    delta_from_previous = models.FloatField(default=0)
    risk_index = models.FloatField(default=0)
    news_count = models.PositiveIntegerField(default=0)
    factor_count = models.PositiveIntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [("date", "engine", "model_name", "prompt_version", "factor_catalog_version")]
        indexes = [
            models.Index(fields=["engine", "-date"], name="daily_summary_engine_date_idx"),
        ]
        ordering = ["-date"]

    def __str__(self) -> str:
        return f"Daily sentiment {self.date}"


class DailyFactorSentiment(models.Model):
    factor = models.ForeignKey(RiskFactor, on_delete=models.CASCADE, related_name="daily_summaries")
    date = models.DateField()
    engine = models.CharField(max_length=16, choices=ENGINE_CHOICES, default=ENGINE_BERT)
    model_name = models.CharField(max_length=255, default=BERT_MODEL_NAME)
    prompt_version = models.CharField(max_length=80, default=BERT_PROMPT_VERSION)
    factor_catalog_version = models.CharField(max_length=80, default=FACTOR_CATALOG_VERSION)
    average_sentiment = models.FloatField(default=0)
    delta_from_previous = models.FloatField(default=0)
    total_news_count = models.PositiveIntegerField(default=0)
    relevant_news_count = models.PositiveIntegerField(default=0)
    positive_hits = models.PositiveIntegerField(default=0)
    negative_hits = models.PositiveIntegerField(default=0)
    neutral_hits = models.PositiveIntegerField(default=0)
    max_positive_score = models.FloatField(default=0)
    max_negative_score = models.FloatField(default=0)
    top_positive_news = models.ForeignKey(
        NewsItem,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
    )
    top_negative_news = models.ForeignKey(
        NewsItem,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
    )
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [("factor", "date", "engine", "model_name", "prompt_version", "factor_catalog_version")]
        indexes = [
            models.Index(fields=["engine", "factor", "-date"], name="factor_sent_engine_date_idx"),
        ]
        ordering = ["-date", "factor__display_order"]

    def __str__(self) -> str:
        return f"{self.factor.name} {self.date}"


class MonitoringSummary(models.Model):
    TREND_CHOICES = [
        ("improving", "Improving"),
        ("worsening", "Worsening"),
        ("stable", "Stable"),
        ("mixed", "Mixed"),
    ]
    RISK_CHOICES = [
        ("low", "Low"),
        ("medium", "Medium"),
        ("high", "High"),
    ]

    factor = models.ForeignKey(RiskFactor, on_delete=models.CASCADE, related_name="llm_summaries")
    period_start = models.DateField()
    period_end = models.DateField()
    engine = models.CharField(max_length=16, choices=ENGINE_CHOICES, default=ENGINE_LLM)
    model_name = models.CharField(max_length=255)
    prompt_version = models.CharField(max_length=80)
    factor_catalog_version = models.CharField(max_length=80, default=FACTOR_CATALOG_VERSION)
    input_hash = models.CharField(max_length=64)
    trend = models.CharField(max_length=16, choices=TREND_CHOICES, default="stable")
    risk_level = models.CharField(max_length=16, choices=RISK_CHOICES, default="low")
    confidence = models.FloatField(default=0)
    summary = models.TextField(blank=True)
    main_drivers = models.JSONField(default=list, blank=True)
    raw_response = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [
            (
                "factor",
                "period_start",
                "period_end",
                "engine",
                "model_name",
                "prompt_version",
                "factor_catalog_version",
            )
        ]
        ordering = ["-period_end", "factor__display_order"]

    def __str__(self) -> str:
        return f"{self.engine}:{self.factor.name}:{self.period_start}/{self.period_end}"
