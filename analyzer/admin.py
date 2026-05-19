from django.contrib import admin

from analyzer.models import (
    DailyFactorSentiment,
    DailySentimentSummary,
    MonitoringBatch,
    MonitoringSummary,
    NewsClassification,
    NewsItem,
    RiskFactor,
)


@admin.register(RiskFactor)
class RiskFactorAdmin(admin.ModelAdmin):
    list_display = ("display_order", "key", "name", "weight", "is_active")
    list_filter = ("is_active",)
    search_fields = ("key", "name", "description")
    ordering = ("display_order", "name")


@admin.register(NewsItem)
class NewsItemAdmin(admin.ModelAdmin):
    list_display = ("published_at", "source", "title", "category", "is_seed")
    list_filter = ("source", "is_seed", "category")
    search_fields = ("title", "summary", "text", "url", "external_id")
    date_hierarchy = "published_at"
    ordering = ("-published_at", "-id")


@admin.register(NewsClassification)
class NewsClassificationAdmin(admin.ModelAdmin):
    list_display = (
        "classified_at",
        "engine",
        "factor",
        "sentiment_label",
        "sentiment_score",
        "relevance",
        "news_item",
    )
    list_filter = ("engine", "sentiment_label", "is_relevant", "factor")
    search_fields = ("news_item__title", "factor__name", "evidence", "reason")
    autocomplete_fields = ("news_item", "factor", "batch")
    ordering = ("-classified_at",)


@admin.register(DailySentimentSummary)
class DailySentimentSummaryAdmin(admin.ModelAdmin):
    list_display = ("date", "engine", "average_sentiment", "risk_index", "news_count", "factor_count")
    list_filter = ("engine", "factor_catalog_version")
    date_hierarchy = "date"
    ordering = ("-date",)


@admin.register(DailyFactorSentiment)
class DailyFactorSentimentAdmin(admin.ModelAdmin):
    list_display = ("date", "engine", "factor", "average_sentiment", "relevant_news_count", "total_news_count")
    list_filter = ("engine", "factor")
    autocomplete_fields = ("factor", "top_positive_news", "top_negative_news")
    date_hierarchy = "date"
    ordering = ("-date", "factor__display_order")


@admin.register(MonitoringBatch)
class MonitoringBatchAdmin(admin.ModelAdmin):
    list_display = ("started_at", "finished_at", "mode", "status", "engine", "source", "news_fetched", "news_stored")
    list_filter = ("mode", "status", "engine", "source")
    search_fields = ("source", "model_name", "prompt_version")
    date_hierarchy = "started_at"
    ordering = ("-started_at",)


@admin.register(MonitoringSummary)
class MonitoringSummaryAdmin(admin.ModelAdmin):
    list_display = ("period_end", "factor", "trend", "risk_level", "confidence", "model_name")
    list_filter = ("trend", "risk_level", "model_name", "factor_catalog_version")
    autocomplete_fields = ("factor",)
    date_hierarchy = "period_end"
    ordering = ("-period_end", "factor__display_order")
