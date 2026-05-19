from dataclasses import dataclass
from typing import Any, Sequence

from analyzer.constants import ENGINE_BERT, ENGINE_LLM, normalize_engine


@dataclass(frozen=True)
class AnalyzerResult:
    classified_count: int
    affected_dates: list[Any]
    errors: list[str]
    warnings: list[str]
    cached_count: int = 0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "AnalyzerResult":
        return cls(
            classified_count=int(payload.get("classified_count", 0)),
            affected_dates=list(payload.get("affected_dates", [])),
            errors=list(payload.get("errors", [])),
            warnings=list(payload.get("warnings", [])),
            cached_count=int(payload.get("cached_count", 0)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "classified_count": self.classified_count,
            "affected_dates": self.affected_dates,
            "errors": self.errors,
            "warnings": self.warnings,
            "cached_count": self.cached_count,
        }


class MonitoringAnalyzer:
    engine = ENGINE_BERT

    def classify(self, news_items: Sequence[Any], factors: Sequence[Any], batch=None, force: bool = False) -> AnalyzerResult:
        raise NotImplementedError


class BertMonitoringAnalyzer(MonitoringAnalyzer):
    engine = ENGINE_BERT

    def classify(self, news_items: Sequence[Any], factors: Sequence[Any], batch=None, force: bool = False) -> AnalyzerResult:
        from analyzer.monitoring_service import classify_live_news_items

        return AnalyzerResult.from_payload(classify_live_news_items(news_items, factors, batch=batch, force=force))


class LLMMonitoringAnalyzer(MonitoringAnalyzer):
    engine = ENGINE_LLM

    def classify(self, news_items: Sequence[Any], factors: Sequence[Any], batch=None, force: bool = False) -> AnalyzerResult:
        from analyzer.llm_services import classify_news_items_with_llm

        return AnalyzerResult.from_payload(classify_news_items_with_llm(news_items, factors, batch=batch, force=force))


def get_monitoring_analyzer(engine: str) -> MonitoringAnalyzer:
    normalized = normalize_engine(engine)
    if normalized == ENGINE_LLM:
        return LLMMonitoringAnalyzer()
    return BertMonitoringAnalyzer()
