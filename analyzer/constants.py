from analyzer.factors import FACTOR_ALIASES, FACTOR_CATALOG_VERSION

ENGINE_BERT = "bert"
ENGINE_LLM = "llm"

ENGINE_CHOICES = [
    (ENGINE_BERT, "BERT"),
    (ENGINE_LLM, "LLM"),
]

BERT_MODEL_NAME = "cointegrated/rubert-base-cased-nli-threeway"
BERT_PROMPT_VERSION = "bert-zero-shot-v1"

LLM_CLASSIFICATION_PROMPT_VERSION = "social-risk-classification-v1"
LLM_SUMMARY_PROMPT_VERSION = "social-risk-summary-v1"

def normalize_engine(value: str | None) -> str:
    engine = (value or ENGINE_BERT).strip().lower()
    if engine not in {ENGINE_BERT, ENGINE_LLM}:
        return ENGINE_BERT
    return engine


def normalize_factor_key(value: str | None) -> str | None:
    if not value:
        return None
    return FACTOR_ALIASES.get(value.strip().lower(), value.strip())


def parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
