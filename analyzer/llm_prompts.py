import json
from typing import Any, Sequence

from analyzer.constants import LLM_CLASSIFICATION_PROMPT_VERSION, LLM_SUMMARY_PROMPT_VERSION

CLASSIFICATION_SYSTEM_PROMPT = """Ты быстрый классификатор новостей для мониторинга социальных сигналов.
Твоя задача - оценить, как каждая новость влияет на заданные социальные факторы.
Возвращай только валидный JSON.
Не используй markdown.
Не добавляй пояснения вне JSON.
Не раскрывай ход рассуждений.
Не пиши "я думаю".
Пиши кратко.
Все числа возвращай в диапазонах, указанных в схеме.
Если новость не относится к фактору, ставь relevance=0, sentiment=0, pressure=0, label="neutral"."""

SUMMARY_SYSTEM_PROMPT = """Ты модуль краткой суммаризации новостных социальных сигналов.
Твоя задача - кратко объяснить, что произошло по каждому социальному фактору за период.
Возвращай только валидный JSON.
Не используй markdown.
Не добавляй текст вне JSON.
Не раскрывай ход рассуждений.
Не пиши общие советы.
Не пиши фразы вроде "хотите, я помогу".
Пиши кратко и по делу."""


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_classification_user_prompt(news_items: Sequence[dict], factor_catalog: Sequence[dict]) -> str:
    return f"""Проанализируй новости по социальным факторам.

Правила оценки:

relevance:
0 - новость не относится к фактору.
0.3 - слабая косвенная связь.
0.6 - заметная связь.
1.0 - фактор является одним из главных смыслов новости.

sentiment:
-1 - сильный негативный сигнал для фактора.
-0.5 - умеренный негативный сигнал.
0 - нейтрально или нерелевантно.
0.5 - умеренный позитивный сигнал.
1 - сильный позитивный сигнал.

pressure:
0 - нет социального давления/риска.
0.5 - умеренное давление.
1 - сильное давление.
Если sentiment положительный, pressure обычно должен быть 0 или низким.
Если sentiment отрицательный и relevance высокий, pressure должен быть выше.

label:
positive - позитивный сигнал.
negative - негативный сигнал.
neutral - нейтрально или нерелевантно.

confidence:
0 - низкая уверенность.
1 - высокая уверенность.

Факторы:

{compact_json(factor_catalog)}

Новости:

{compact_json(news_items)}

Верни строго JSON по схеме:

{{
  "items": [
    {{
      "news_id": "string",
      "factors": [
        {{
          "factor_id": "birth_rate_per_1000",
          "relevance": 0.0,
          "sentiment": 0.0,
          "pressure": 0.0,
          "confidence": 0.0,
          "label": "neutral",
          "evidence": "короткая фраза",
          "reason": "до 12 слов"
        }}
      ]
    }}
  ]
}}

Требования к ответу:
- Верни все news_id из входа.
- В factors возвращай только факторы, у которых relevance > 0.
- Если по новости нет релевантных факторов, верни "factors": [].
- Пропущенные factor_id система считает нейтральными: relevance=0, sentiment=0, pressure=0, label="neutral".
- Не добавляй factor_id, которых нет в списке.
- Не добавляй текст вне JSON.
- Не используй markdown.
- evidence не длиннее 12 слов.
- reason не длиннее 12 слов.
- Если данных недостаточно, ставь confidence ниже.
- Не возвращай нерелевантные факторы только ради заполнения списка."""


def build_summary_user_prompt(summary_input: dict) -> str:
    return f"""Сформируй краткие summaries по социальным факторам за период.

Это не прогноз и не вывод о реальном состоянии общества. Это только summary новостных сигналов.

Входные данные содержат:
- период;
- фактор;
- агрегированные метрики;
- список главных новостей, которые дали вклад в сигнал.

Правила:
- summary: 2-4 коротких предложения;
- main_drivers: 2-5 пунктов;
- не пересказывай все новости подряд;
- выделяй только главное;
- если данные слабые или новостей мало, прямо укажи это в summary;
- не добавляй длинные рассуждения;
- не используй markdown;
- не добавляй текст вне JSON.

Вход:

{compact_json(summary_input)}

Верни строго JSON по схеме:

{{
  "period_start": "YYYY-MM-DD",
  "period_end": "YYYY-MM-DD",
  "engine": "llm",
  "factors": [
    {{
      "factor_id": "outpatient_clinics",
      "trend": "improving",
      "risk_level": "low",
      "confidence": 0.0,
      "summary": "2-4 коротких предложения.",
      "main_drivers": [
        {{
          "title": "заголовок новости",
          "date": "YYYY-MM-DD",
          "impact": -0.4,
          "why": "до 14 слов"
        }}
      ]
    }}
  ]
}}

Допустимые trend:
- improving
- worsening
- stable
- mixed

Допустимые risk_level:
- low
- medium
- high

Требования:
- Верни summary для каждого factor_id из входа.
- Не добавляй factor_id, которых нет во входе.
- Не добавляй текст вне JSON.
- Не используй markdown.
- why не длиннее 14 слов.
- Если по фактору мало данных, risk_level="low" или "medium", confidence ниже, а в summary укажи, что сигнал слабый."""
