Сформируй краткие summaries по социальным факторам за период.

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

{"period_start":"{PERIOD_START}","period_end":"{PERIOD_END}","factors":[{"factor_id":"{FACTOR_ID}","metrics":"{AGGREGATED_METRICS}","top_news":["{TOP_NEWS_ITEM}"]}]}

Верни строго JSON по схеме:

{
  "period_start": "YYYY-MM-DD",
  "period_end": "YYYY-MM-DD",
  "engine": "llm",
  "factors": [
    {
      "factor_id": "outpatient_clinics",
      "trend": "improving",
      "risk_level": "low",
      "confidence": 0.0,
      "summary": "2-4 коротких предложения.",
      "main_drivers": [
        {
          "title": "заголовок новости",
          "date": "YYYY-MM-DD",
          "impact": -0.4,
          "why": "до 14 слов"
        }
      ]
    }
  ]
}

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
- Если по фактору мало данных, risk_level="low" или "medium", confidence ниже, а в summary укажи, что сигнал слабый.
