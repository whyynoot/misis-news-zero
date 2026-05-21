Ты классифицируешь новости только по одной группе социальных факторов.

Группа: {GROUP_TITLE}
group_id: {GROUP_ID}

Цель: высокий recall по ручной event-level разметке.
Лучше вернуть спорный, но реально возможный фактор с relevance=0.3, чем пропустить фактор, который поставил бы разметчик.
Но не добавляй факторы по чистой ассоциации слов: нужна связь события новости с определением фактора.

Политика разметки:
- Факторы являются event-level социальными сигналами, а не только буквальными статистическими показателями.
- Если событие новости соответствует event_definition фактора, верни этот фактор даже без статистики.
- Для одной новости в этой группе можно вернуть несколько факторов.
- relevance=0.3: слабая, но реальная связь с фактором.
- relevance=0.6: фактор заметно затронут.
- relevance=0.9 или 1.0: фактор является главным смыслом новости.
- Если по этой группе нет факторов, верни пустой список.

Факторы этой группы:
[{"factor_id":"{FACTOR_ID}","dashboard_name":"{DASHBOARD_NAME}","social_signal_meaning":"{SOCIAL_SIGNAL_MEANING}","count_as_signal_when":"{COUNT_AS_SIGNAL_WHEN}","do_not_use_when":"{DO_NOT_USE_WHEN}","positive_signal":"{POSITIVE_SIGNAL}","negative_signal":"{NEGATIVE_SIGNAL}"}]

Позитивные примеры из gold-разметки.
Формат: title -> expected_factors_in_this_group.
Используй их как стиль разметки, но не копируй автоматически.
[{"title":"{POSITIVE_EXAMPLE_TITLE}","expected_factors_in_this_group":["{FACTOR_ID}"]}]

Негативные/пустые примеры.
Формат: title -> expected_factors_in_this_group=[].
[{"title":"{NEGATIVE_EXAMPLE_TITLE}","expected_factors_in_this_group":[]}]

Новости для классификации:
[{"news_id":"{NEWS_ID}","date":"{DATE}","title":"{TITLE}","summary":"{SUMMARY}","text":"{NEWS_TEXT}","url":"{URL}"}]

Верни строго JSON:
{
  "items": [
    {
      "news_id": "string",
      "factors": [
        {
          "factor_id": "crime_count",
          "relevance": 0.6,
          "sentiment": -0.5,
          "pressure": 0.5,
          "confidence": 0.8,
          "label": "negative",
          "evidence": "короткая фраза из новости",
          "reason": "до 12 слов"
        }
      ]
    }
  ]
}

Требования:
- Верни все news_id из входа.
- Возвращай только factor_id из списка факторов этой группы.
- Не возвращай relevance=0 факторы.
- Если факторов в группе нет, верни "factors": [].
- sentiment: -1, -0.5, 0, 0.5, 1.
- label: positive, negative или neutral.
- Не добавляй текст вне JSON.
