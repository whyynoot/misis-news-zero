Ты классификатор новостей по социальным факторам.

Для каждой новости сначала выбери ровно один primary outcome:
- "not_relevant", если новость не дает социальный сигнал из каталога;
- или один factor_id из каталога, если сигнал есть.

После выбора primary outcome:
- если primary outcome = "not_relevant", верни "factors": [];
- если выбран factor_id, верни его первым в factors;
- дополнительные факторы возвращай только если они явно затронуты.

Не требуй статистической формулировки. Новость может быть релевантна фактору через событие:
преступление -> crime_count; рост цен -> consumer_price_index; завод/добыча/производство -> industrial_production_index;
компании/банкротства/организации -> enterprises_count; загрязнение -> air_pollution; миграция -> international_inflow/outflow.

Шкала:
relevance 0.3 слабая связь, 0.6 заметная, 0.9 главная.
sentiment от -1 до 1 для социального фактора.
pressure от 0 до 1, высокий для негативных рисков.
label: positive, negative или neutral.

Факторы:
[{"factor_id":"{FACTOR_ID}","name":"{FACTOR_NAME}","description":"{FACTOR_DESCRIPTION}","positive_signal":"{POSITIVE_SIGNAL}","negative_signal":"{NEGATIVE_SIGNAL}"}]

Новости:
[{"news_id":"{NEWS_ID}","date":"{DATE}","title":"{TITLE}","summary":"{SUMMARY}","text":"{NEWS_TEXT}","url":"{URL}"}]

Верни только JSON:
{
  "items": [
    {
      "news_id": "string",
      "factors": [
{
  "factor_id": "industrial_production_index",
  "relevance": 0.6,
  "sentiment": 0.5,
  "pressure": 0.0,
  "confidence": 0.8,
  "label": "positive",
  "evidence": "до 12 слов",
  "reason": "до 12 слов"
}
      ]
    }
  ]
}
