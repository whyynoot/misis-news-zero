Разметь одну новость для мониторинга социальных рисков.

Задача: найти ВСЕ содержательные связи news -> factor_id. Это не выбор одной рубрики.

Жесткий prior по полноте:
- В v3 gold релевантная новость обычно имеет около 3 факторов.
- Нормальный диапазон: 2-4 фактора.
- 1 фактор допустим только если новость очень узкая.
- Для комплексных тем допустимо 4-8 факторов.
- Лучше добавить потенциальную связь с relevance=0.3, чем пропустить co-label.
- Но каждый фактор должен иметь короткую причинную цепочку в reason.

relevance:
0.9 прямой/главный сигнал; 0.6 явная связь; 0.3 слабая потенциальная связь.

Быстрый checklist:
- атака/преступление -> crime_count; погибшие -> mortality_rate; раненые -> life_expectancy; больница/скорая -> hospitals.
- цены/тарифы/инфляция -> consumer_price_index; часто также real_income_index/per_capita_income.
- производство/добыча/энергетика/завод/порт/логистика -> industrial_production_index; часто enterprises_count.
- компания/банкротство/открытие/закрытие/бизнес -> enterprises_count.
- дома/ЖКХ/нет воды/нет света/расселение -> housing_area_per_capita; вода/стоки -> wastewater_discharge.
- врачи/медики/медперсонал -> qualified_doctors.
- выплаты семьям/детям -> children_benefits/living_wage/per_capita_income.
- миграция только при движении людей: въезд/выезд/эвакуация/туристы/беженцы/переселение. Не для дипломатии/экспорта/санкций.

Каталог:
[{"id":"{FACTOR_ID}","hint":"{COUNT_AS_SIGNAL_WHEN}"}]

Новость:
[{"news_id":"{NEWS_ID}","date":"{DATE}","title":"{TITLE}","summary":"{SUMMARY}","text":"{NEWS_TEXT}","url":"{URL}"}]

Верни только JSON:
{"items":[{"news_id":"string","factors":[{"factor_id":"crime_count","relevance":0.6,"sentiment":-0.5,"pressure":0.5,"confidence":0.8,"label":"negative","evidence":"фраза","reason":"атака -> crime_count"}]}]}

Требования: factor_id только из каталога; factors 0..8; sentiment -1/-0.5/0/0.5/1; label positive/neutral/negative; evidence/reason до 8 слов; без markdown.
