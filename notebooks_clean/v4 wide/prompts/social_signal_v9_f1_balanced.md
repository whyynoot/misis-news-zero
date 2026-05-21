# social_signal_v9_f1_balanced

Ты размечаешь русскую новость для датасета социальных сигналов РФ.

Задача: вернуть все factor_id из списка 36, для которых в тексте есть видимый причинный или аналитический канал. Это multi-label задача, не выбор одной рубрики.

Главная цель: высокий micro-F1.

Поэтому:
- не пропускай основные co-labels;
- не добавляй слабую или скрытую связь без evidence;
- используй direct analytical signal: фактор может быть прямым аналитическим социальным сигналом, даже если в новости нет буквального статистического показателя Росстата;
- не используй relevance < 0.60;
- обычная релевантная новость: 1-3 фактора;
- сложная бизнес / промышленность / безопасность / цены: 3-5 факторов;
- максимум 6 факторов.

Allowed factor_id:
population_size, divorce_rate, marriage_rate, international_inflow, international_outflow, internal_arrivals, internal_departures, infant_mortality, life_expectancy, male_population, female_population, per_capita_income, real_income_index, unemployment_rate, living_wage, child_living_wage, hospitals, outpatient_clinics, abortions, qualified_doctors, preschool_coverage, children_benefits, maternity_capital, childcare_allowance, large_families_housing, housing_area_per_capita, consumer_price_index, primary_housing_price_index, secondary_housing_price_index, industrial_production_index, enterprises_count, crime_count, air_pollution, wastewater_discharge, mortality_rate, birth_rate_per_1000.

Не используй not_relevant, other, unclear или новые классы.

Scope:
Включай новости про РФ, российские регионы, население, компании, бюджет, промышленность, транспорт, инфраструктуру, медицину, жилье, безопасность, цены, занятость, доходы, экологию, семьи, детей, миграцию.

Зарубежную новость включай только если есть конкретный канал влияния на РФ: российские компании/граждане/активы, санкции против РФ, экспорт/импорт РФ, нефть/газ/СПГ/уголь/уран/продовольствие, валюты/ставки/цены, логистика, конфликт вокруг Украины, энергетический или торговый канал.

Если нет фактора: annotations=[].

Core factor rules:

1. enterprises_count
Ставь для named company, bank, business, commercial operator, plant as business actor, export/import contract, bankruptcy/liquidation, profit/loss, corporate lawsuit, investment, sanctions/regulation affecting firms.
Обычно ставь вместе с industrial_production_index, если компания производит, добывает, транспортирует, импортирует, экспортирует или поставляет industrial goods, energy, vehicles, fuel, food, raw materials.

2. industrial_production_index
Ставь для factory/plant, extraction, oil/gas/coal/уран/LNG, electricity/energy infrastructure, mining/refinery, production volume, industrial goods, aircraft/vehicles/equipment, ports/tankers/rail/airports/logistics shutdown, industrial accident, commodity/resource policy affecting production or supply.

3. consumer_price_index
Ставь для explicit prices, inflation, tariffs, fuel/food/utility prices, key rate, duties, oil/gas/coal/food prices, exchange rate.
Также ставь для direct supply/logistics/production/trade/weather shocks, которые plausibly affect consumer prices.

4. crime_count
Ставь для crime, law enforcement, court/criminal case, punishment, detention/arrest, fraud/corruption, FSB/police/prosecutor, attack/shelling/drone strike/sabotage/terrorism, military/security incident, strategic-facility disruption, prison/convicts, genocide/crimes memory, legal definition of crimes.

5. mortality_rate
Ставь для killed/dead/fatalities/death toll; genocide/death memory; war losses; severe fire/explosion/attack/fatal accident/death-risk disaster, когда death/fatal hazard центральный.

6. life_expectancy
Ставь для injuries, disease outbreak, poisoning, infection, serious health threat, dangerous event affecting health.

7. hospitals
Ставь только если есть hospital, ambulance, hospitalization, emergency medical care, people brought/transferred to doctors.

8. qualified_doctors
Ставь для doctors, physicians, medical staff, ambulance doctor.

9. air_pollution
Ставь для smoke, emissions, oil/fuel contamination, environmental contamination affecting air/coast, major fuel/chemical spill/fire with pollution channel.

10. wastewater_discharge
Ставь для sewerage, wastewater, polluted water discharge, water utility/sewage accident.

11. housing_area_per_capita
Ставь для damaged homes, resettlement, housing repair, housing utilities/living conditions, property/living-space rules.

12. Income/labor:
per_capita_income = wages, pensions, social payments, taxes, compensation, household money.
real_income_index = purchasing power, inflation/debt burden on income.
unemployment_rate = layoffs, job loss, labor restrictions, unemployment.

13. Family/demography factors
Требуют буквального социального топика: births, marriages, divorces, children benefits, maternity capital, childcare, preschool, abortions, infant deaths, population size or sex structure.

Strict negative rules:
- Migration factors only for physical movement of people. Never use international_inflow, international_outflow, internal_arrivals, internal_departures for exports/imports, trade, sanctions, foreign firms, diplomacy, ships, aircraft restrictions, money flows, markets, investments, cargo/logistics without people.
- Do not use hospitals for utility accidents, water supply, rescue, injuries, or "people saved" unless hospital/ambulance/doctors/medical transfer is explicit.
- Do not use mortality_rate for ordinary injury/market/economic news unless deaths, fatal risk, genocide, war losses, or severe disaster are central.
- Do not use industrial_production_index for pure stock movement, finance, IPO, valuation, court, culture, or politics without production/extraction/energy/logistics/supply.
- Do not use consumer_price_index for corporate revenue, asset price, IPO, sanctions, politics, or finance unless consumer prices/rates/tariffs/commodities/supply costs are visible.
- Do not use air_pollution for weather without smoke/emissions/pollution.
- Do not use population_size, male_population, female_population for education, politics, culture, or business unless численность/структура населения is explicit.

Gold calibration examples:
- "Старт вечерней сессии на срочном рынке Мосбиржи задерживается" -> crime_count.
- "Аэропорты Волгограда и Краснодара приостановили работу" -> industrial_production_index.
- "Глава Минпромторга сообщил о запросах из-за рубежа на поставки Ил-76" -> industrial_production_index, enterprises_count, consumer_price_index.
- "Мировые цены на продовольствие выросли" -> consumer_price_index, industrial_production_index.
- "Более 100 человек обратились в больницу с кишечной инфекцией" -> life_expectancy, hospitals.
- "Введение уголовного наказания за отрицание геноцида" -> crime_count, mortality_rate.
- "КТК сообщил о повреждении причала в результате террористической атаки" -> crime_count, industrial_production_index, enterprises_count; add mortality_rate/life_expectancy only if fatal or health risk is explicit; add air_pollution/wastewater_discharge only if pollution/water channel is explicit.

Relevance:
- 0.90-0.95: main direct/gold-style signal;
- 0.75-0.89: secondary direct signal;
- 0.60-0.74: visible weaker but still direct analytical signal;
- below 0.60: do not return.

Sentiment:
- negative: worsens risk/social condition/pressure;
- positive: improves support, access, production, income, safety, housing, medicine;
- neutral: factual/regulatory without clear improvement/worsening.

Pressure:
- 0.0-0.3: low/no risk pressure;
- 0.4-0.6: moderate;
- 0.7-1.0: high risk/negative pressure.

Return only valid JSON, no markdown, no prose:
{
  "items": [
    {
      "dataset_row_id": "{DATASET_ROW_ID}",
      "exclude": false,
      "exclude_reason": "",
      "no_factor_reason": "",
      "annotations": [
        {
          "factor_id": "industrial_production_index",
          "relevance": 0.9,
          "sentiment": "neutral",
          "pressure": 0.5,
          "confidence": 0.9,
          "evidence": "short exact quote from news",
          "reason": "visible channel -> factor_id"
        }
      ]
    }
  ]
}
