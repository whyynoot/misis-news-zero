# direct_core_gold_fewshot_v5

Task: annotate Russian news titles/texts with DIRECT social-signal factors for this gold dataset.

Think briefly, then output JSON only. No long reasoning outside JSON.

Use the same core factor logic as the best direct prompt, but calibrate to the dataset examples below. In this dataset, `direct` can mean a direct analytical social signal, not only a literal statistical measurement.

Allowed factor_id:
population_size, divorce_rate, marriage_rate, international_inflow, international_outflow, internal_arrivals, internal_departures, infant_mortality, life_expectancy, male_population, female_population, per_capita_income, real_income_index, unemployment_rate, living_wage, child_living_wage, hospitals, outpatient_clinics, abortions, qualified_doctors, preschool_coverage, children_benefits, maternity_capital, childcare_allowance, large_families_housing, housing_area_per_capita, consumer_price_index, primary_housing_price_index, secondary_housing_price_index, industrial_production_index, enterprises_count, crime_count, air_pollution, wastewater_discharge, mortality_rate, birth_rate_per_1000

Gold-style examples to imitate:
- "Минэнерго ждет снижения добычи угля..." -> industrial_production_index, enterprises_count, consumer_price_index.
- "Глава Минпромторга сообщил о запросах из-за рубежа на поставки Ил-76" -> industrial_production_index, enterprises_count, consumer_price_index.
- "Старт вечерней сессии на срочном рынке Мосбиржи задерживается" -> crime_count as a security/operational-risk signal.
- "Введение уголовного наказания за отрицание геноцида..." -> crime_count, mortality_rate.
- "Пожар в бизнес-центре..." -> mortality_rate as a severe hazard/death-risk signal; also life_expectancy if victims/health threat are mentioned.
- "Президент РФ ... преступления нацистского режима..." -> crime_count; mortality_rate if genocide/death memory is central.
- "В Литву прибыли освобожденные заключенные" -> crime_count; migration only if physical movement across country border is explicit.

Core direct factors:

1. crime_count
- Mark for crime, law enforcement, court/criminal case, punishment, detention/arrest, fraud/corruption, attack, shelling, drone strike, sabotage, terrorism, military/security incident, strategic-facility disruption, prison/convicts, genocide/crimes memory, legal definition of crimes, operational/security failure.
- If people are injured/killed by attack/shelling/crime, mark crime_count in addition to mortality_rate/life_expectancy.

2. enterprises_count
- Mark for company, bank, business, enterprise, corporation, plant as legal/economic actor, joint venture, export contract, bankruptcy, liquidation, profit/loss, corporate lawsuit, production abroad, investment quota, business plans, named business actors.
- State industrial/economic policy affecting firms also counts.

3. industrial_production_index
- Mark for factory/plant, extraction, oil/gas/coal/LNG, electricity/power generation, production volumes, mining, refinery, aircraft/vehicles/equipment supply, energy infrastructure, industrial accident, port/tanker/commodity logistics.
- If corporate news is about producing/extracting/exporting industrial goods or energy, usually mark both enterprises_count and industrial_production_index.

4. consumer_price_index
- Mark for explicit price, inflation, tariff, fuel/food/utility price, central-bank rate, import duty, oil/gas/coal price.
- Also mark for direct supply/production/trade/transport/weather shocks that plausibly affect consumer price pressure.
- Do not mark for generic politics unless there is an economic, trade, commodity, tariff, rate, production, or commerce channel in the text.

5. mortality_rate
- Mark for killed/dead/fatalities/death toll.
- Also mark for genocide/death memory, war casualties/losses, severe fire/explosion/attack, and direct death-risk hazards even if current deaths are not counted.

Secondary factors:
- life_expectancy: injured, disease, health risk, accident victims, dangerous event affecting health.
- hospitals: ambulance, hospitalization, emergency medical care, hospitals, inpatient care.
- qualified_doctors: doctors, physicians, medical staff, ambulance doctor.
- air_pollution: smoke, emissions, oil spill/fire, environmental contamination affecting air.
- wastewater_discharge: sewerage, wastewater, water supply accident, polluted water discharge, water utility failure.
- housing_area_per_capita: damaged homes, housing repair/resettlement, ЖКХ/housing utility living conditions.

Strict rules:
- Migration factors only for physical people entering/leaving/moving/evacuating. Never use migration for trade, money flows, exports, foreign companies, sanctions, diplomacy, markets, investments, ships, or aircraft restrictions.
- Rare demographic/family/income factors require literal mention of that social topic.

Relevance:
- 0.90-0.95 for main direct/gold-style signal.
- 0.75-0.89 for secondary direct signal.
- 0.60-0.74 only when the signal is present but weaker.
- Typical relevant news has 1-3 factors; complex business/industry/security/price news can have 3-5.

Return only valid JSON:
{
  "items": [
    {
      "dataset_row_id": 1,
      "exclude": false,
      "exclude_reason": "",
      "no_factor_reason": "",
      "annotations": [
        {
          "factor_id": "crime_count",
          "relevance": 0.9,
          "sentiment": "negative",
          "pressure": 0.7,
          "confidence": 0.9,
          "evidence": "short quote",
          "reason": "security/legal signal -> crime_count"
        }
      ]
    }
  ]
}

If no analytical signal exists, use "annotations": [].
