# direct_core_filtered_v3

Task: annotate Russian news titles/texts with DIRECT social-signal factors.

Think briefly, then output JSON only. No long reasoning outside JSON.

Goal: maximize direct factor F1 for this dataset. Prefer the core direct factors; avoid rare factors unless the news literally states them.

Allowed factor_id:
population_size, divorce_rate, marriage_rate, international_inflow, international_outflow, internal_arrivals, internal_departures, infant_mortality, life_expectancy, male_population, female_population, per_capita_income, real_income_index, unemployment_rate, living_wage, child_living_wage, hospitals, outpatient_clinics, abortions, qualified_doctors, preschool_coverage, children_benefits, maternity_capital, childcare_allowance, large_families_housing, housing_area_per_capita, consumer_price_index, primary_housing_price_index, secondary_housing_price_index, industrial_production_index, enterprises_count, crime_count, air_pollution, wastewater_discharge, mortality_rate, birth_rate_per_1000

Primary direct factors to check first:

1. crime_count
- Mark for crime, law enforcement, court/criminal case, punishment, detention/arrest, fraud/corruption, attack, shelling, drone strike, sabotage, terrorism, military/security incident, conflict around strategic facilities, prison/convicts, genocide/crimes memory.
- If people are injured/killed by attack/shelling/crime, also mark crime_count.

2. enterprises_count
- Mark for company, bank, enterprise, corporation, plant as business actor, joint venture, export contract, bankruptcy, liquidation, profit/loss, corporate lawsuit, production abroad, investment quota, business plans, named business actors.

3. industrial_production_index
- Mark for factory/plant, extraction, oil/gas/coal/LNG, electricity/power, production volumes, mining, refinery, aircraft/vehicles/equipment supply, energy infrastructure, industrial accident, port/tanker/commodity logistics.

4. consumer_price_index
- Mark for explicit price/inflation/tariff/rate/duty/fuel/food/utility cost.
- Also mark for oil/gas/coal prices, central-bank rate, import duties, sanctions/trade restrictions, weather/transport/industrial supply disruption when the text plausibly affects consumer price pressure.

5. mortality_rate
- Mark for killed/dead/fatalities, death toll, genocide/death memory, war/conflict casualties, severe fire/explosion/attack with possible deaths.

Secondary direct factors:
- life_expectancy: injured, disease, health threat, accident victims, dangerous event affecting health.
- hospitals: ambulance, hospitalization, emergency medical care, hospitals, inpatient care.
- air_pollution: smoke, emissions, oil spill/fire, environmental contamination affecting air.
- wastewater_discharge: sewerage, wastewater, water supply accident, polluted water discharge, water utility failure.

Suppress noisy rare factors:
- Do NOT mark per_capita_income, real_income_index, maternity_capital, childcare_allowance, housing_area_per_capita, outpatient_clinics, qualified_doctors, unemployment_rate, population/male/female/marriage/divorce/birth factors unless the news literally states that exact social topic.
- Do NOT mark migration factors unless physical people enter/leave/move/evacuate. Never use migration for trade, money flows, exports, foreign companies, sanctions, diplomacy, markets, investments, ships, or aircraft restrictions.

Relevance:
- 0.90-0.95 for main direct core signal.
- 0.75-0.89 for secondary direct signal.
- 0.60-0.74 only when the signal is present but weaker.
- Avoid returning more than 3 factors unless the news clearly combines business/industry/security/price or casualties.

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
          "reason": "security/crime signal -> crime_count"
        }
      ]
    }
  ]
}

If no direct analytical signal exists, use "annotations": [].
