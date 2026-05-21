# direct_core_recall_v2

Task: annotate Russian news titles/texts with DIRECT social-signal factors.

Think briefly, then output JSON only. No long reasoning outside JSON.

Goal: high recall on direct factors without adding weak geopolitical implications. The title is valid evidence.

Allowed factor_id:
population_size, divorce_rate, marriage_rate, international_inflow, international_outflow, internal_arrivals, internal_departures, infant_mortality, life_expectancy, male_population, female_population, per_capita_income, real_income_index, unemployment_rate, living_wage, child_living_wage, hospitals, outpatient_clinics, abortions, qualified_doctors, preschool_coverage, children_benefits, maternity_capital, childcare_allowance, large_families_housing, housing_area_per_capita, consumer_price_index, primary_housing_price_index, secondary_housing_price_index, industrial_production_index, enterprises_count, crime_count, air_pollution, wastewater_discharge, mortality_rate, birth_rate_per_1000

Always check these high-frequency direct factors:

1. enterprises_count
- Mark for company, bank, business, enterprise, corporation, joint venture, export contract, bankruptcy, liquidation, profit/loss, lawsuit against a company, production abroad, investment quota, corporate plan/project.
- If the news is about a named business actor or business operation, usually mark enterprises_count.

2. crime_count
- Mark for any crime/security/violence/law-enforcement signal: attack, shelling, drone strike, sabotage, terrorism, explosion caused by attack, shooting, fraud, corruption, criminal case, detention, arrest, court sentence, illegal action, investigation, police/FSB/prosecutor.
- If people are injured/killed by attack/shelling/crime, mark crime_count in addition to mortality_rate/life_expectancy.

3. industrial_production_index
- Mark for plant/factory, extraction, LNG/gas/oil/coal, electricity/power generation, industrial output, production volumes, industrial infrastructure, production launch/stop, mining, refinery, energy facility, manufacturing.
- If a corporate news item is about producing/extracting/exporting industrial goods or energy, mark both enterprises_count and industrial_production_index.

4. consumer_price_index
- Mark for price, inflation, tariff, fuel price, food price, utility cost, electricity/gas/heating/water tariff, cost of consumer goods/services.
- Do not infer prices from sanctions or trade unless the text explicitly mentions prices/tariffs/costs/inflation.

Other direct mappings:
- killed, dead, fatality, death toll -> mortality_rate
- injured, disease, life/health threat -> life_expectancy
- hospital, ambulance, hospitalization, inpatient care -> hospitals
- doctor, physician, medical worker -> qualified_doctors
- housing damage, home repair/resettlement, living space, housing utilities -> housing_area_per_capita
- air emissions, smoke, air pollution -> air_pollution
- sewage, wastewater, polluted water discharge -> wastewater_discharge
- birth, newborn, fertility -> birth_rate_per_1000
- infant death -> infant_mortality and mortality_rate
- unemployment, layoffs, job loss -> unemployment_rate
- wage, salary, pension, household payment -> per_capita_income
- purchasing power, income eroded by inflation, debt burden -> real_income_index

Strict migration rule:
- Use international_inflow/outflow only for physical people migrating or entering/leaving a country.
- Use internal_arrivals/departures only for physical people moving/evacuating inside the country.
- Do NOT use migration factors for exports/imports, foreign companies, sanctions, diplomacy, ships, aircraft restrictions, money flows, markets, investments, or international cooperation.

Relevance:
- 0.85-1.00 main direct signal.
- 0.60-0.84 secondary direct signal.
- Do not use relevance below 0.60.
- Typical relevant news has 1-3 direct factors; complex attacks/industrial-business news can have 3-5.

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
          "reason": "attack/criminal event -> crime_count"
        }
      ]
    }
  ]
}

If no direct factor exists, use "annotations": [].
