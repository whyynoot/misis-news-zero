# direct_business_migration_fixed

Task: annotate Russian news titles/texts with DIRECT social-signal factors.

Think briefly, then output JSON only. No long reasoning outside JSON.

Important: the title is enough evidence. If the title directly names a company, bank, plant, project, export, bankruptcy, profit, investment quota, production plan, merger/liquidation, lawsuit against a company, or business activity, mark `enterprises_count`. This dataset treats corporate/economic actors as direct enterprise signals.

Allowed factor_id:
population_size, divorce_rate, marriage_rate, international_inflow, international_outflow, internal_arrivals, internal_departures, infant_mortality, life_expectancy, male_population, female_population, per_capita_income, real_income_index, unemployment_rate, living_wage, child_living_wage, hospitals, outpatient_clinics, abortions, qualified_doctors, preschool_coverage, children_benefits, maternity_capital, childcare_allowance, large_families_housing, housing_area_per_capita, consumer_price_index, primary_housing_price_index, secondary_housing_price_index, industrial_production_index, enterprises_count, crime_count, air_pollution, wastewater_discharge, mortality_rate, birth_rate_per_1000

Direct mapping:
- company, bank, business, enterprise, corporation, plant as legal/economic actor, joint venture, export contract, bankruptcy, liquidation, corporate profit/loss, lawsuit against company, production abroad, investment quota -> enterprises_count
- factory, extraction, gas/oil/LNG/coal, power generation, industrial output, production volume, plant operation -> industrial_production_index
- consumer price, inflation, tariff, fuel/food/utility price -> consumer_price_index
- crime, attack, shelling, fraud, corruption, criminal case, detention, violence -> crime_count
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
- Typical relevant news has 1-3 direct factors; complex attacks/accidents/corporate-industrial news can have 3-5.

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
          "factor_id": "enterprises_count",
          "relevance": 0.9,
          "sentiment": "neutral",
          "pressure": 0.5,
          "confidence": 0.9,
          "evidence": "short quote",
          "reason": "company/business event -> enterprises_count"
        }
      ]
    }
  ]
}

If no direct factor exists, use "annotations": [].
