# direct_strict

You annotate Russian news for a social-signal factor dataset.

Allowed factor_id values:
population_size, divorce_rate, marriage_rate, international_inflow, international_outflow, internal_arrivals, internal_departures, infant_mortality, life_expectancy, male_population, female_population, per_capita_income, real_income_index, unemployment_rate, living_wage, child_living_wage, hospitals, outpatient_clinics, abortions, qualified_doctors, preschool_coverage, children_benefits, maternity_capital, childcare_allowance, large_families_housing, housing_area_per_capita, consumer_price_index, primary_housing_price_index, secondary_housing_price_index, industrial_production_index, enterprises_count, crime_count, air_pollution, wastewater_discharge, mortality_rate, birth_rate_per_1000

Direct-only target:
- Return a factor only when the news text directly describes the event represented by that factor.
- Do not return weak macro implications, generic political context, or long causal chains.
- Typical direct positive count is 0-2 factors. Complex accidents/attacks can have 3-5.
- Use relevance >= 0.80 for a direct main signal and 0.55-0.79 for a direct secondary signal.
- Do not use relevance below 0.55 in direct-only prompts.

Direct examples:
- death, killed, fatalities -> mortality_rate.
- injured, threat to life/health, disease outbreak -> life_expectancy.
- hospital, ambulance, hospitalization, inpatient care -> hospitals.
- doctors, medical staff, shortage of doctors -> qualified_doctors.
- attack, crime, fraud, corruption, criminal case, violence -> crime_count.
- consumer prices, tariffs, inflation, fuel/food/utility prices -> consumer_price_index.
- wages, pensions, household payments, personal income -> per_capita_income.
- purchasing power, inflation pressure on income, debts/credit burden -> real_income_index.
- production, extraction, plant, industrial infrastructure, energy output -> industrial_production_index.
- company, bank, business opening/closure, bankruptcy, corporate activity -> enterprises_count.
- damaged housing, resettlement, housing conditions, utilities in homes -> housing_area_per_capita.
- water supply, sewerage, wastewater, water pollution -> wastewater_discharge.
- smoke, emissions, air pollution, oil spill/fire with air/ecological damage -> air_pollution.
- actual migration or evacuation of people -> migration factors. Diplomacy/trade is not migration.


Hard negative rules:
- Do not mark international_inflow/outflow for diplomacy, sanctions, exports/imports, negotiations, documents, ships, logistics, or trade unless people physically move.
- Do not mark consumer_price_index for stock prices, IPO, company revenue, commodity quotes, or exchange news without a channel to consumer costs/tariffs/inflation.
- Do not mark industrial_production_index for a generic company, bank, investment, or financial transaction without production/extraction/infrastructure/output.
- Do not mark hospitals for water/electricity/utility accidents unless medical care, ambulance, hospitalization, or medical institutions are mentioned.
- Do not mark air_pollution for weather, temperature, rain, cold, or forecasts without pollution/smoke/emissions.
- Do not mark mortality_rate when the text explicitly says there are no casualties.
- Do not mark child/family factors from the word "family" alone.

Be conservative: precision matters more than recall.

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
          "relevance": 0.8,
          "sentiment": "negative",
          "pressure": 0.6,
          "confidence": 0.8,
          "evidence": "short quote from news",
          "reason": "attack -> crime_count"
        }
      ]
    }
  ]
}
Use only factor_id from the list. If no factor is relevant, annotations=[].
