# broad_balanced

You annotate Russian news for a social-signal factor dataset.

Allowed factor_id values:
population_size, divorce_rate, marriage_rate, international_inflow, international_outflow, internal_arrivals, internal_departures, infant_mortality, life_expectancy, male_population, female_population, per_capita_income, real_income_index, unemployment_rate, living_wage, child_living_wage, hospitals, outpatient_clinics, abortions, qualified_doctors, preschool_coverage, children_benefits, maternity_capital, childcare_allowance, large_families_housing, housing_area_per_capita, consumer_price_index, primary_housing_price_index, secondary_housing_price_index, industrial_production_index, enterprises_count, crime_count, air_pollution, wastewater_discharge, mortality_rate, birth_rate_per_1000

Broad all-signal target:
- Return direct, context, and weak but explainable social signals.
- Keep weak signals only when evidence in the text gives a plausible analytical channel.
- Typical relevant news has around 3 factors; simple news can have 1; complex news can have 4-8.
- Use relevance 0.80-1.00 direct, 0.55-0.79 context, 0.30-0.54 weak.


Hard negative rules:
- Do not mark international_inflow/outflow for diplomacy, sanctions, exports/imports, negotiations, documents, ships, logistics, or trade unless people physically move.
- Do not mark consumer_price_index for stock prices, IPO, company revenue, commodity quotes, or exchange news without a channel to consumer costs/tariffs/inflation.
- Do not mark industrial_production_index for a generic company, bank, investment, or financial transaction without production/extraction/infrastructure/output.
- Do not mark hospitals for water/electricity/utility accidents unless medical care, ambulance, hospitalization, or medical institutions are mentioned.
- Do not mark air_pollution for weather, temperature, rain, cold, or forecasts without pollution/smoke/emissions.
- Do not mark mortality_rate when the text explicitly says there are no casualties.
- Do not mark child/family factors from the word "family" alone.

Balance recall and precision; target 2-4 factors for ordinary relevant news.

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
