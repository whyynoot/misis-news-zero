# direct_brief_thinking

Task: annotate Russian news with direct social-signal factors.

Think very briefly. Do not do long chain-of-thought. Decide factors, then output JSON only.

Allowed factor_id:
population_size, divorce_rate, marriage_rate, international_inflow, international_outflow, internal_arrivals, internal_departures, infant_mortality, life_expectancy, male_population, female_population, per_capita_income, real_income_index, unemployment_rate, living_wage, child_living_wage, hospitals, outpatient_clinics, abortions, qualified_doctors, preschool_coverage, children_benefits, maternity_capital, childcare_allowance, large_families_housing, housing_area_per_capita, consumer_price_index, primary_housing_price_index, secondary_housing_price_index, industrial_production_index, enterprises_count, crime_count, air_pollution, wastewater_discharge, mortality_rate, birth_rate_per_1000

Return a factor only if the news itself contains a direct event/signal for it.
No broad geopolitics, no long causal chains, no generic context.
Most relevant news has 1-3 direct factors; accidents/attacks can have 3-5.

Mapping rules:
- killed, dead, fatality, death toll -> mortality_rate
- birth, newborn, fertility -> birth_rate_per_1000
- infant death -> infant_mortality and mortality_rate
- injured, disease, life/health threat -> life_expectancy
- hospital, ambulance, hospitalization, inpatient care -> hospitals
- clinic, outpatient visit/polyclinic -> outpatient_clinics
- doctors, medical staff, physician shortage -> qualified_doctors
- crime, fraud, corruption, attack, violence, criminal case -> crime_count
- price, inflation, tariff, fuel/food/utility cost -> consumer_price_index
- income, wage, salary, pension, household payment -> per_capita_income
- purchasing power, debt burden, income eroded by inflation -> real_income_index
- unemployment, layoffs, job loss -> unemployment_rate
- plant, production, extraction, factory, industrial output, energy generation -> industrial_production_index
- company, bank, business, enterprise, bankruptcy, opening/closure -> enterprises_count
- housing damage, resettlement, home conditions, housing utilities -> housing_area_per_capita
- new-build housing prices -> primary_housing_price_index
- resale housing prices -> secondary_housing_price_index
- air emissions, smoke, air pollution -> air_pollution
- sewage, wastewater, polluted water discharge -> wastewater_discharge
- migration, evacuation, arrivals/departures of people -> corresponding migration factor

Use relevance:
- 0.85-1.00 main direct signal
- 0.60-0.84 secondary direct signal
- never below 0.60 in this direct prompt

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
          "reason": "attack -> crime_count"
        }
      ]
    }
  ]
}

If no direct factor exists, use "annotations": [].
