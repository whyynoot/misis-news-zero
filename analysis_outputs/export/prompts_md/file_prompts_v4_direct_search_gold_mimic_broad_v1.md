# gold_mimic_broad_v1

Task: annotate Russian news for the current gold dataset logic.

Think briefly, then output JSON only. No long reasoning outside JSON.

Important: in this dataset, `direct` often means "the news is a direct analytical social signal for this factor", not only a literal statistical measurement. Use broad but text-grounded links. Prefer recall over strictness. Typical relevant news has 2-4 factors.

Allowed factor_id:
population_size, divorce_rate, marriage_rate, international_inflow, international_outflow, internal_arrivals, internal_departures, infant_mortality, life_expectancy, male_population, female_population, per_capita_income, real_income_index, unemployment_rate, living_wage, child_living_wage, hospitals, outpatient_clinics, abortions, qualified_doctors, preschool_coverage, children_benefits, maternity_capital, childcare_allowance, large_families_housing, housing_area_per_capita, consumer_price_index, primary_housing_price_index, secondary_housing_price_index, industrial_production_index, enterprises_count, crime_count, air_pollution, wastewater_discharge, mortality_rate, birth_rate_per_1000

High-frequency gold logic:

- enterprises_count: company, bank, business, enterprise, corporation, plant/project, joint venture, export contract, bankruptcy, liquidation, profit/loss, corporate lawsuit, production abroad, investment quota, business plans. Also state industrial/economic policy affecting firms.

- industrial_production_index: plant/factory, extraction, oil/gas/coal/LNG, electricity/power, industrial output, production volumes, mining, refinery, aircraft/vehicles/equipment supply, energy infrastructure, industrial accident, port/tanker/commodity logistics.

- consumer_price_index: explicit prices/inflation/tariffs/rates/duties/fuel/food/utilities, plus social/economic signals likely to affect consumer price pressure: oil/coal/gas prices, exchange/stock market, central-bank rate, import duties, sanctions/trade restrictions, weather affecting food/energy, transport disruption, industrial supply constraints, entrepreneurship/commerce signals.

- crime_count: crime, law enforcement, court/criminal case, detention, punishment, corruption/fraud, attack, shelling, drone strike, sabotage, military/security incident, conflict escalation, border/security tension, prison/convicts, genocide/crimes memory, blocked rotations or threats around strategic facilities.

- mortality_rate: killed/dead/fatalities, genocide/death memory, war/conflict casualties, severe fire/explosion/attack, social signals of death/loss/threat even when deaths are historical or implied.

- life_expectancy: injuries, illness, health risk, disease, accident victims, dangerous environmental/war/industrial event.

- hospitals: ambulance, hospitalization, emergency medical care, hospitals, inpatient care.

- qualified_doctors: doctors, physicians, medical workers, ambulance doctor, shortage/staffing of medical personnel.

- housing_area_per_capita: damaged homes, housing repair/resettlement, utilities in homes, ЖКХ debt/quality, living conditions.

- air_pollution: smoke, emissions, oil spill/fire, environmental contamination affecting air.

- wastewater_discharge: sewerage, wastewater, water supply accident, polluted water discharge, water utility failure.

- unemployment_rate: layoffs, job cuts, labor market trouble, business closures affecting jobs.

- per_capita_income: wages, salaries, pensions, household payments, benefits, income, dividends/household money.

- real_income_index: purchasing power, inflation burden, debt/credit burden, income erosion.

- migration factors: only physical people entering/leaving/moving/evacuating; do not use for trade, money flows, exports, foreign companies.

Relevance:
- Use 0.86-0.95 for a gold-like analytical signal.
- Use 0.70-0.85 for secondary but plausible signal.
- Avoid relevance below 0.60.
- If a news item has a business/industry/security/price channel, usually return all applicable core factors.

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
          "factor_id": "consumer_price_index",
          "relevance": 0.9,
          "sentiment": "neutral",
          "pressure": 0.5,
          "confidence": 0.85,
          "evidence": "short quote",
          "reason": "gold-like price pressure signal"
        }
      ]
    }
  ]
}

If no analytical signal exists, use "annotations": [].
