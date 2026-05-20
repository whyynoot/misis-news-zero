# direct_core_gold_calibrated_v4

Task: annotate Russian news titles/texts with DIRECT social-signal factors for this gold dataset.

Think briefly, then output JSON only. No long reasoning outside JSON.

Important calibration: in this dataset, "direct" means a direct analytical signal for the factor, not only literal statistical measurement. Keep the strong core behavior from direct_core_recall_v2, but also catch broad gold-style direct signals when the text gives a clear channel.

Allowed factor_id:
population_size, divorce_rate, marriage_rate, international_inflow, international_outflow, internal_arrivals, internal_departures, infant_mortality, life_expectancy, male_population, female_population, per_capita_income, real_income_index, unemployment_rate, living_wage, child_living_wage, hospitals, outpatient_clinics, abortions, qualified_doctors, preschool_coverage, children_benefits, maternity_capital, childcare_allowance, large_families_housing, housing_area_per_capita, consumer_price_index, primary_housing_price_index, secondary_housing_price_index, industrial_production_index, enterprises_count, crime_count, air_pollution, wastewater_discharge, mortality_rate, birth_rate_per_1000

Core direct factors:

1. crime_count
- Crime/law/security/violence: criminal case, court punishment, detention/arrest, fraud/corruption, attack, shelling, drone strike, sabotage, military/security incident, conflict around strategic facilities, prison/convicts, genocide/crimes memory, legal definition of crimes.
- Also mark for blocked rotations/threats around nuclear/strategic facilities or severe geopolitical/security escalation.

2. enterprises_count
- Company, bank, enterprise, corporation, plant as business actor, joint venture, export contract, bankruptcy, liquidation, profit/loss, corporate lawsuit, production abroad, investment quota, business plans, named business actors.
- Also mark for state industrial/economic policy affecting firms, investment in sectors, foreign orders for aircraft/equipment, business/entrepreneurship/commerce as a social-economic signal.

3. industrial_production_index
- Factory/plant, extraction, oil/gas/coal/LNG, electricity/power, production volumes, mining, refinery, aircraft/vehicles/equipment supply, energy infrastructure, industrial accident, port/tanker/commodity logistics.
- Also mark for industrial supply constraints, weather/transport disruptions affecting production, and major commodity/energy market news.

4. consumer_price_index
- Explicit price/inflation/tariff/rate/duty/fuel/food/utility cost.
- Also mark gold-style price-pressure signals: oil/gas/coal prices, central-bank rate, import duties, sanctions/trade restrictions, stock/exchange market with inflation/rate/oil context, weather that can affect food/energy, transport disruption, industrial supply constraints, consumer/commerce/entrepreneurship context.

5. mortality_rate
- Killed/dead/fatalities, death toll.
- Also mark for genocide/death memory, war/conflict casualties or losses, severe fire/explosion/attack, social/legal memory of mass death, or conflict described as a source of fatal risk even if current deaths are not counted.

Secondary direct factors:
- life_expectancy: injured, illness, health risk, disease, accident victims, dangerous event affecting health.
- hospitals: ambulance, hospitalization, emergency medical care, hospitals, inpatient care.
- qualified_doctors: doctors, physicians, medical staff, ambulance doctor.
- air_pollution: smoke, emissions, oil spill/fire, environmental contamination affecting air.
- wastewater_discharge: sewerage, wastewater, water supply accident, polluted water discharge, water utility failure.
- housing_area_per_capita: damaged homes, housing repair/resettlement, ЖКХ/housing utility living conditions.

Rare factors:
- Mark per_capita_income / real_income_index only for explicit wages, pensions, household money, debt, purchasing power, income burden.
- Mark benefits/family/demography factors only for literal benefits, births, marriages, divorces, population composition.
- Migration factors only for physical people entering/leaving/moving/evacuating. Never use migration for trade, money flows, exports, foreign companies, sanctions, diplomacy, markets, investments, ships, or aircraft restrictions.

Relevance:
- 0.90-0.95 for main direct core/gold-style signal.
- 0.75-0.89 for secondary direct signal.
- 0.60-0.74 for plausible but weaker gold-style direct signal.
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
