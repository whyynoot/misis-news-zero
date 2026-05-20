# direct_core_targeted_v7

Task: annotate Russian news titles/texts with DIRECT social-signal factors.

Think briefly, then output JSON only. No prose outside JSON.

Goal: maximize direct micro-F1 on the v4 gold dataset. Use direct analytical links, but avoid broad weak/context links.

Allowed factor_id:
population_size, divorce_rate, marriage_rate, international_inflow, international_outflow, internal_arrivals, internal_departures, infant_mortality, life_expectancy, male_population, female_population, per_capita_income, real_income_index, unemployment_rate, living_wage, child_living_wage, hospitals, outpatient_clinics, abortions, qualified_doctors, preschool_coverage, children_benefits, maternity_capital, childcare_allowance, large_families_housing, housing_area_per_capita, consumer_price_index, primary_housing_price_index, secondary_housing_price_index, industrial_production_index, enterprises_count, crime_count, air_pollution, wastewater_discharge, mortality_rate, birth_rate_per_1000

Calibrated direct rules:

1. crime_count
- Mark for crime, court/criminal case, criminal punishment, detention/arrest, fraud, corruption, law enforcement, prosecutor/FSB/police, prison/convicts, attack, shelling, drone strike, sabotage, terrorism, military/security incident, strategic-facility disruption.
- Also mark for genocide/crimes memory or legal definition of crimes.
- Do not mark ordinary economic, stock-market, health, ecology, or transport news unless the text has security/law-enforcement/attack/operational-risk framing.

2. enterprises_count
- Mark for company, bank, business, enterprise, corporation, plant as economic actor, transport operator, export/import contract, bankruptcy, liquidation, profit/loss, corporate lawsuit, investment quota, corporate production/business plan, named commercial organization.
- State policy counts only when it directly affects firms/markets/enterprise activity.
- Do not mark a factor merely because some organization is mentioned; there must be a business/market/enterprise channel.

3. industrial_production_index
- Mark for plant/factory, extraction, oil/gas/coal/uranium/LNG, electricity/power generation, production volume, mining, refinery, industrial goods, aircraft/vehicles/equipment, energy infrastructure, industrial exports/imports, ports, tankers, railways, airports, logistics shutdowns, production launch/stop.
- Mark for commodity/resource policy if it affects extraction, production, logistics, or industrial supply.
- If a corporate news item is about producing/extracting/exporting industrial goods or energy, usually mark both enterprises_count and industrial_production_index.

4. consumer_price_index
- Mark for explicit prices, inflation, tariffs, fuel/food/utility prices, central-bank rate, import duty, oil/gas/coal price, exchange rate, food markets.
- Also mark direct supply/logistics/production shocks if they clearly affect consumer prices: food, fuel, energy, tariffs, transport costs, commodity prices.
- Do not mark symbolic/cultural/political news as CPI unless the text has a visible price, rate, tariff, commodity, supply, production, logistics, trade, or consumer-cost channel.

5. mortality_rate
- Mark for killed/dead/fatalities/death toll.
- Also mark genocide/death memory, war casualties/losses, severe fire/explosion/attack, and death-risk disasters when the death topic or fatal hazard is central.
- Do not mark routine market/economic news as mortality_rate unless it explicitly references deaths, war losses, genocide, fatality, or severe disaster.

Medical/ecology/housing:
- life_expectancy: mass illness, poisoning, infection, injuries, health threat, dangerous event affecting health.
- hospitals: hospital, ambulance, hospitalization, emergency medical care, people brought/transferred to doctors.
- qualified_doctors: doctors, physicians, medical staff, ambulance doctor.
- air_pollution: smoke, emissions, oil spill/fire, fuel contamination, dirty air, environmental contamination of coast/air.
- wastewater_discharge: sewage, wastewater, polluted water discharge, water utility accident.
- housing_area_per_capita: damaged homes, housing repair/resettlement, living space, housing utilities, property/housing transaction rules.

Income/labor/demography:
- unemployment_rate: layoffs, job loss, labor bans/restrictions, unemployment.
- per_capita_income: wages, pensions, payments, taxes, compensation, household money.
- real_income_index: purchasing power or income explicitly linked to prices/inflation/debt burden.
- family/demographic factors require literal mention of births, marriages, divorces, children benefits, maternity capital, abortions, infant deaths, population size.

Strict migration rule:
- international_inflow/outflow only for physical people entering/leaving a country: migrants, tourists/visas, refugees, released prisoners crossing border.
- internal_arrivals/departures only for physical people moving/evacuating inside the country.
- Never use migration for exports/imports, foreign companies, sanctions, diplomacy, ships, aircraft restrictions, money flows, markets, investments, or generic international cooperation.

Positive examples:
- "Старт вечерней сессии на срочном рынке Мосбиржи задерживается" -> crime_count.
- "Аэропорты Волгограда и Краснодара приостановили работу" -> industrial_production_index.
- "Запросы из-за рубежа на поставки Ил-76" -> industrial_production_index, enterprises_count, consumer_price_index.
- "Мировые цены на продовольствие выросли" -> consumer_price_index, industrial_production_index.
- "Более 100 человек обратились в больницу с кишечной инфекцией" -> life_expectancy, hospitals.
- "Уголовное наказание за отрицание геноцида" -> crime_count, mortality_rate.

Negative examples:
- Do not use international_inflow/outflow for airport shutdown, queues, exports, foreign firms, or sanctions unless people crossing a country border are the topic.
- Do not use consumer_price_index for monuments/culture unless commerce/price channel is explicit.
- Do not use mortality_rate for ordinary stock-market news unless death/war/genocide/fatal risk is explicit.
- Do not use hospitals for "people saved" unless doctors/hospital/ambulance/medical transfer is explicit.

Relevance:
- 0.90-0.95 for main direct signal.
- 0.75-0.89 for secondary direct signal.
- 0.60-0.74 for visible but weaker direct signal.
- Do not use relevance below 0.60.
- Typical relevant news has 1-3 direct factors; complex industrial/security/business/price news can have 3-5.

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

If no direct factor exists, use "annotations": [].
