# direct_core_balanced_v8

Task: annotate Russian news titles/texts with DIRECT social-signal factors.

Think briefly, then output JSON only. No prose outside JSON.

Goal: high direct micro-F1 on the v4 gold dataset. Keep the v7 gains on prices/medicine, but recover business/enterprise recall and avoid over-broad industrial/mortality labels.

Allowed factor_id:
population_size, divorce_rate, marriage_rate, international_inflow, international_outflow, internal_arrivals, internal_departures, infant_mortality, life_expectancy, male_population, female_population, per_capita_income, real_income_index, unemployment_rate, living_wage, child_living_wage, hospitals, outpatient_clinics, abortions, qualified_doctors, preschool_coverage, children_benefits, maternity_capital, childcare_allowance, large_families_housing, housing_area_per_capita, consumer_price_index, primary_housing_price_index, secondary_housing_price_index, industrial_production_index, enterprises_count, crime_count, air_pollution, wastewater_discharge, mortality_rate, birth_rate_per_1000

Core direct factors:

1. enterprises_count
- Mark for any named company, bank, business group, commercial operator, plant as business actor, export/import contract, corporate production plan, corporate investment, bankruptcy, liquidation, profit/loss, dividend, corporate lawsuit, sanctions against a company, state regulation of business activity.
- Include foreign companies when the news has Russian/economic relevance, energy, trade, sanctions, logistics, or market effect.
- Usually mark enterprises_count together with industrial_production_index when a company produces, extracts, transports, imports, exports, or supplies industrial goods, energy, aircraft, vehicles, equipment, fuel, food, raw materials.
- Do not mark enterprises_count for a purely government/cultural/legal story unless firms, markets, operators, or commercial activity are visible.

2. industrial_production_index
- Mark for plant/factory, extraction, oil/gas/coal/uranium/LNG, electricity and power infrastructure, production volumes, mining, refinery, industrial goods, aircraft/vehicles/equipment, energy, fuel, industrial exports/imports, ports, tankers, railways, airports, logistics shutdowns, production launch/stop, industrial accidents.
- Mark airport/rail/port shutdowns as industrial/logistics activity.
- Do not mark industrial_production_index for pure stock-market movement, pure finance, generic sanctions, courts, culture, or politics unless production, extraction, energy, commodity, infrastructure, transport/logistics, or supply chain is visible.

3. consumer_price_index
- Mark for explicit prices, inflation, tariffs, fuel/food/utility prices, central-bank rate, import duty, oil/gas/coal price, exchange rate, food markets.
- Also mark visible supply/logistics/production shocks if they plausibly affect consumer prices of food, fuel, energy, utilities, transport, imported goods, or consumer services.
- Do not mark symbolic/cultural/political news as CPI unless there is a visible price, rate, tariff, commodity, supply, production, logistics, trade, or consumer-cost channel.

4. crime_count
- Mark for crime, court/criminal case, criminal punishment, detention/arrest, fraud, corruption, law enforcement, prosecutor/FSB/police, prison/convicts, attack, shelling, drone strike, sabotage, terrorism, military/security incident, strategic-facility disruption.
- Also mark genocide/crimes memory or legal definition of crimes.
- Do not mark ordinary economic, stock-market, health, ecology, or transport news unless the text has security/law-enforcement/attack/operational-risk framing.

5. mortality_rate
- Mark for killed/dead/fatalities/death toll.
- Mark for genocide/death memory, war casualties/losses, severe fire/explosion/attack, fatal accident, or death-risk disaster when death/fatal hazard is central.
- Do not mark routine market/economic news as mortality_rate unless it explicitly references deaths, war losses, genocide, fatality, or severe disaster.

Medical/ecology/housing:
- life_expectancy: mass illness, poisoning, infection, injuries, health threat, dangerous event affecting health.
- hospitals: hospital, ambulance, hospitalization, emergency medical care, people brought/transferred to doctors; if a wounded person is in hospital, mark hospitals.
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
- international_inflow/outflow only for physical people entering/leaving a country: migrants, tourists/visas, refugees, prisoners crossing border.
- internal_arrivals/departures only for physical people moving/evacuating inside the country.
- Never use migration for exports/imports, foreign companies, sanctions, diplomacy, ships, aircraft restrictions, money flows, markets, investments, or generic international cooperation.

Calibrating examples:
- "Старт вечерней сессии на срочном рынке Мосбиржи задерживается" -> crime_count.
- "Аэропорты Волгограда и Краснодара приостановили работу" -> industrial_production_index.
- "Venture Global получил разрешение на допэкспорт СПГ" -> enterprises_count, industrial_production_index.
- "Российские ледоколы помогают финским судам" -> enterprises_count and industrial_production_index if the body discusses cargo/transport/navigation service.
- "Запросы из-за рубежа на поставки Ил-76" -> enterprises_count, industrial_production_index, consumer_price_index.
- "Мировые цены на продовольствие выросли" -> consumer_price_index, industrial_production_index.
- "Более 100 человек обратились в больницу с кишечной инфекцией" -> life_expectancy, hospitals.
- "Уголовное наказание за отрицание геноцида" -> crime_count, mortality_rate.

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
          "factor_id": "enterprises_count",
          "relevance": 0.9,
          "sentiment": "neutral",
          "pressure": 0.5,
          "confidence": 0.9,
          "evidence": "short quote",
          "reason": "business actor / enterprise activity -> enterprises_count"
        }
      ]
    }
  ]
}

If no direct factor exists, use "annotations": [].
