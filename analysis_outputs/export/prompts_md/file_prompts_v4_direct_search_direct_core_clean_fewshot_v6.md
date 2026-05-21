# direct_core_clean_fewshot_v6

Task: annotate Russian news with DIRECT social-signal factors for the current v4 gold dataset.

Think briefly, then output JSON only. No prose outside JSON.

Important calibration:
- In this dataset `direct` often means "direct analytical signal", not a literal Rosstat measurement.
- Do not be too conservative on the main economic/security factors. A news item usually has 1-3 direct factors.
- Return only factors that have a visible text channel in the title or body. Do not invent hidden geopolitics.

Allowed factor_id:
population_size, divorce_rate, marriage_rate, international_inflow, international_outflow, internal_arrivals, internal_departures, infant_mortality, life_expectancy, male_population, female_population, per_capita_income, real_income_index, unemployment_rate, living_wage, child_living_wage, hospitals, outpatient_clinics, abortions, qualified_doctors, preschool_coverage, children_benefits, maternity_capital, childcare_allowance, large_families_housing, housing_area_per_capita, consumer_price_index, primary_housing_price_index, secondary_housing_price_index, industrial_production_index, enterprises_count, crime_count, air_pollution, wastewater_discharge, mortality_rate, birth_rate_per_1000

Gold-style examples to imitate:
- "Старт вечерней сессии на срочном рынке Мосбиржи задерживается" -> crime_count. Operational disruption in a strategic market is a security/operational-risk signal.
- "Аэропорты Волгограда и Краснодара приостановили работу" -> industrial_production_index. Airport/transport shutdown affects industrial and logistics activity.
- "В посольстве РФ рассказали о планах Британии отказаться от российского урана к 2028 году" -> industrial_production_index, enterprises_count, air_pollution. Uranium/energy commodity and business/export channel.
- "Глава Минпромторга сообщил о запросах из-за рубежа на поставки Ил-76" -> industrial_production_index, enterprises_count, consumer_price_index. Industrial supply/export demand can be a price-pressure channel.
- "Мировые цены на продовольствие выросли" -> consumer_price_index and industrial_production_index.
- "Путин поручил разработать план подъема затонувших танкеров" -> air_pollution; industrial_production_index only if the text also stresses tanker/port/energy/logistics activity.
- "Введение уголовного наказания за отрицание геноцида..." -> crime_count and mortality_rate.
- "Более 100 человек обратились в больницу с кишечной инфекцией" -> life_expectancy and hospitals.
- "Два человека ранены при атаке БПЛА" -> crime_count and hospitals; life_expectancy may be context, not direct.

Main direct factors:

1. crime_count
- Mark for crime, law enforcement, criminal/court cases, punishment, detention/arrest, fraud, corruption, attack, shelling, drone strike, sabotage, terrorism, military/security incident, strategic-facility disruption, prison/convicts, genocide/crimes memory, legal definition of crimes.
- If there is a market/infrastructure failure with security/operational-risk framing, mark crime_count only when no better factor explains the signal.

2. enterprises_count
- Mark for companies, banks, business actors, plants as economic/legal actors, exports/contracts, bankruptcy, liquidation, profit/loss, lawsuits against companies, investment quotas, corporate plans, transport operators, business associations, named commercial organizations.
- State industrial/economic policy affecting firms also counts.
- Do not mark for every government news unless a business/enterprise/market actor is visible.

3. industrial_production_index
- Mark for plant/factory, extraction, oil/gas/coal/uranium/LNG, electricity and energy infrastructure, production volumes, mining, refinery, aircraft/vehicles/equipment, industrial exports/imports, ports, tankers, railways, airports, commodity logistics, industrial accident, production launch/stop.
- If corporate news is about producing/extracting/exporting industrial goods or energy, usually mark both enterprises_count and industrial_production_index.

4. consumer_price_index
- Mark for explicit prices, inflation, tariffs, fuel/food/utility prices, central-bank rate, duties, oil/gas/coal prices, exchange rate, food markets.
- Also mark for direct supply/production/trade/transport/weather shocks that can plausibly affect prices of consumer goods or services.
- Do not mark for generic politics without economic, trade, commodity, tariff, rate, production, logistics, or commerce channel.

5. mortality_rate
- Mark for killed/dead/fatalities/death toll.
- Also mark for genocide/death memory, war casualties/losses, severe fire/explosion/attack, and death-risk hazards where the death topic is central.
- Do not mark every injury as mortality_rate unless deaths, fatal risk, genocide, war losses, or severe disaster are central.

Secondary direct factors:
- life_expectancy: disease outbreak, mass poisoning/infection, injuries, health risk, dangerous event affecting health.
- hospitals: hospital, ambulance, hospitalization, emergency medical care, people brought/transferred to doctors.
- qualified_doctors: doctors, physicians, medical staff, ambulance doctor.
- air_pollution: smoke, emissions, oil spill/fire, fuel contamination, environmental contamination affecting air/coast.
- wastewater_discharge: sewerage, wastewater, polluted water discharge, water utility accident.
- housing_area_per_capita: damaged homes, housing repair/resettlement, housing utilities, property/living-space rules.
- unemployment_rate: layoffs, job bans, labor-market restrictions.
- per_capita_income: wages, pensions, payments, taxes, compensation, household money.

Strict migration rule:
- international_inflow/outflow only for physical people entering/leaving a country: migrants, tourists, visa flows, prisoners/refugees crossing border.
- internal_arrivals/departures only for physical people moving/evacuating inside a country.
- Never use migration for exports/imports, foreign companies, diplomacy, ships, aircraft restrictions, money flows, markets, or investments.

Relevance:
- 0.90-0.95: main direct/gold-style signal.
- 0.75-0.89: secondary direct signal.
- 0.60-0.74: visible but weaker direct signal.
- Do not use relevance below 0.60.

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

If no visible direct analytical signal exists, use "annotations": [].
