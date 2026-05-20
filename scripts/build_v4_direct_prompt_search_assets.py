from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = ROOT / "prompts" / "v4_direct_search"

FACTOR_IDS = (
    "population_size, divorce_rate, marriage_rate, international_inflow, international_outflow, "
    "internal_arrivals, internal_departures, infant_mortality, life_expectancy, male_population, "
    "female_population, per_capita_income, real_income_index, unemployment_rate, living_wage, "
    "child_living_wage, hospitals, outpatient_clinics, abortions, qualified_doctors, "
    "preschool_coverage, children_benefits, maternity_capital, childcare_allowance, "
    "large_families_housing, housing_area_per_capita, consumer_price_index, "
    "primary_housing_price_index, secondary_housing_price_index, industrial_production_index, "
    "enterprises_count, crime_count, air_pollution, wastewater_discharge, mortality_rate, "
    "birth_rate_per_1000"
)

DIRECT_RULES = """
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
"""

NEGATIVE_RULES = """
Hard negative rules:
- Do not mark international_inflow/outflow for diplomacy, sanctions, exports/imports, negotiations, documents, ships, logistics, or trade unless people physically move.
- Do not mark consumer_price_index for stock prices, IPO, company revenue, commodity quotes, or exchange news without a channel to consumer costs/tariffs/inflation.
- Do not mark industrial_production_index for a generic company, bank, investment, or financial transaction without production/extraction/infrastructure/output.
- Do not mark hospitals for water/electricity/utility accidents unless medical care, ambulance, hospitalization, or medical institutions are mentioned.
- Do not mark air_pollution for weather, temperature, rain, cold, or forecasts without pollution/smoke/emissions.
- Do not mark mortality_rate when the text explicitly says there are no casualties.
- Do not mark child/family factors from the word "family" alone.
"""

BROAD_RULES = """
Broad all-signal target:
- Return direct, context, and weak but explainable social signals.
- Keep weak signals only when evidence in the text gives a plausible analytical channel.
- Typical relevant news has around 3 factors; simple news can have 1; complex news can have 4-8.
- Use relevance 0.80-1.00 direct, 0.55-0.79 context, 0.30-0.54 weak.
"""

OUTPUT_SCHEMA = """
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
"""


def prompt(title: str, body: str) -> str:
    return f"""# {title}

You annotate Russian news for a social-signal factor dataset.

Allowed factor_id values:
{FACTOR_IDS}

{body.strip()}

{OUTPUT_SCHEMA.strip()}
"""


PROMPTS = {
    "direct_strict.md": prompt(
        "direct_strict",
        f"{DIRECT_RULES}\n{NEGATIVE_RULES}\nBe conservative: precision matters more than recall.",
    ),
    "direct_recall.md": prompt(
        "direct_recall",
        f"{DIRECT_RULES}\n{NEGATIVE_RULES}\nBe recall-oriented inside direct-only rules: if a direct event clearly touches two factors, return both.",
    ),
    "direct_checklist.md": prompt(
        "direct_checklist",
        f"{DIRECT_RULES}\nBefore JSON, silently check safety/death/health/prices/income/jobs/production/business/housing/utilities/ecology/family/migration. Return only JSON.",
    ),
    "direct_negative.md": prompt(
        "direct_negative",
        f"{DIRECT_RULES}\n{NEGATIVE_RULES}\nFinal self-check: remove every factor whose evidence is only broad context.",
    ),
    "direct_fewshot.md": prompt(
        "direct_fewshot",
        f"""{DIRECT_RULES}
Examples:
- "погибли два человека при пожаре" -> mortality_rate, crime_count if fire/attack/crime context, life_expectancy if injuries/threat.
- "тарифы ЖКХ выросли" -> consumer_price_index.
- "завод остановил производство" -> industrial_production_index, enterprises_count.
- "банк лишился лицензии" -> enterprises_count.
- "переговоры РФ-США" -> annotations=[] unless a direct factor event is stated.
{NEGATIVE_RULES}""",
    ),
    "direct_minimal.md": prompt(
        "direct_minimal",
        f"{DIRECT_RULES}\nReturn the smallest complete direct set. No weak/context factors.",
    ),
    "direct_taxonomy.md": prompt(
        "direct_taxonomy",
        f"{DIRECT_RULES}\nTreat factor names as event categories, not statistical formulas. Example: mortality_rate means deaths/fatalities; hospitals means hospitalization/ambulance/inpatient care.",
    ),
    "direct_precision.md": prompt(
        "direct_precision",
        f"{DIRECT_RULES}\n{NEGATIVE_RULES}\nOnly return a factor if you can quote evidence from the text in evidence.",
    ),
    "broad_balanced.md": prompt(
        "broad_balanced",
        f"{BROAD_RULES}\n{NEGATIVE_RULES}\nBalance recall and precision; target 2-4 factors for ordinary relevant news.",
    ),
    "broad_recall.md": prompt(
        "broad_recall",
        f"{BROAD_RULES}\nHigh recall: prefer adding a plausible weak/context factor with relevance=0.30 rather than missing an analytical signal. Still obey hard negatives.",
    ),
    "broad_weak_context.md": prompt(
        "broad_weak_context",
        f"{BROAD_RULES}\nFocus on context and weak signals too. Direct signals should still receive higher relevance. Explain the channel in reason.",
    ),
}


def main() -> None:
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in PROMPTS.items():
        (PROMPT_DIR / name).write_text(text, encoding="utf-8")
    print(f"saved {len(PROMPTS)} prompts to {PROMPT_DIR}")


if __name__ == "__main__":
    main()
