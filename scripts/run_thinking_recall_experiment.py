from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "analysis_outputs"
DATASET_CANDIDATES = [
    ROOT / "interfax_news_multilabel_factor_dataset_1000_wide.csv",
    OUTPUT_DIR / "interfax_news_multilabel_factor_dataset_1000_wide.csv",
    Path(r"C:\Users\whynot\Downloads\interfax_news_multilabel_factor_dataset_1000_wide.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=300, help="Number of search split news rows to evaluate.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["recall_max_v1_think_false", "recall_max_v1_think_true"],
        choices=[
            "recall_max_v1_think_false",
            "recall_max_v1_think_true",
            "latent_candidate_v2_think_false",
            "latent_candidate_v2_think_true",
            "latent_candidate_v3_think_false",
            "latent_candidate_v3_think_true",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=360)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_search_ids(limit: int) -> list[int]:
    existing = sorted(OUTPUT_DIR.glob("prompt_search_search_balanced_recall_v2_*.csv"))
    if existing:
        ids = sorted(pd.read_csv(existing[-1], usecols=["dataset_row_id"])["dataset_row_id"].unique().astype(int).tolist())
        return ids[:limit]

    dataset_path = next((path for path in DATASET_CANDIDATES if path.exists()), None)
    if dataset_path is None:
        raise FileNotFoundError("Wide multilabel dataset not found.")
    df = pd.read_csv(dataset_path, encoding="utf-8-sig")
    row_id_col = df.columns[0]
    return sorted(df[row_id_col].astype(int).sample(n=limit, random_state=42).tolist())


def high_recall_prompt(news_items: list[dict[str, Any]], catalog: list[dict[str, Any]], compact_json) -> str:
    return f"""Проанализируй новости по социальным факторам.

Цель этого эксперимента: высокий recall. Лучше вернуть лишний слабый фактор с relevance=0.3, чем пропустить реальный фактор.
Дальше downstream-фильтр сможет отсечь слабые связи, поэтому не будь чрезмерно консервативным.

Важная политика разметки:
- Факторы здесь являются не только буквальными статистическими показателями, а event-level социальными сигналами.
- Прямой статистики в новости может не быть. Если событие очевидно относится к фактору, фактор нужно вернуть.
- Для одной новости можно вернуть несколько факторов. Обычно 1-4, максимум 6.
- Если фактор связан только очень косвенно, ставь relevance=0.3.
- Если фактор явно затронут, но не главный смысл новости, ставь relevance=0.6.
- Если фактор является главным смыслом новости, ставь relevance=0.9 или 1.0.
- Если нет ни одного фактора даже после event-level трактовки, верни пустой список factors.

Карта решений:
- crime_count: преступления, уголовные дела, атаки, обстрелы, теракты, насилие, убийства, ранения из-за атак, мошенничество, коррупция, незаконные действия, угрозы безопасности.
- mortality_rate: гибель людей, число погибших, смерть, рост жертв, летальные исходы, смертность. Единичная смерть тоже считается социальным сигналом.
- life_expectancy: здоровье людей, тяжелые травмы, угрозы жизни, массовые заболевания, безопасность жизни, состояние пострадавших, долгосрочные риски для здоровья.
- hospitals: больницы, госпитализация, скорая помощь, работа медорганизаций, доступность медицинской помощи, больничная инфраструктура.
- qualified_doctors: врачи, медицинский персонал, дефицит/наличие врачей, пострадавшие врачи или медики.
- industrial_production_index: производство, добыча, заводы, промышленность, энергетика, остановка/запуск производств, портовая/транспортная инфраструктура если она влияет на производство.
- enterprises_count: компании, бизнес-активность, банкротства, регистрации, закрытия, корпоративные решения, деятельность организаций.
- unemployment_rate: занятость, увольнения, наем, простой работников, корпоративные отпуска, рынок труда.
- consumer_price_index: инфляция, потребительские цены, тарифы, индексация платежей, рост/снижение стоимости товаров и услуг.
- per_capita_income, real_income_index, living_wage: доходы, зарплаты, пенсии, пособия, покупательная способность, рассрочка, долговая нагрузка населения.
- children_benefits, maternity_capital, childcare_allowance, child_living_wage: детские выплаты, маткапитал, поддержка семей с детьми, расходы на детей.
- housing_area_per_capita, primary_housing_price_index, secondary_housing_price_index, large_families_housing: жилье, жилищные условия, коммунальная доступность, цены на жилье, улучшение/ухудшение условий проживания.
- air_pollution: выбросы, загрязнение воздуха, пожары/взрывы/утечки с риском загрязнения воздуха.
- wastewater_discharge: загрязнение воды, сбросы, водоснабжение, очистные сооружения, канализация, коммунальная водная инфраструктура.
- international_inflow/outflow: приезд/выезд людей, миграция, беженцы, туристы, пересечение границ людьми. Не используй для обычной торговли, экспорта, дипломатии без движения людей.
- internal_arrivals/internal_departures: внутренняя миграция, переезд людей между регионами, эвакуация/возвращение внутри страны.
- population_size, birth_rate_per_1000, infant_mortality, male_population, female_population, marriage_rate, divorce_rate, abortions: демография, рождаемость, младенческая смертность, структура населения, браки/разводы, аборты.
- outpatient_clinics: поликлиники, амбулаторная помощь, первичное звено медицины.
- preschool_coverage: детские сады, дошкольное образование, доступность мест для детей.

Контроль ложных срабатываний:
- Не возвращай international_inflow/outflow для обычных внешнеполитических переговоров, экспорта, санкций или торговли, если нет движения людей.
- Не возвращай consumer_price_index для биржевых/сырьевых цен, если нет связи с потребительскими ценами, тарифами или расходами населения.
- Не возвращай industrial_production_index для любой компании автоматически; нужна связь с производством, выпуском, добычей, инфраструктурой или промышленной активностью.
- Не возвращай marriage_rate/family только из-за слова "школьники", "семья" в нерелевантном контексте.

Оценки:
- sentiment: -1 сильный негатив, -0.5 умеренный негатив, 0 нейтрально, 0.5 умеренный позитив, 1 сильный позитив.
- pressure: социальное давление/риск от 0 до 1.
- confidence: уверенность от 0 до 1.
- label: positive, negative или neutral.

Факторы:
{compact_json(catalog)}

Новости:
{compact_json(news_items)}

Верни строго JSON по схеме:
{{
  "items": [
    {{
      "news_id": "string",
      "factors": [
        {{
          "factor_id": "crime_count",
          "relevance": 0.6,
          "sentiment": -0.5,
          "pressure": 0.5,
          "confidence": 0.8,
          "label": "negative",
          "evidence": "короткая фраза из новости",
          "reason": "до 12 слов"
        }}
      ]
    }}
  ]
}}

Требования:
- Верни все news_id из входа.
- В factors возвращай только factor_id из каталога.
- Не возвращай факторы с relevance=0.
- Если нет релевантных факторов, верни "factors": [].
- Не добавляй текст вне JSON.
- evidence и reason должны быть короткими.
"""


def latent_candidate_prompt(news_items: list[dict[str, Any]], catalog: list[dict[str, Any]], compact_json) -> str:
    return f"""Ты не финальный классификатор, а генератор кандидатов для downstream-фильтра.

Твоя цель: максимальный recall по ручной multilabel-разметке социальных факторов.
Ищи не только буквальные упоминания статистических показателей, а скрытые и косвенные event-level связи:
событие -> прямой социальный эффект -> возможный фактор дашборда.

Ключевая политика:
- Верни фактор, если новость может быть разумным сигналом для этого фактора даже без прямой статистики.
- Слабую, но реальную связь возвращай с relevance=0.3 и confidence=0.35-0.55.
- Явную связь возвращай с relevance=0.6.
- Главный смысл новости возвращай с relevance=0.9 или 1.0.
- Одна новость может иметь несколько факторов. Обычно 1-6, если событие комплексное - до 10.
- Не бойся вернуть больше кандидатов: следующий фильтр будет отсекать лишнее.
- Но не возвращай все подряд: нужна конкретная цепочка связи от события в тексте к фактору.

Как искать скрытые связи:
1. Найди событие новости: авария, смерть, преступление, медицина, цены, доходы, предприятие, ЖКХ, жилье, экология, миграция, семья, дети.
2. Для каждого события проверь downstream-последствия:
   - смерть/погибшие/летальный исход -> mortality_rate; часто также crime_count, если есть насилие, атака, уголовное событие.
   - раненые, угроза жизни, тяжелые травмы -> life_expectancy; если есть госпитализация/скорая/больница -> hospitals.
   - преступление, обстрел, теракт, мошенничество, коррупция, незаконные действия -> crime_count.
   - больницы, скорая, госпитализация, врачи, медпомощь -> hospitals / outpatient_clinics / qualified_doctors.
   - цены, тарифы, инфляция, платежи, стоимость товаров/услуг для людей -> consumer_price_index.
   - зарплаты, доходы, пенсии, пособия, покупательная способность -> per_capita_income / real_income_index / living_wage.
   - увольнения, найм, простой работников, рынок труда -> unemployment_rate.
   - заводы, добыча, производство, энергетика, порты/инфраструктура производства -> industrial_production_index.
   - компании, банкротства, открытие/закрытие организаций -> enterprises_count.
   - жилье, аварийные дома, расселение, коммунальные условия -> housing_area_per_capita; цены на жилье -> primary/secondary_housing_price_index.
   - выбросы, пожары, дым, загрязнение воздуха -> air_pollution; вода, стоки, водоснабжение, канализация -> wastewater_discharge.
   - въезд/выезд людей, беженцы, туристы, переселение -> migration factors.
   - рождение детей, браки, разводы, аборты, семьи, выплаты детям, детсады -> demographic/family factors.
3. Если связь не буквальная, в reason коротко напиши цепочку: "смерть людей -> mortality_rate" или "госпитализация -> hospitals".

Контроль ложных срабатываний:
- Не используй mortality_rate для смерти животных или метафор.
- Не используй migration для экспорта/импорта товаров, дипломатии или санкций без движения людей.
- Не используй consumer_price_index для биржевых/сырьевых цен без связи с расходами населения.
- Не используй industrial_production_index для любой компании автоматически: нужна связь с выпуском, добычей, производством или инфраструктурой.
- Не используй family/demography только из-за слов "дети", "семья", если нет демографического, социального или поддерживающего события.

Каталог факторов с event-level правилами:
{compact_json(catalog)}

Новости:
{compact_json(news_items)}

Верни строго JSON:
{{
  "items": [
    {{
      "news_id": "string",
      "factors": [
        {{
          "factor_id": "mortality_rate",
          "relevance": 0.3,
          "sentiment": -0.5,
          "pressure": 0.5,
          "confidence": 0.45,
          "label": "negative",
          "evidence": "короткая фраза из новости",
          "reason": "смерть людей -> mortality_rate"
        }}
      ]
    }}
  ]
}}

Требования:
- Верни все news_id из входа.
- В factors возвращай только factor_id из каталога.
- Не возвращай факторы с relevance=0.
- Если совсем нет социальной связи, верни "factors": [].
- sentiment: -1, -0.5, 0, 0.5, 1.
- label: positive, negative или neutral.
- evidence не длиннее 12 слов.
- reason не длиннее 12 слов.
- Не добавляй текст вне JSON.
"""


def latent_candidate_v3_prompt(news_items: list[dict[str, Any]], catalog: list[dict[str, Any]], compact_json) -> str:
    return f"""Ты классифицируешь новости по социальным факторам. Режим: HIGH-RECALL CANDIDATES.

Это не финальный фильтр. Твоя задача - вернуть как можно больше вероятно связанных факторов,
чтобы downstream-фильтр потом отсёк слабые и ложные связи.

Главное отличие от обычной классификации:
- Не требуй прямого упоминания статистического показателя.
- Ищи event-level связь: событие в новости -> социальное последствие -> фактор.
- Если связь скрытая, но разумная, верни фактор с relevance=0.3 и confidence=0.35-0.55.
- Если связь явная, верни relevance=0.6.
- Если фактор главный смысл новости, верни relevance=0.9 или 1.0.
- Для релевантной новости обычно верни 1-5 факторов, для сложной новости можно до 8.
- Для нерелевантной новости верни пустой список.

Быстрая карта скрытых связей:
- Погиб, умер, погибшие, жертвы, летальный исход -> mortality_rate.
- Раненые, тяжелые травмы, угроза жизни, здоровье пострадавших -> life_expectancy.
- Преступление, нападение, обстрел, теракт, взрыв, мошенничество, коррупция, незаконные действия -> crime_count.
- Госпитализация, скорая, больница, стационар, лечение пострадавших -> hospitals.
- Поликлиника, амбулаторная помощь, первичное звено -> outpatient_clinics.
- Врачи, медики, дефицит/работа медперсонала -> qualified_doctors.
- Цены для населения, тарифы, инфляция, платежи, стоимость товаров/услуг -> consumer_price_index.
- Доходы, зарплаты, пенсии, пособия, покупательная способность -> per_capita_income или real_income_index.
- Прожиточный минимум, базовые расходы, минимальные выплаты -> living_wage; если про детей -> child_living_wage.
- Увольнения, найм, простой, занятость, рынок труда -> unemployment_rate.
- Производство, добыча, заводы, энергетика, выпуск продукции, промышленная инфраструктура -> industrial_production_index.
- Компании, открытие/закрытие бизнеса, банкротство, число организаций -> enterprises_count.
- Жилье, расселение, аварийные дома, коммунальная доступность жилья -> housing_area_per_capita.
- Новостройки/первичка/ипотека нового жилья -> primary_housing_price_index.
- Вторичное жилье/готовое жилье -> secondary_housing_price_index.
- Выбросы, дым, пожар с загрязнением воздуха, вредные вещества -> air_pollution.
- Вода, стоки, водоснабжение, канализация, очистные сооружения -> wastewater_discharge.
- Въезд людей, мигранты, туристы, беженцы, иностранные работники/студенты -> international_inflow.
- Выезд людей, эмиграция, эвакуация за рубеж, отток граждан -> international_outflow.
- Переезд/эвакуация/размещение внутри страны или региона -> internal_arrivals/internal_departures.
- Рождение детей, рождаемость, демографические меры рождения -> birth_rate_per_1000.
- Браки/свадьбы/регистрация брака -> marriage_rate.
- Разводы/расторжение брака -> divorce_rate.
- Аборт/ограничение или доступность абортов -> abortions.
- Детские выплаты/пособия семьям с детьми -> children_benefits.
- Маткапитал/сертификаты материнского капитала -> maternity_capital.
- Пособие по уходу за ребёнком -> childcare_allowance.
- Детсады/дошкольное образование/места для дошкольников -> preschool_coverage.
- Жилье для многодетных семей -> large_families_housing.
- Численность населения/депопуляция/массовое переселение -> population_size.
- Младенцы умерли/угроза жизни младенцев -> infant_mortality.
- Мужское/женское население только если есть демографический смысл -> male_population/female_population.

Контроль ложных связей:
- Не ставь migration для экспорта, импорта, дипломатии и санкций без движения людей.
- Не ставь consumer_price_index для биржевых/сырьевых цен без связи с расходами населения.
- Не ставь industrial_production_index для любой компании без производства/добычи/инфраструктуры.
- Не ставь mortality_rate для животных, метафор или "смерти проекта".
- Не ставь family/demography только из-за слова "дети" или "семья" без социального события.

Каталог допустимых factor_id:
{compact_json(catalog)}

Новости:
{compact_json(news_items)}

Верни строго JSON с единственным верхним ключом "items":
{{
  "items": [
    {{
      "news_id": "string",
      "factors": [
        {{
          "factor_id": "mortality_rate",
          "relevance": 0.3,
          "sentiment": -0.5,
          "pressure": 0.5,
          "confidence": 0.45,
          "label": "negative",
          "evidence": "короткая фраза из новости",
          "reason": "погибшие -> mortality_rate"
        }}
      ]
    }}
  ]
}}

Жёсткие требования:
- Верни все news_id из входа.
- Не используй другие верхние ключи, кроме "items".
- Не возвращай factor_id вне каталога.
- Не возвращай факторы с relevance=0.
- Если факторов нет, используй "factors": [].
- sentiment: -1, -0.5, 0, 0.5, 1.
- label: positive, negative или neutral.
- evidence и reason до 12 слов.
- Никакого markdown и текста вне JSON.
"""


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    OUTPUT_DIR.mkdir(exist_ok=True)

    from analyzer.event_factor_taxonomy import EVENT_FACTOR_TAXONOMY
    from analyzer.factors import FACTOR_CONFIG
    from analyzer.llm_client import LLMClient, get_llm_settings
    from analyzer.llm_prompts import CLASSIFICATION_SYSTEM_PROMPT, compact_json
    from analyzer.llm_services import factor_catalog_payload, news_payload, normalize_classification_response

    dataset_path = next((path for path in DATASET_CANDIDATES if path.exists()), None)
    if dataset_path is None:
        raise FileNotFoundError("Wide multilabel dataset not found.")

    df = pd.read_csv(dataset_path, encoding="utf-8-sig")
    row_id_col = df.columns[0]
    title_col = df.columns[9]
    description_col = df.columns[10]
    full_text_col = df.columns[12]
    url_col = df.columns[15]
    published_col = df.columns[7]
    df["dataset_row_id"] = pd.to_numeric(df[row_id_col], errors="raise").astype(int)
    df["published_at"] = pd.to_datetime(df[published_col], errors="coerce")

    factor_keys = [item["key"] for item in FACTOR_CONFIG]
    factors = [SimpleNamespace(id=i + 1, **item) for i, item in enumerate(FACTOR_CONFIG)]
    factor_catalog = factor_catalog_payload(factors)
    latent_factor_catalog = [
        {
            "factor_id": factor.key,
            "dashboard_name": factor.name,
            "event_name": EVENT_FACTOR_TAXONOMY[factor.key]["event_name"],
            "count_as_signal_when": EVENT_FACTOR_TAXONOMY[factor.key]["include"],
            "do_not_use_when": EVENT_FACTOR_TAXONOMY[factor.key]["exclude"],
            "positive_signal": factor.positive_label,
            "negative_signal": factor.negative_label,
        }
        for factor in factors
    ]

    y_true_all = np.column_stack(
        [
            pd.to_numeric(df[f"factor__{key}"], errors="coerce").fillna(0).astype(int).clip(0, 1).to_numpy()
            for key in factor_keys
        ]
    )
    row_index = pd.Index(df["dataset_row_id"].to_numpy())
    y_true_by_id = pd.DataFrame(y_true_all, index=row_index, columns=factor_keys)
    supported_factor_keys = [key for key in factor_keys if int(y_true_by_id[key].sum()) > 0]

    def make_news(row: pd.Series) -> SimpleNamespace:
        published_at = row["published_at"]
        if pd.isna(published_at):
            published_at = pd.Timestamp("1970-01-01T12:00:00+03:00")
        text = clean_text(row[full_text_col]) or clean_text(row[description_col]) or clean_text(row[title_col])
        return SimpleNamespace(
            id=int(row["dataset_row_id"]),
            published_at=published_at.to_pydatetime(),
            title=clean_text(row[title_col]),
            summary=clean_text(row[description_col]),
            text=text,
            url=clean_text(row[url_col]),
        )

    news_by_id = {int(row["dataset_row_id"]): make_news(row) for _, row in df.iterrows()}
    ids = [item_id for item_id in load_search_ids(args.limit) if item_id in news_by_id]
    y_true = y_true_by_id.loc[ids, factor_keys].to_numpy()

    settings_base = get_llm_settings()
    settings_base = replace(
        settings_base,
        enabled=True,
        timeout_seconds=max(float(settings_base.timeout_seconds), args.timeout),
        max_tokens=max(int(settings_base.max_tokens), args.max_tokens),
        batch_news_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "n_ids": len(ids),
                "variants": args.variants,
                "model": settings_base.model,
                "base_url": settings_base.base_url,
                "batch_size": args.batch_size,
                "timeout": settings_base.timeout_seconds,
                "max_tokens": settings_base.max_tokens,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    def normalize_rows(response: dict[str, Any], batch_news: list[SimpleNamespace], raw_text: str, call_status: str) -> list[dict[str, Any]]:
        normalized_items, _warnings = normalize_classification_response(response, batch_news, factors)
        rows: list[dict[str, Any]] = []
        for item_payload in normalized_items:
            news_id = int(item_payload["news_item"].id)
            for factor_payload in item_payload["factors"]:
                sentiment = float(factor_payload["sentiment"])
                relevance = float(factor_payload["relevance"])
                pressure = float(factor_payload["pressure"])
                label = str(factor_payload["label"])
                is_relevant = relevance >= 0.3 and (abs(sentiment) >= 0.2 or pressure >= 0.2 or label != "neutral")
                rows.append(
                    {
                        "dataset_row_id": news_id,
                        "factor_key": factor_payload["factor_id"],
                        "factor_name": next(f.name for f in factors if f.key == factor_payload["factor_id"]),
                        "relevance": relevance,
                        "sentiment_score": sentiment,
                        "pressure": pressure,
                        "confidence": float(factor_payload["confidence"]),
                        "sentiment_label": label,
                        "is_relevant": bool(is_relevant),
                        "evidence": factor_payload["evidence"] if factor_payload["raw"] else "",
                        "reason": factor_payload["reason"] if factor_payload["raw"] else "",
                        "raw_factor_returned": bool(factor_payload["raw"]),
                        "call_status": call_status,
                        "raw_text_chars": len(raw_text),
                    }
                )
        return rows

    def run_variant(variant: str) -> pd.DataFrame:
        think = variant.endswith("_think_true")
        final_path = OUTPUT_DIR / f"thinking_recall_experiment_{variant}.csv"
        partial_path = OUTPUT_DIR / f"thinking_recall_experiment_{variant}.partial.csv"
        raw_path = OUTPUT_DIR / f"thinking_recall_experiment_{variant}.raw.jsonl"
        if args.force:
            for path in [final_path, partial_path, raw_path]:
                if path.exists():
                    path.unlink()
        if final_path.exists() and not args.force:
            print(f"{variant}: using cached {final_path}", flush=True)
            return pd.read_csv(final_path)

        completed_ids: set[int] = set()
        if partial_path.exists():
            completed_ids = set(pd.read_csv(partial_path, usecols=["dataset_row_id"])["dataset_row_id"].unique().astype(int).tolist())
        pending_ids = [item_id for item_id in ids if item_id not in completed_ids]
        client = LLMClient(replace(settings_base, think=think))
        started = time.time()
        print(f"{variant}: completed={len(completed_ids)}, pending={len(pending_ids)}, think={think}", flush=True)

        def validate_response_schema(response: dict[str, Any], expected_ids: list[int]) -> None:
            items = response.get("items")
            if not isinstance(items, list):
                raise ValueError("classification response must contain an 'items' list")
            returned_ids = {str(item.get("news_id") or "") for item in items if isinstance(item, dict)}
            if expected_ids and returned_ids.isdisjoint({str(item_id) for item_id in expected_ids}):
                raise ValueError("classification response items do not match requested news_id values")

        def request_batch(batch_ids: list[int], status: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            batch_news = [news_by_id[item_id] for item_id in batch_ids]
            if variant.startswith("latent_candidate_v3"):
                prompt = latent_candidate_v3_prompt(news_payload(batch_news), factor_catalog, compact_json)
            elif variant.startswith("latent_candidate_v2"):
                prompt = latent_candidate_prompt(news_payload(batch_news), latent_factor_catalog, compact_json)
            else:
                prompt = high_recall_prompt(news_payload(batch_news), factor_catalog, compact_json)
            try:
                response, raw_text = client.complete_json(CLASSIFICATION_SYSTEM_PROMPT, prompt)
                validate_response_schema(response, batch_ids)
                rows = normalize_rows(response, batch_news, raw_text, status)
                row_ids = {int(row["dataset_row_id"]) for row in rows}
                missing_ids = sorted(set(batch_ids) - row_ids)
                append_jsonl(raw_path, {"ids": batch_ids, "status": "ok", "missing_ids": missing_ids, "raw_text": raw_text})
                if missing_ids and len(batch_ids) > 1:
                    good_rows = [row for row in rows if int(row["dataset_row_id"]) not in missing_ids]
                    fallback_rows: list[dict[str, Any]] = []
                    fallback_errors: list[dict[str, Any]] = []
                    for missing_id in missing_ids:
                        one_rows, one_errors = request_batch([missing_id], "single_fallback")
                        fallback_rows.extend(one_rows)
                        fallback_errors.extend(one_errors)
                    return good_rows + fallback_rows, fallback_errors
                return rows, []
            except Exception as exc:
                append_jsonl(
                    raw_path,
                    {
                        "ids": batch_ids,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                if len(batch_ids) > 1:
                    fallback_rows = []
                    fallback_errors = []
                    for item_id in batch_ids:
                        one_rows, one_errors = request_batch([item_id], "single_after_error")
                        fallback_rows.extend(one_rows)
                        fallback_errors.extend(one_errors)
                    return fallback_rows, fallback_errors
                neutral_rows = []
                for factor in factors:
                    neutral_rows.append(
                        {
                            "dataset_row_id": int(batch_ids[0]),
                            "factor_key": factor.key,
                            "factor_name": factor.name,
                            "relevance": 0.0,
                            "sentiment_score": 0.0,
                            "pressure": 0.0,
                            "confidence": 0.0,
                            "sentiment_label": "neutral",
                            "is_relevant": False,
                            "evidence": "",
                            "reason": "",
                            "raw_factor_returned": False,
                            "call_status": "neutral_after_error",
                            "raw_text_chars": 0,
                        }
                    )
                return neutral_rows, [{"ids": batch_ids, "error_type": type(exc).__name__, "error": str(exc)}]

        errors: list[dict[str, Any]] = []
        for start in range(0, len(pending_ids), args.batch_size):
            batch_ids = pending_ids[start : start + args.batch_size]
            rows, batch_errors = request_batch(batch_ids, "batch")
            pd.DataFrame(rows).to_csv(
                partial_path,
                mode="a",
                header=not partial_path.exists(),
                index=False,
                encoding="utf-8",
            )
            errors.extend(batch_errors)
            done = len(completed_ids) + min(start + len(batch_ids), len(pending_ids))
            print(
                f"{variant}: {done:4d}/{len(ids)} news; elapsed={time.time()-started:,.1f}s; errors={len(errors)}",
                flush=True,
            )

        out = pd.read_csv(partial_path)
        out = out.drop_duplicates(["dataset_row_id", "factor_key"], keep="last")
        out = out[out["dataset_row_id"].isin(ids) & out["factor_key"].isin(factor_keys)].sort_values(["dataset_row_id", "factor_key"])
        expected = len(ids) * len(factor_keys)
        if len(out) != expected:
            raise RuntimeError(f"{variant} incomplete: rows={len(out)}, expected={expected}")
        out.to_csv(final_path, index=False, encoding="utf-8-sig")
        print(f"{variant}: saved={final_path}", flush=True)
        return out

    def per_sample_f1_empty_correct(y: np.ndarray, p: np.ndarray) -> float:
        scores = []
        for true_row, pred_row in zip(y, p):
            true_sum = int(true_row.sum())
            pred_sum = int(pred_row.sum())
            if true_sum == 0 and pred_sum == 0:
                scores.append(1.0)
            elif true_sum == 0 or pred_sum == 0:
                scores.append(0.0)
            else:
                tp = int(((true_row == 1) & (pred_row == 1)).sum())
                scores.append(2 * tp / (true_sum + pred_sum))
        return float(np.mean(scores))

    def evaluate_prediction(pred: pd.DataFrame, variant: str, mode: str, threshold: float | None) -> dict[str, Any]:
        pred = pred.copy()
        pivot_scores = (
            pred.pivot(index="dataset_row_id", columns="factor_key", values="relevance")
            .reindex(index=ids, columns=factor_keys)
            .fillna(0.0)
        )
        if mode == "default":
            labels = (
                pred.pivot(index="dataset_row_id", columns="factor_key", values="is_relevant")
                .reindex(index=ids, columns=factor_keys)
                .fillna(False)
                .astype(int)
                .to_numpy()
            )
        else:
            labels = (pivot_scores.to_numpy() >= float(threshold)).astype(int)

        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, labels, average="micro", zero_division=0)
        supported_indices = [factor_keys.index(key) for key in supported_factor_keys]
        macro_supported = f1_score(
            y_true[:, supported_indices],
            labels[:, supported_indices],
            average="macro",
            zero_division=0,
        )
        any_true = y_true.sum(axis=1) > 0
        any_pred = labels.sum(axis=1) > 0
        any_p, any_r, any_f1, _ = precision_recall_fscore_support(any_true, any_pred, average="binary", zero_division=0)
        pred_positive_pairs = int(labels.sum())
        true_positive_pairs = int(y_true.sum())
        return {
            "variant": variant,
            "mode": mode,
            "threshold": threshold,
            "n_news": len(ids),
            "true_positive_pairs": true_positive_pairs,
            "pred_positive_pairs": pred_positive_pairs,
            "mean_pred_labels_per_news": pred_positive_pairs / len(ids),
            "factor_micro_f1": micro_f1,
            "factor_micro_precision": micro_p,
            "factor_micro_recall": micro_r,
            "factor_macro_f1_supported": macro_supported,
            "sample_f1_empty_correct": per_sample_f1_empty_correct(y_true, labels),
            "any_relevant_f1": any_f1,
            "any_relevant_precision": any_p,
            "any_relevant_recall": any_r,
        }

    prediction_by_variant = {variant: run_variant(variant) for variant in args.variants}
    rows = []
    for variant, pred in prediction_by_variant.items():
        rows.append(evaluate_prediction(pred, variant, "default", None))
        for threshold in np.round(np.arange(0.10, 0.96, 0.05), 2):
            rows.append(evaluate_prediction(pred, variant, "threshold", float(threshold)))

    metrics = pd.DataFrame(rows).sort_values(["factor_micro_f1", "sample_f1_empty_correct"], ascending=False)
    metrics_path = OUTPUT_DIR / "thinking_recall_experiment_metrics.csv"
    summary_path = OUTPUT_DIR / "thinking_recall_experiment_summary.json"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    baseline_path = OUTPUT_DIR / "final_multilabel_prompt_search_metrics.csv"
    baseline = None
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        baseline = baseline_df.head(8).to_dict(orient="records")

    summary = {
        "dataset_path": str(dataset_path),
        "ids_count": len(ids),
        "model": settings_base.model,
        "variants": args.variants,
        "best_row": metrics.iloc[0].to_dict(),
        "baseline_prompt_search_top_rows": baseline,
        "outputs": {
            "metrics": str(metrics_path),
            "summary": str(summary_path),
            **{variant: str(OUTPUT_DIR / f"thinking_recall_experiment_{variant}.csv") for variant in args.variants},
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nTOP METRICS", flush=True)
    print(
        metrics[
            [
                "variant",
                "mode",
                "threshold",
                "factor_micro_f1",
                "factor_micro_precision",
                "factor_micro_recall",
                "factor_macro_f1_supported",
                "sample_f1_empty_correct",
                "any_relevant_f1",
                "mean_pred_labels_per_news",
            ]
        ]
        .head(20)
        .round(4)
        .to_string(index=False),
        flush=True,
    )
    print(f"metrics={metrics_path}", flush=True)
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
