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
DEFAULT_DATASET = OUTPUT_DIR / "v3_clean_broad_weak_dataset_1000.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=237)
    parser.add_argument("--ids-from", type=Path, default=OUTPUT_DIR / "prompt_search_search_balanced_recall_v2_gemma4-e2b_5df03950348d_e98b1a3c5abc.csv")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["social_signal_v3"],
        choices=["social_signal_v1", "social_signal_v2", "social_signal_v3", "social_signal_v4", "social_signal_v5"],
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=480)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--num-ctx", type=int, default=0, help="Optional Ollama num_ctx override; 0 keeps model/default context.")
    parser.add_argument("--model", default="", help="Override LLM model from .env for this run.")
    parser.add_argument("--text-limit", type=int, default=1800, help="Max news text characters passed to the prompt.")
    parser.add_argument("--tag", default="", help="Optional output filename tag for pilots and reruns.")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--print-thinking", action="store_true", help="Print per-request thinking/content counters from Ollama debug payloads.")
    parser.add_argument("--thinking-tail-chars", type=int, default=0, help="Print this many trailing thinking characters after each request; 0 disables tails.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_eval_ids(dataset: pd.DataFrame, ids_from: Path, limit: int) -> list[int]:
    if ids_from.exists():
        ids = sorted(pd.read_csv(ids_from, usecols=["dataset_row_id"])["dataset_row_id"].unique().astype(int).tolist())
    else:
        ids = sorted(dataset["dataset_row_id"].astype(int).sample(n=min(limit, len(dataset)), random_state=42).tolist())
    available = set(dataset["dataset_row_id"].astype(int).tolist())
    return [item_id for item_id in ids if item_id in available][:limit]


def build_gold_examples(df: pd.DataFrame, eval_ids: set[int], factor_keys: list[str], max_examples: int = 14) -> list[dict[str, Any]]:
    pool = df[~df["dataset_row_id"].isin(eval_ids)].copy()
    pool["gold_count"] = pool["v3_factor_count"].fillna(0).astype(int)
    preferred = pool[pool["gold_count"].between(2, 6)].sort_values(["gold_count", "dataset_row_id"], ascending=[False, True])
    rows = []
    covered: set[str] = set()
    for _, row in preferred.iterrows():
        factors = [key for key in factor_keys if int(row.get(f"factor__{key}", 0) or 0) == 1]
        if not factors:
            continue
        strengths = {
            key: clean_text(row.get(f"strength__{key}", ""))
            for key in factors
            if clean_text(row.get(f"strength__{key}", ""))
        }
        rows.append(
            {
                "title": clean_text(row["title"]),
                "expected_factors": factors,
                "why_this_many": "новость может давать несколько слабых социальных сигналов одновременно",
                "gold_strength_hint": strengths,
            }
        )
        covered.update(factors)
        if len(rows) >= max_examples and len(covered) >= 16:
            break
        if len(rows) >= max_examples + 4:
            break
    return rows[:max_examples]


def normalize_input_dataset(df: pd.DataFrame, factor_keys: list[str] | None = None) -> pd.DataFrame:
    """Support both normalized benchmark CSVs and the older wide gold export."""
    df = df.copy()
    old_column_map = {
        "dataset_row_id": "Номер строки в датасете 1000",
        "published_at": "Дата и время публикации",
        "title": "Заголовок новости",
        "full_text": "Полный текст новости",
        "url": "Ссылка на новость",
        "v3_factor_count": "Количество релевантных факторов",
    }
    for normalized, old_name in old_column_map.items():
        if normalized not in df.columns and old_name in df.columns:
            df[normalized] = df[old_name]

    if "full_text" not in df.columns:
        text_parts = [name for name in ["Описание новости", "Первый абзац новости", "Заголовок новости"] if name in df.columns]
        if text_parts:
            df["full_text"] = df[text_parts].fillna("").astype(str).agg(" ".join, axis=1)
    if "title" not in df.columns:
        df["title"] = ""
    if "url" not in df.columns:
        df["url"] = ""
    if "published_at" not in df.columns:
        df["published_at"] = ""
    if "v3_factor_count" not in df.columns and factor_keys:
        factor_cols = [f"factor__{key}" for key in factor_keys if f"factor__{key}" in df.columns]
        if factor_cols:
            df["v3_factor_count"] = df[factor_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).clip(0, 1).sum(axis=1)
    return df


def build_factor_catalog(factor_config: list[dict[str, Any]], event_taxonomy: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    catalog = []
    for factor in factor_config:
        event = event_taxonomy[factor["key"]]
        catalog.append(
            {
                "factor_id": factor["key"],
                "dashboard_name": factor["name"],
                "social_signal_meaning": event["event_name"],
                "count_as_signal_when": event["include"],
                "do_not_use_when": event["exclude"],
                "positive_signal": factor["positive_label"],
                "negative_signal": factor["negative_label"],
            }
        )
    return catalog


def shorten(value: str, limit: int = 180) -> str:
    value = clean_text(value)
    if len(value) <= limit:
        return value
    return value[:limit].rsplit(" ", 1)[0] + "..."


def compact_signal_catalog(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": item["factor_id"],
            "signal": item["social_signal_meaning"],
            "use": shorten(item["count_as_signal_when"], 170),
            "avoid": shorten(item["do_not_use_when"], 120),
        }
        for item in catalog
    ]


def tiny_signal_catalog(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": item["factor_id"],
            "hint": shorten(item["count_as_signal_when"], 95),
        }
        for item in catalog
    ]


def parse_llm_json(raw_text: str) -> dict[str, Any]:
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        fragment = cleaned[start : end + 1]
        try:
            parsed = json.loads(fragment)
        except json.JSONDecodeError:
            parsed = json.loads(repair_json_closers(fragment))
    if isinstance(parsed, list):
        return {"items": parsed}
    if isinstance(parsed, dict):
        return parsed
    raise ValueError("LLM response root must be a JSON object or list")


def repair_json_closers(text: str) -> str:
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    closers = {"{": "}", "[": "]"}
    return text + "".join(closers[ch] for ch in reversed(stack))


def normalize_response_shape(response: dict[str, Any], expected_ids: list[int]) -> dict[str, Any]:
    if "items" in response:
        items = response["items"]
        if isinstance(items, dict):
            response["items"] = [items]
        return response
    if len(expected_ids) == 1 and "factors" in response:
        return {"items": [{"news_id": str(expected_ids[0]), "factors": response.get("factors") or []}]}
    if len(expected_ids) == 1 and "factor_id" in response:
        return {"items": [{"news_id": str(expected_ids[0]), "factors": [response]}]}
    return response


def social_signal_prompt(
    *,
    variant: str,
    news_items: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> str:
    if variant == "social_signal_v5":
        return f"""Разметь одну новость для мониторинга социальных рисков.

Задача: найти ВСЕ содержательные связи news -> factor_id. Это не выбор одной рубрики.

Жесткий prior по полноте:
- В v3 gold релевантная новость обычно имеет около 3 факторов.
- Нормальный диапазон: 2-4 фактора.
- 1 фактор допустим только если новость очень узкая.
- Для комплексных тем допустимо 4-8 факторов.
- Лучше добавить потенциальную связь с relevance=0.3, чем пропустить co-label.
- Но каждый фактор должен иметь короткую причинную цепочку в reason.

relevance:
0.9 прямой/главный сигнал; 0.6 явная связь; 0.3 слабая потенциальная связь.

Быстрый checklist:
- атака/преступление -> crime_count; погибшие -> mortality_rate; раненые -> life_expectancy; больница/скорая -> hospitals.
- цены/тарифы/инфляция -> consumer_price_index; часто также real_income_index/per_capita_income.
- производство/добыча/энергетика/завод/порт/логистика -> industrial_production_index; часто enterprises_count.
- компания/банкротство/открытие/закрытие/бизнес -> enterprises_count.
- дома/ЖКХ/нет воды/нет света/расселение -> housing_area_per_capita; вода/стоки -> wastewater_discharge.
- врачи/медики/медперсонал -> qualified_doctors.
- выплаты семьям/детям -> children_benefits/living_wage/per_capita_income.
- миграция только при движении людей: въезд/выезд/эвакуация/туристы/беженцы/переселение. Не для дипломатии/экспорта/санкций.

Каталог:
{compact_json(tiny_signal_catalog(catalog))}

Новость:
{compact_json(news_items)}

Верни только JSON:
{{"items":[{{"news_id":"string","factors":[{{"factor_id":"crime_count","relevance":0.6,"sentiment":-0.5,"pressure":0.5,"confidence":0.8,"label":"negative","evidence":"фраза","reason":"атака -> crime_count"}}]}}]}}

Требования: factor_id только из каталога; factors 0..8; sentiment -1/-0.5/0/0.5/1; label positive/neutral/negative; evidence/reason до 8 слов; без markdown.
"""

    if variant in {"social_signal_v3", "social_signal_v4"}:
        examples_v3 = [
            {
                "news": "В результате удара по энергообъекту есть погибший и раненые",
                "expected": ["crime_count", "mortality_rate", "life_expectancy", "industrial_production_index"],
                "logic": "атака/удар, погибший, раненые, энергообъект",
            },
            {
                "news": "Авария на водоканале оставила без воды многоэтажные дома",
                "expected": ["housing_area_per_capita", "wastewater_discharge"],
                "logic": "условия проживания + водная/коммунальная инфраструктура",
            },
            {
                "news": "Утвержден бюджет ФОМС и Социального фонда",
                "expected": ["hospitals", "per_capita_income", "living_wage", "children_benefits"],
                "logic": "медицина, выплаты, доходы/минимальное обеспечение",
            },
            {
                "news": "Санкции затронули энергетическую компанию",
                "expected": ["industrial_production_index", "enterprises_count"],
                "logic": "промышленность и деятельность компании; consumer_price_index только если есть цены/тарифы для населения",
            },
            {
                "news": "Переговоры РФ-США без перемещения людей",
                "expected": [],
                "logic": "не ставить international_inflow/outflow только за дипломатию",
            },
        ]
        if variant == "social_signal_v4":
            count_policy = """
Количественный prior:
- В v3-разметке релевантная новость обычно содержит около 3 релевантных факторов.
- Цель не минимальный список, а полный аналитический набор связей.
- Если новость явно релевантна и ты вернул только 1 фактор, это подозрительно: перепроверь все co-label правила.
- Для обычной релевантной новости целевой диапазон 2-4 фактора.
- Для комплексной новости про атаку, аварию, бюджет, санкции, промышленность, цены, медицину или ЖКХ допустимо 4-8 факторов.
- Лучше вернуть дополнительную потенциальную связь с relevance=0.3, чем пропустить фактор, который потом может быть отфильтрован downstream-верификатором.
- Но не возвращай бессодержательные ассоциации: у каждого factor_id должна быть короткая причинная цепочка в reason.
"""
        else:
            count_policy = """
Если релевантная новость дала только 0-1 фактор, перепроверь co-labels.
"""

        return f"""Разметь ОДНУ новость как источник слабых социальных сигналов.

Цель: вернуть ВСЕ factor_id, с которыми есть содержательная связь. Не выбирай один главный класс.
Новость - единица наблюдения, фактор - единица интерпретации. Слабый сигнал сохраняем, если аналитик мог бы использовать его в агрегации.

{count_policy}

Шкала relevance:
0.9 = прямой/главный сигнал.
0.6 = явная содержательная связь.
0.3 = слабая, но реальная аналитическая связь.
Не возвращай фактор без связи.

Перед ответом молча проверь группы: безопасность/смертность, медицина, цены/доходы, производство/компании, жилье/ЖКХ, экология, семья/дети, миграция/демография.

Co-label правила:
- цены/тарифы/инфляция -> consumer_price_index; часто также real_income_index/per_capita_income.
- промышленность/добыча/энергетика/заводы/порты/производственная логистика -> industrial_production_index; часто enterprises_count.
- преступление/атака/обстрел/теракт/коррупция/мошенничество -> crime_count.
- погибшие -> mortality_rate; раненые/угроза жизни -> life_expectancy; больница/скорая/госпитализация -> hospitals.
- повреждение домов/нет воды/нет света/расселение/ЖКХ -> housing_area_per_capita; вода/стоки/водоснабжение -> wastewater_discharge.
- врачи/медики/дефицит кадров/пострадавший врач -> qualified_doctors.
- миграционные факторы только при движении людей: въезд, выезд, эвакуация, туристы, мигранты, беженцы, переселение. Не для дипломатии/экспорта/санкций.

Примеры:
{compact_json(examples_v3)}

Каталог:
{compact_json(compact_signal_catalog(catalog))}

Новость:
{compact_json(news_items)}

Верни только JSON:
{{"items":[{{"news_id":"string","factors":[{{"factor_id":"crime_count","relevance":0.6,"sentiment":-0.5,"pressure":0.5,"confidence":0.8,"label":"negative","evidence":"фраза из новости","reason":"атака -> crime_count"}}]}}]}}

Требования:
- Верни ровно один item на входной news_id.
- factors может содержать 0..10 факторов.
- factor_id только из каталога.
- sentiment: -1, -0.5, 0, 0.5, 1.
- label: positive, negative или neutral.
- evidence и reason до 10 слов.
- Никакого текста вне JSON.
"""

    if variant == "social_signal_v2":
        policy = """
Работай в режиме расширенного аналитического recall.
Не останавливайся после первого очевидного фактора: сделай мысленный checklist по всем 36 factor_id.
Верни все содержательные связи, если новость может быть использована аналитиком как слабый социальный сигнал.
Обычно у релевантной новости 2-5 факторов; у комплексной экономической, промышленной, военной, медицинской или социальной новости может быть до 8-10 факторов.
"""
    else:
        policy = """
Работай в режиме social-signal annotation.
Новость не является прямым измерением социальной реальности. Она является слабым цифровым сигналом, который потом агрегируется со статистикой и экспертной оценкой.
Поэтому нужно вернуть не один "главный класс", а все факторы, для которых есть содержательная аналитическая связь.
Обычно у релевантной новости 1-5 факторов; у комплексной новости может быть больше.
"""

    return f"""Ты размечаешь новости для аналитической системы социальных рисков по логике ВКР.

Методическая рамка:
- новость = единица наблюдения;
- фактор = единица интерпретации;
- LLM-разметка = управляемая семантическая аннотация, а не автономная экспертная истина;
- новостной сигнал не равен фактическому состоянию социальной реальности;
- слабые и контекстные сигналы нужно сохранять, потому что из них в агрегации может собираться латентный паттерн.

{policy}

Что считать факторной связью:
- Прямой сигнал: в тексте прямо описан фактор или его событие.
  Примеры: погибшие -> mortality_rate; цены/тарифы -> consumer_price_index; больница/скорая -> hospitals.
- Контекстная содержательная связь: фактор не назван как статистика, но событие влияет на социальный контекст.
  Примеры: ранение врача -> qualified_doctors и life_expectancy; повреждение домов -> housing_area_per_capita.
- Слабая потенциальная связь: новость не про социальную статистику напрямую, но может быть микросигналом для аналитика.
  Примеры: санкции против промышленной компании -> industrial_production_index и enterprises_count;
  Brent/Urals/топливо, если есть связь с расходами населения или инфляцией -> consumer_price_index;
  меры поддержки семей -> children_benefits / maternity_capital / childcare_allowance.

Не возвращай отдельное поле direct/context/weak. Используй эту шкалу только для relevance:
- relevance=0.9 или 1.0: фактор является главным или прямым смыслом новости.
- relevance=0.6: фактор явно затронут, но не единственный смысл.
- relevance=0.3: слабая, но реальная аналитическая связь, которую стоит сохранить для downstream-фильтра.
- Не возвращай фактор, если связи нет даже как слабого социального сигнала.

Важные co-label правила:
- Если событие связано с ценами/инфляцией/тарифами, часто также проверь real_income_index и per_capita_income.
- Если событие связано с промышленностью, добычей, энергетикой, портами, заводами, логистикой производства, проверь industrial_production_index и enterprises_count.
- Если есть атака, преступление, обстрел, теракт, мошенничество или коррупция, проверь crime_count; если есть погибшие, добавь mortality_rate; если есть раненые/угроза жизни, добавь life_expectancy; если есть больница/скорая, добавь hospitals.
- Если есть жилье, аварийные дома, повреждение домов, расселение или коммунальная инфраструктура, проверь housing_area_per_capita; если речь о ценах жилья, проверь primary_housing_price_index / secondary_housing_price_index.
- Если есть врачи, медики, дефицит кадров или пострадавшие медработники, проверь qualified_doctors.
- Migration factors используй только когда есть движение людей: въезд, выезд, эвакуация, туристы, мигранты, беженцы, переселение. Не используй для обычной дипломатии, экспорта, санкций и торговли без движения людей.

Контроль ложных связей:
- Не ставь consumer_price_index для любых биржевых цен автоматически. Нужна связь с расходами населения, тарифами, потребительскими товарами или инфляцией.
- Не ставь industrial_production_index для любой компании автоматически. Нужна связь с производством, добычей, выпуском, инфраструктурой или промышленной активностью.
- Не ставь family/demography только из-за слов "дети" или "семья", если нет демографического, социального или поддерживающего события.
- Не ставь mortality_rate для смерти животных, метафор или "смерти проекта".

Примеры стиля v3-разметки из gold dataset:
{compact_json(examples)}

Каталог factor_id:
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
          "relevance": 0.6,
          "sentiment": -0.5,
          "pressure": 0.5,
          "confidence": 0.75,
          "label": "negative",
          "evidence": "короткая фраза из новости",
          "reason": "погибшие -> mortality_rate"
        }}
      ]
    }}
  ]
}}

Жесткие требования:
- Верни все news_id из входа.
- В factors возвращай только factor_id из каталога.
- Не возвращай relevance=0.
- Если совсем нет социальной связи, используй "factors": [].
- sentiment: -1, -0.5, 0, 0.5, 1.
- label: positive, negative или neutral.
- evidence и reason до 12 слов.
- Никакого markdown и текста вне JSON.
"""


def validate_response_schema(response: dict[str, Any], expected_ids: list[int]) -> None:
    items = response.get("items")
    if not isinstance(items, list):
        raise ValueError("classification response must contain an 'items' list")
    returned_ids = {str(item.get("news_id") or "") for item in items if isinstance(item, dict)}
    expected = {str(item_id) for item_id in expected_ids}
    if expected and returned_ids.isdisjoint(expected):
        raise ValueError("classification response items do not match requested news_id values")


def sample_f1_empty_correct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for true_row, pred_row in zip(y_true, y_pred):
        true_sum = int(true_row.sum())
        pred_sum = int(pred_row.sum())
        tp = int(((true_row == 1) & (pred_row == 1)).sum())
        if true_sum == 0 and pred_sum == 0:
            scores.append(1.0)
        elif true_sum == 0 or pred_sum == 0:
            scores.append(0.0)
        else:
            scores.append(2 * tp / (true_sum + pred_sum))
    return float(np.mean(scores))


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    OUTPUT_DIR.mkdir(exist_ok=True)

    from analyzer.event_factor_taxonomy import EVENT_FACTOR_TAXONOMY
    from analyzer.factors import FACTOR_CONFIG
    from analyzer.llm_client import LLMClient, get_llm_settings
    from analyzer.llm_prompts import CLASSIFICATION_SYSTEM_PROMPT
    from analyzer.llm_services import normalize_classification_response

    factor_keys = [item["key"] for item in FACTOR_CONFIG]
    df = normalize_input_dataset(pd.read_csv(args.dataset, encoding="utf-8-sig"), factor_keys)
    df["dataset_row_id"] = pd.to_numeric(df["dataset_row_id"], errors="raise").astype(int)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

    factors = [SimpleNamespace(id=i + 1, **item) for i, item in enumerate(FACTOR_CONFIG)]
    factor_catalog = build_factor_catalog(FACTOR_CONFIG, EVENT_FACTOR_TAXONOMY)
    eval_ids = load_eval_ids(df, args.ids_from, args.limit)
    examples = build_gold_examples(df, set(eval_ids), factor_keys)

    y_true_by_id = pd.DataFrame(
        {
            key: pd.to_numeric(df[f"factor__{key}"], errors="coerce").fillna(0).astype(int).clip(0, 1).to_numpy()
            for key in factor_keys
        },
        index=df["dataset_row_id"],
    )
    y_true = y_true_by_id.loc[eval_ids, factor_keys].to_numpy(int)
    supported = y_true.sum(axis=0) > 0

    def make_news(row: pd.Series) -> SimpleNamespace:
        published_at = row["published_at"]
        if pd.isna(published_at):
            published_at = pd.Timestamp("1970-01-01T12:00:00+03:00")
        text = clean_text(row.get("full_text", "")) or clean_text(row.get("title", ""))
        return SimpleNamespace(
            id=int(row["dataset_row_id"]),
            published_at=published_at.to_pydatetime(),
            title=clean_text(row.get("title", "")),
            summary="",
            text=text[: max(int(args.text_limit), 200)],
            url=clean_text(row.get("url", "")),
        )

    news_by_id = {int(row["dataset_row_id"]): make_news(row) for _, row in df.iterrows()}

    def news_payload(batch_news: list[SimpleNamespace]) -> list[dict[str, Any]]:
        payload = []
        for item in batch_news:
            payload.append(
                {
                    "news_id": str(item.id),
                    "date": item.published_at.date().isoformat(),
                    "title": item.title,
                    "text": item.text[: max(int(args.text_limit), 200)],
                    "url": item.url,
                }
            )
        return payload

    settings = get_llm_settings()
    settings = replace(
        settings,
        enabled=True,
        timeout_seconds=max(float(settings.timeout_seconds), args.timeout),
        max_tokens=max(int(settings.max_tokens), args.max_tokens),
        num_ctx=max(int(settings.num_ctx), int(args.num_ctx or 0)),
        batch_news_size=args.batch_size,
        think=args.think,
        model=args.model or settings.model,
    )
    print(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "n_eval_ids": len(eval_ids),
                "gold_pairs": int(y_true.sum()),
                "mean_gold_labels": float(y_true.sum(axis=1).mean()),
                "variants": args.variants,
                "model": settings.model,
                "think": settings.think,
                "json_mode": settings.json_mode,
                "batch_size": args.batch_size,
                "timeout": settings.timeout_seconds,
                "max_tokens": settings.max_tokens,
                "num_ctx": settings.num_ctx,
                "text_limit": max(int(args.text_limit), 200),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    def normalize_rows(response: dict[str, Any], batch_news: list[SimpleNamespace], raw_text: str, status: str) -> list[dict[str, Any]]:
        normalized_items, _warnings = normalize_classification_response(response, batch_news, factors)
        rows: list[dict[str, Any]] = []
        for item_payload in normalized_items:
            news_id = int(item_payload["news_item"].id)
            for factor_payload in item_payload["factors"]:
                relevance = float(factor_payload["relevance"])
                sentiment = float(factor_payload["sentiment"])
                pressure = float(factor_payload["pressure"])
                label = str(factor_payload["label"])
                is_relevant = relevance >= 0.3
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
                        "call_status": status,
                        "raw_text_chars": len(raw_text),
                    }
                )
        return rows

    def run_variant(variant: str) -> pd.DataFrame:
        suffix = f"{variant}_think_{str(args.think).lower()}"
        if args.tag:
            suffix = f"{suffix}_{args.tag}"
        final_path = OUTPUT_DIR / f"v3_social_signal_experiment_{suffix}.csv"
        partial_path = OUTPUT_DIR / f"v3_social_signal_experiment_{suffix}.partial.csv"
        raw_path = OUTPUT_DIR / f"v3_social_signal_experiment_{suffix}.raw.jsonl"
        raw_response_path = OUTPUT_DIR / f"v3_social_signal_experiment_{suffix}.ollama_response.jsonl"
        if args.force:
            for path in [final_path, partial_path, raw_path, raw_response_path]:
                if path.exists():
                    path.unlink()
        if final_path.exists() and not args.force:
            print(f"{variant}: using cached {final_path}", flush=True)
            return pd.read_csv(final_path)

        completed_ids: set[int] = set()
        if partial_path.exists():
            completed_ids = set(pd.read_csv(partial_path, usecols=["dataset_row_id"])["dataset_row_id"].unique().astype(int).tolist())
        pending_ids = [item_id for item_id in eval_ids if item_id not in completed_ids]
        client = LLMClient(settings)
        started = time.time()
        print(f"{variant}: completed={len(completed_ids)}, pending={len(pending_ids)}", flush=True)

        errors: list[dict[str, Any]] = []

        def response_debug_payload(client: LLMClient, batch_ids: list[int], status: str, **extra: Any) -> dict[str, Any]:
            data = getattr(client, "last_response_data", None) or {}
            message = data.get("message") if isinstance(data, dict) else {}
            if not isinstance(message, dict):
                message = {}
            content = message.get("content") or data.get("response") if isinstance(data, dict) else ""
            thinking = message.get("thinking") or ""
            return {
                "ids": batch_ids,
                "status": status,
                "content_chars": len(str(content or "")),
                "thinking_chars": len(str(thinking or "")),
                "done_reason": data.get("done_reason") if isinstance(data, dict) else None,
                "total_duration": data.get("total_duration") if isinstance(data, dict) else None,
                "load_duration": data.get("load_duration") if isinstance(data, dict) else None,
                "prompt_eval_count": data.get("prompt_eval_count") if isinstance(data, dict) else None,
                "eval_count": data.get("eval_count") if isinstance(data, dict) else None,
                "raw_response": data,
                **extra,
            }

        def print_response_debug(payload: dict[str, Any]) -> None:
            if not args.print_thinking and args.thinking_tail_chars <= 0:
                return
            data = payload.get("raw_response") or {}
            message = data.get("message") if isinstance(data, dict) else {}
            if not isinstance(message, dict):
                message = {}
            thinking = str(message.get("thinking") or "")
            summary = {
                "ids": payload.get("ids"),
                "status": payload.get("status"),
                "done_reason": payload.get("done_reason"),
                "content_chars": payload.get("content_chars"),
                "thinking_chars": payload.get("thinking_chars"),
                "eval_count": payload.get("eval_count"),
                "duration_s": round(float(payload.get("total_duration") or 0) / 1_000_000_000, 1),
                "error_type": payload.get("error_type"),
                "error": payload.get("error"),
            }
            print("thinking_debug=" + json.dumps(summary, ensure_ascii=False), flush=True)
            if args.thinking_tail_chars > 0 and thinking:
                tail = re.sub(r"\s+", " ", thinking[-args.thinking_tail_chars:]).strip()
                print(f"thinking_tail ids={payload.get('ids')}: {tail}", flush=True)

        def request_batch(batch_ids: list[int], status: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            batch_news = [news_by_id[item_id] for item_id in batch_ids]
            prompt = social_signal_prompt(
                variant=variant,
                news_items=news_payload(batch_news),
                catalog=factor_catalog,
                examples=examples,
            )
            thinking_guard = os.environ.get("LLM_THINKING_GUARD", "").strip()
            if args.think and thinking_guard:
                prompt = f"{thinking_guard}\n\n{prompt}"
            try:
                raw_text = client.complete(CLASSIFICATION_SYSTEM_PROMPT, prompt)
                debug_payload = response_debug_payload(client, batch_ids, "ok")
                append_jsonl(raw_response_path, debug_payload)
                print_response_debug(debug_payload)
                response = normalize_response_shape(parse_llm_json(raw_text), batch_ids)
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
                debug_payload = response_debug_payload(
                    client,
                    batch_ids,
                    "error",
                    error_type=type(exc).__name__,
                    error=str(exc),
                    raw_text=locals().get("raw_text", ""),
                )
                append_jsonl(raw_response_path, debug_payload)
                print_response_debug(debug_payload)
                append_jsonl(
                    raw_path,
                    {
                        "ids": batch_ids,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "raw_text": locals().get("raw_text", ""),
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
            print(f"{variant}: {done:4d}/{len(eval_ids)} news; elapsed={time.time()-started:,.1f}s; errors={len(errors)}", flush=True)

        out = pd.read_csv(partial_path)
        out = out.drop_duplicates(["dataset_row_id", "factor_key"], keep="last")
        out = out[out["dataset_row_id"].isin(eval_ids) & out["factor_key"].isin(factor_keys)].sort_values(["dataset_row_id", "factor_key"])
        expected = len(eval_ids) * len(factor_keys)
        if len(out) != expected:
            raise RuntimeError(f"{variant} incomplete: rows={len(out)}, expected={expected}")
        out.to_csv(final_path, index=False, encoding="utf-8-sig")
        print(f"{variant}: saved={final_path}", flush=True)
        return out

    def evaluate(pred: pd.DataFrame, variant: str, threshold: float | None) -> dict[str, Any]:
        scores = (
            pred.pivot(index="dataset_row_id", columns="factor_key", values="relevance")
            .reindex(index=eval_ids, columns=factor_keys)
            .fillna(0.0)
            .to_numpy(float)
        )
        if threshold is None:
            labels = (
                pred.pivot(index="dataset_row_id", columns="factor_key", values="is_relevant")
                .reindex(index=eval_ids, columns=factor_keys)
                .fillna(False)
                .astype(int)
                .to_numpy()
            )
            label = "default"
        else:
            labels = (scores >= threshold).astype(int)
            label = f"{threshold:.2f}"
        micro = precision_recall_fscore_support(y_true, labels, average="micro", zero_division=0)
        any_metric = precision_recall_fscore_support(y_true.sum(axis=1) > 0, labels.sum(axis=1) > 0, average="binary", zero_division=0)
        return {
            "variant": variant,
            "threshold": label,
            "n_news": len(eval_ids),
            "gold_positive_pairs": int(y_true.sum()),
            "pred_positive_pairs": int(labels.sum()),
            "pred_mean_labels": float(labels.sum(axis=1).mean()),
            "micro_precision": float(micro[0]),
            "micro_recall": float(micro[1]),
            "micro_f1": float(micro[2]),
            "macro_f1_supported": float(f1_score(y_true[:, supported], labels[:, supported], average="macro", zero_division=0)),
            "sample_f1_empty_correct": sample_f1_empty_correct(y_true, labels),
            "any_relevant_precision": float(any_metric[0]),
            "any_relevant_recall": float(any_metric[1]),
            "any_relevant_f1": float(any_metric[2]),
            "false_relevant_news": int(((y_true.sum(axis=1) == 0) & (labels.sum(axis=1) > 0)).sum()),
            "missed_all_relevant_news": int(((y_true.sum(axis=1) > 0) & (labels.sum(axis=1) == 0)).sum()),
        }

    all_metrics = []
    thresholds = [None, 0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70]
    for variant in args.variants:
        pred = run_variant(variant)
        for threshold in thresholds:
            all_metrics.append(evaluate(pred, variant, threshold))

    metrics = pd.DataFrame(all_metrics).sort_values(["micro_f1", "micro_recall"], ascending=False)
    metrics_suffix = f"think_{str(args.think).lower()}"
    if args.tag:
        metrics_suffix = f"{metrics_suffix}_{args.tag}"
    metrics_path = OUTPUT_DIR / f"v3_social_signal_experiment_metrics_{metrics_suffix}.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(metrics.head(20).round(4).to_string(index=False), flush=True)
    print(f"metrics saved={metrics_path}", flush=True)


if __name__ == "__main__":
    main()
