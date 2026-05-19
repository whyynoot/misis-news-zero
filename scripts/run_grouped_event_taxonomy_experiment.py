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
    parser.add_argument("--split", choices=["search", "full"], default="search")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=420)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--examples-per-group", type=int, default=12)
    parser.add_argument("--negatives-per-group", type=int, default=4)
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


def build_group_prompt(
    *,
    group_name: str,
    group_title: str,
    news_items: list[dict[str, Any]],
    factors_payload: list[dict[str, Any]],
    positive_examples: list[dict[str, Any]],
    negative_examples: list[dict[str, Any]],
) -> str:
    return f"""Ты классифицируешь новости только по одной группе социальных факторов.

Группа: {group_title}
group_id: {group_name}

Цель: высокий recall по ручной event-level разметке.
Лучше вернуть спорный, но реально возможный фактор с relevance=0.3, чем пропустить фактор, который поставил бы разметчик.
Но не добавляй факторы по чистой ассоциации слов: нужна связь события новости с определением фактора.

Политика разметки:
- Факторы являются event-level социальными сигналами, а не только буквальными статистическими показателями.
- Если событие новости соответствует event_definition фактора, верни этот фактор даже без статистики.
- Для одной новости в этой группе можно вернуть несколько факторов.
- relevance=0.3: слабая, но реальная связь с фактором.
- relevance=0.6: фактор заметно затронут.
- relevance=0.9 или 1.0: фактор является главным смыслом новости.
- Если по этой группе нет факторов, верни пустой список.

Факторы этой группы:
{compact_json(factors_payload)}

Позитивные примеры из gold-разметки.
Формат: title -> expected_factors_in_this_group.
Используй их как стиль разметки, но не копируй автоматически.
{compact_json(positive_examples)}

Негативные/пустые примеры.
Формат: title -> expected_factors_in_this_group=[].
{compact_json(negative_examples)}

Новости для классификации:
{compact_json(news_items)}

Верни строго JSON:
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
- Возвращай только factor_id из списка факторов этой группы.
- Не возвращай relevance=0 факторы.
- Если факторов в группе нет, верни "factors": [].
- sentiment: -1, -0.5, 0, 0.5, 1.
- label: positive, negative или neutral.
- Не добавляй текст вне JSON.
"""


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    OUTPUT_DIR.mkdir(exist_ok=True)

    from analyzer.event_factor_taxonomy import EVENT_FACTOR_GROUPS, event_factor_payload
    from analyzer.factors import FACTOR_CONFIG
    from analyzer.llm_client import LLMClient, get_llm_settings
    from analyzer.llm_prompts import CLASSIFICATION_SYSTEM_PROMPT
    from analyzer.llm_services import normalize_classification_response

    dataset_path = next((path for path in DATASET_CANDIDATES if path.exists()), None)
    if dataset_path is None:
        raise FileNotFoundError("Wide multilabel dataset not found.")

    df = pd.read_csv(dataset_path, encoding="utf-8-sig")
    row_id_col = df.columns[0]
    published_col = df.columns[7]
    title_col = df.columns[9]
    description_col = df.columns[10]
    full_text_col = df.columns[12]
    url_col = df.columns[15]
    df["dataset_row_id"] = pd.to_numeric(df[row_id_col], errors="raise").astype(int)
    df["published_at"] = pd.to_datetime(df[published_col], errors="coerce")

    factor_keys = [item["key"] for item in FACTOR_CONFIG]
    factor_by_key = {item["key"]: item for item in FACTOR_CONFIG}
    groups = EVENT_FACTOR_GROUPS
    covered = [key for group in groups.values() for key in group["factor_keys"]]
    missing = sorted(set(factor_keys) - set(covered))
    duplicates = sorted({key for key in covered if covered.count(key) > 1})
    if missing or duplicates:
        raise ValueError({"missing_factor_keys": missing, "duplicate_factor_keys": duplicates})

    y_true_df = pd.DataFrame(
        {
            key: pd.to_numeric(df[f"factor__{key}"], errors="coerce").fillna(0).astype(int).clip(0, 1).to_numpy()
            for key in factor_keys
        },
        index=df["dataset_row_id"],
    )

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
    if args.split == "search":
        ids = [item_id for item_id in load_search_ids(args.limit) if item_id in news_by_id]
    else:
        ids = sorted(news_by_id)

    example_pool_ids = [item_id for item_id in sorted(news_by_id) if item_id not in set(ids)]
    if not example_pool_ids:
        # Full evaluation cannot have a fully disjoint in-dataset example bank.
        # Keep the same examples used for search, and mark this in summary.
        search_ids = set(load_search_ids(min(300, len(news_by_id))))
        example_pool_ids = [item_id for item_id in sorted(news_by_id) if item_id not in search_ids]

    def news_payload(batch_news: list[SimpleNamespace]) -> list[dict[str, Any]]:
        payload = []
        for item in batch_news:
            payload.append(
                {
                    "news_id": str(item.id),
                    "date": item.published_at.date().isoformat(),
                    "title": item.title,
                    "summary": item.summary,
                    "text": (item.text or item.summary or item.title)[:1600],
                    "url": item.url,
                }
            )
        return payload

    rng = np.random.default_rng(42)

    def group_examples(group_keys: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        pool = y_true_df.loc[example_pool_ids, group_keys]
        positive_ids = pool[pool.sum(axis=1) > 0].index.to_list()
        negative_ids = pool[pool.sum(axis=1) == 0].index.to_list()
        rng.shuffle(positive_ids)
        rng.shuffle(negative_ids)
        positives = []
        seen_factors: set[str] = set()
        for item_id in positive_ids:
            expected = [key for key in group_keys if int(y_true_df.loc[item_id, key]) == 1]
            if not expected:
                continue
            positives.append({"title": news_by_id[int(item_id)].title, "expected_factors_in_this_group": expected})
            seen_factors.update(expected)
            if len(positives) >= args.examples_per_group and seen_factors.issuperset(
                {key for key in group_keys if int(y_true_df[key].sum()) > 0}
            ):
                break
            if len(positives) >= args.examples_per_group:
                break
        negatives = [
            {"title": news_by_id[int(item_id)].title, "expected_factors_in_this_group": []}
            for item_id in negative_ids[: args.negatives_per_group]
        ]
        return positives, negatives

    settings_base = get_llm_settings()
    settings_base = replace(
        settings_base,
        enabled=True,
        think=bool(args.think),
        timeout_seconds=max(float(settings_base.timeout_seconds), args.timeout),
        max_tokens=max(int(settings_base.max_tokens), args.max_tokens),
        batch_news_size=args.batch_size,
    )
    client = LLMClient(settings_base)
    variant = f"grouped_event_v1_{args.split}_{'think_true' if args.think else 'think_false'}"
    final_path = OUTPUT_DIR / f"{variant}.csv"
    partial_path = OUTPUT_DIR / f"{variant}.partial.csv"
    raw_path = OUTPUT_DIR / f"{variant}.raw.jsonl"
    metrics_path = OUTPUT_DIR / f"{variant}_metrics.csv"
    summary_path = OUTPUT_DIR / f"{variant}_summary.json"
    if args.force:
        for path in [final_path, partial_path, raw_path, metrics_path, summary_path]:
            if path.exists():
                path.unlink()

    print(
        json.dumps(
            {
                "variant": variant,
                "dataset": str(dataset_path),
                "split": args.split,
                "n_ids": len(ids),
                "think": args.think,
                "model": settings_base.model,
                "groups": {name: group["factor_keys"] for name, group in groups.items()},
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    factors_by_group = {
        group_name: [SimpleNamespace(id=i + 1, **factor_by_key[key]) for i, key in enumerate(group["factor_keys"])]
        for group_name, group in groups.items()
    }
    prompts_examples = {
        group_name: group_examples(group["factor_keys"])
        for group_name, group in groups.items()
    }

    def normalize_rows(
        response: dict[str, Any],
        batch_news: list[SimpleNamespace],
        group_factors: list[SimpleNamespace],
        raw_text: str,
        status: str,
        group_name: str,
    ) -> list[dict[str, Any]]:
        normalized_items, _warnings = normalize_classification_response(response, batch_news, group_factors)
        rows = []
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
                        "factor_name": next(f.name for f in group_factors if f.key == factor_payload["factor_id"]),
                        "group_name": group_name,
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

    completed_pairs: set[tuple[int, str]] = set()
    if partial_path.exists():
        partial = pd.read_csv(partial_path, usecols=["dataset_row_id", "group_name"])
        completed_pairs = set(zip(partial["dataset_row_id"].astype(int), partial["group_name"].astype(str)))

    started = time.time()
    errors: list[dict[str, Any]] = []

    def request_batch(batch_ids: list[int], group_name: str, status: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        group = groups[group_name]
        group_factors = factors_by_group[group_name]
        positive_examples, negative_examples = prompts_examples[group_name]
        batch_news = [news_by_id[item_id] for item_id in batch_ids]
        prompt = build_group_prompt(
            group_name=group_name,
            group_title=group["title"],
            news_items=news_payload(batch_news),
            factors_payload=[event_factor_payload(factor_by_key[key]) for key in group["factor_keys"]],
            positive_examples=positive_examples,
            negative_examples=negative_examples,
        )
        try:
            response, raw_text = client.complete_json(CLASSIFICATION_SYSTEM_PROMPT, prompt)
            rows = normalize_rows(response, batch_news, group_factors, raw_text, status, group_name)
            row_ids = {int(row["dataset_row_id"]) for row in rows}
            missing_ids = sorted(set(batch_ids) - row_ids)
            append_jsonl(raw_path, {"ids": batch_ids, "group": group_name, "status": "ok", "missing_ids": missing_ids, "raw_text": raw_text})
            if missing_ids and len(batch_ids) > 1:
                good_rows = [row for row in rows if int(row["dataset_row_id"]) not in missing_ids]
                fallback_rows = []
                fallback_errors = []
                for missing_id in missing_ids:
                    one_rows, one_errors = request_batch([missing_id], group_name, "single_fallback")
                    fallback_rows.extend(one_rows)
                    fallback_errors.extend(one_errors)
                return good_rows + fallback_rows, fallback_errors
            return rows, []
        except Exception as exc:
            append_jsonl(
                raw_path,
                {
                    "ids": batch_ids,
                    "group": group_name,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            if len(batch_ids) > 1:
                fallback_rows = []
                fallback_errors = []
                for item_id in batch_ids:
                    one_rows, one_errors = request_batch([item_id], group_name, "single_after_error")
                    fallback_rows.extend(one_rows)
                    fallback_errors.extend(one_errors)
                return fallback_rows, fallback_errors
            neutral_rows = []
            for factor in factors_by_group[group_name]:
                neutral_rows.append(
                    {
                        "dataset_row_id": int(batch_ids[0]),
                        "factor_key": factor.key,
                        "factor_name": factor.name,
                        "group_name": group_name,
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
            return neutral_rows, [{"ids": batch_ids, "group": group_name, "error_type": type(exc).__name__, "error": str(exc)}]

    total_units = len(ids) * len(groups)
    done_units = len(completed_pairs)
    for group_name in groups:
        pending_ids = [item_id for item_id in ids if (item_id, group_name) not in completed_pairs]
        for start in range(0, len(pending_ids), args.batch_size):
            batch_ids = pending_ids[start : start + args.batch_size]
            rows, batch_errors = request_batch(batch_ids, group_name, "batch")
            pd.DataFrame(rows).to_csv(
                partial_path,
                mode="a",
                header=not partial_path.exists(),
                index=False,
                encoding="utf-8",
            )
            errors.extend(batch_errors)
            done_units += len(batch_ids)
            print(
                f"{variant}: group={group_name} units={done_units:4d}/{total_units}; news_batch={batch_ids[0]}-{batch_ids[-1]}; elapsed={time.time()-started:,.1f}s; errors={len(errors)}",
                flush=True,
            )

    out = pd.read_csv(partial_path)
    out = out.drop_duplicates(["dataset_row_id", "factor_key"], keep="last")
    expected_pairs = {(item_id, key) for item_id in ids for key in factor_keys}
    observed_pairs = set(zip(out["dataset_row_id"].astype(int), out["factor_key"].astype(str)))
    missing_pairs = sorted(expected_pairs - observed_pairs)
    if missing_pairs:
        neutral_rows = []
        for item_id, factor_key in missing_pairs:
            neutral_rows.append(
                {
                    "dataset_row_id": item_id,
                    "factor_key": factor_key,
                    "factor_name": factor_by_key[factor_key]["name"],
                    "group_name": "",
                    "relevance": 0.0,
                    "sentiment_score": 0.0,
                    "pressure": 0.0,
                    "confidence": 0.0,
                    "sentiment_label": "neutral",
                    "is_relevant": False,
                    "evidence": "",
                    "reason": "",
                    "raw_factor_returned": False,
                    "call_status": "neutral_filled_missing_pair",
                    "raw_text_chars": 0,
                }
            )
        out = pd.concat([out, pd.DataFrame(neutral_rows)], ignore_index=True)
    out = out[out["dataset_row_id"].isin(ids) & out["factor_key"].isin(factor_keys)].sort_values(["dataset_row_id", "factor_key"])
    expected_rows = len(ids) * len(factor_keys)
    if len(out) != expected_rows:
        raise RuntimeError(f"incomplete output: rows={len(out)}, expected={expected_rows}")
    out.to_csv(final_path, index=False, encoding="utf-8-sig")

    Y = y_true_df.loc[ids, factor_keys].to_numpy()
    scores = out.pivot(index="dataset_row_id", columns="factor_key", values="relevance").reindex(index=ids, columns=factor_keys).fillna(0.0).to_numpy()
    default_labels = (
        out.pivot(index="dataset_row_id", columns="factor_key", values="is_relevant")
        .reindex(index=ids, columns=factor_keys)
        .fillna(False)
        .astype(bool)
        .to_numpy()
        .astype(int)
    )
    supported = [i for i, key in enumerate(factor_keys) if int(Y[:, i].sum()) > 0]

    def sample_f1_empty_correct(y: np.ndarray, pred: np.ndarray) -> float:
        values = []
        for true_row, pred_row in zip(y, pred):
            true_sum = int(true_row.sum())
            pred_sum = int(pred_row.sum())
            if true_sum == 0 and pred_sum == 0:
                values.append(1.0)
            elif true_sum == 0 or pred_sum == 0:
                values.append(0.0)
            else:
                tp = int(((true_row == 1) & (pred_row == 1)).sum())
                values.append(2 * tp / (true_sum + pred_sum))
        return float(np.mean(values))

    def evaluate(labels: np.ndarray, mode: str, threshold: float | None) -> dict[str, Any]:
        micro = precision_recall_fscore_support(Y, labels, average="micro", zero_division=0)
        any_metric = precision_recall_fscore_support(Y.sum(axis=1) > 0, labels.sum(axis=1) > 0, average="binary", zero_division=0)
        return {
            "variant": variant,
            "split": args.split,
            "mode": mode,
            "threshold": threshold,
            "n_news": len(ids),
            "true_positive_pairs": int(Y.sum()),
            "pred_positive_pairs": int(labels.sum()),
            "mean_pred_labels_per_news": float(labels.sum() / len(ids)),
            "factor_micro_precision": float(micro[0]),
            "factor_micro_recall": float(micro[1]),
            "factor_micro_f1": float(micro[2]),
            "factor_macro_f1_supported": float(f1_score(Y[:, supported], labels[:, supported], average="macro", zero_division=0)),
            "sample_f1_empty_correct": sample_f1_empty_correct(Y, labels),
            "any_relevant_precision": float(any_metric[0]),
            "any_relevant_recall": float(any_metric[1]),
            "any_relevant_f1": float(any_metric[2]),
            "false_relevant_news": int(((Y.sum(axis=1) == 0) & (labels.sum(axis=1) > 0)).sum()),
            "missed_all_relevant_news": int(((Y.sum(axis=1) > 0) & (labels.sum(axis=1) == 0)).sum()),
        }

    metric_rows = [evaluate(default_labels, "default", None)]
    for threshold in np.round(np.arange(0.05, 0.96, 0.05), 2):
        metric_rows.append(evaluate((scores >= threshold).astype(int), "threshold", float(threshold)))
    metrics = pd.DataFrame(metric_rows).sort_values(["factor_micro_f1", "sample_f1_empty_correct"], ascending=False)
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    # Factor-level diagnostic for the best threshold.
    best = metrics.iloc[0]
    best_labels = default_labels if best["mode"] == "default" else (scores >= float(best["threshold"])).astype(int)
    factor_rows = []
    for i, key in enumerate(factor_keys):
        pr = precision_recall_fscore_support(Y[:, i], best_labels[:, i], average="binary", zero_division=0)
        factor_rows.append(
            {
                "factor_key": key,
                "support": int(Y[:, i].sum()),
                "predicted_positive": int(best_labels[:, i].sum()),
                "precision": float(pr[0]),
                "recall": float(pr[1]),
                "f1": float(pr[2]),
            }
        )
    factor_report_path = OUTPUT_DIR / f"{variant}_factor_report.csv"
    pd.DataFrame(factor_rows).sort_values("support", ascending=False).to_csv(factor_report_path, index=False, encoding="utf-8-sig")

    summary = {
        "variant": variant,
        "dataset_path": str(dataset_path),
        "split": args.split,
        "ids_count": len(ids),
        "think": bool(args.think),
        "model": settings_base.model,
        "groups": {name: group["factor_keys"] for name, group in groups.items()},
        "example_pool_disjoint_from_eval": not bool(set(example_pool_ids) & set(ids)),
        "best_row": best.to_dict(),
        "outputs": {
            "predictions": str(final_path),
            "metrics": str(metrics_path),
            "summary": str(summary_path),
            "factor_report": str(factor_report_path),
            "raw": str(raw_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nTOP METRICS", flush=True)
    print(
        metrics[
            [
                "variant",
                "split",
                "mode",
                "threshold",
                "factor_micro_f1",
                "factor_micro_precision",
                "factor_micro_recall",
                "factor_macro_f1_supported",
                "sample_f1_empty_correct",
                "any_relevant_f1",
                "mean_pred_labels_per_news",
                "false_relevant_news",
                "missed_all_relevant_news",
            ]
        ]
        .head(20)
        .round(4)
        .to_string(index=False),
        flush=True,
    )
    print(f"predictions={final_path}", flush=True)
    print(f"metrics={metrics_path}", flush=True)
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
