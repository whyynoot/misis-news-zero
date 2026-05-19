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
DEFAULT_PROMPT = ROOT / "FINAL_v4_prompt_social_signal_high_recall_ru.md"
DEFAULT_IDS_FROM = OUTPUT_DIR / "prompt_search_search_balanced_recall_v2_gemma4-e2b_5df03950348d_e98b1a3c5abc.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--limit", type=int, default=237)
    parser.add_argument("--ids-from", type=Path, default=DEFAULT_IDS_FROM)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--num-ctx", type=int, default=0, help="Optional Ollama num_ctx override; 0 keeps model/default context.")
    parser.add_argument("--model", default="", help="Override LLM model from .env for this run.")
    parser.add_argument("--text-limit", type=int, default=1800, help="Max news text characters passed to the prompt.")
    parser.add_argument("--tag", default="", help="Optional output filename tag for pilots and reruns.")
    parser.add_argument("--think", action="store_true")
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
        parsed = json.loads(cleaned[start : end + 1])
    if isinstance(parsed, list):
        return {"items": parsed}
    if isinstance(parsed, dict):
        return parsed
    raise ValueError("LLM response root must be a JSON object or list")


def normalize_response_shape(response: dict[str, Any], expected_ids: list[int]) -> dict[str, Any]:
    items = response.get("items")
    if isinstance(items, dict):
        response["items"] = [items]
        return response
    if isinstance(items, list):
        return response
    if len(expected_ids) == 1:
        if "annotations" in response:
            return {"items": [response]}
        if "factors" in response:
            return {"items": [{"news_id": str(expected_ids[0]), "factors": response.get("factors") or []}]}
        if "factor_id" in response:
            return {"items": [{"news_id": str(expected_ids[0]), "factors": [response]}]}
    return response


def item_id(raw_item: dict[str, Any]) -> str:
    value = raw_item.get("dataset_row_id")
    if value is None:
        value = raw_item.get("news_id")
    return str(value or "")


def item_factors(raw_item: dict[str, Any]) -> list[dict[str, Any]]:
    raw_factors = raw_item.get("annotations")
    if raw_factors is None:
        raw_factors = raw_item.get("factors")
    if not isinstance(raw_factors, list):
        return []
    return [factor for factor in raw_factors if isinstance(factor, dict)]


def sentiment_to_score(value: Any) -> float:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "positive":
            return 0.5
        if normalized == "negative":
            return -0.5
        if normalized == "neutral":
            return 0.0
    try:
        return max(-1.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def sentiment_to_label(value: Any, score: float) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"positive", "neutral", "negative"}:
            return normalized
    if score > 0.2:
        return "positive"
    if score < -0.2:
        return "negative"
    return "neutral"


def clamp(value: Any, minimum: float, maximum: float, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def trim_words(value: Any, limit: int = 12) -> str:
    words = str(value or "").strip().split()
    return " ".join(words[:limit])


def build_prompt(prompt_text: str, news_items: list[dict[str, Any]]) -> str:
    return (
        f"{prompt_text.strip()}\n\n"
        "## 10. Новости для разметки\n\n"
        "Разметь следующие новости. В каждом item ответа `dataset_row_id` должен совпадать с входным `dataset_row_id`.\n\n"
        f"{compact_json(news_items)}"
    )


def validate_response_schema(response: dict[str, Any], expected_ids: list[int]) -> None:
    items = response.get("items")
    if not isinstance(items, list):
        raise ValueError("classification response must contain an 'items' list")
    returned_ids = {item_id(item) for item in items if isinstance(item, dict)}
    expected = {str(expected_id) for expected_id in expected_ids}
    if expected and returned_ids.isdisjoint(expected):
        raise ValueError("classification response items do not match requested dataset_row_id values")


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

    from analyzer.factors import FACTOR_CONFIG
    from analyzer.llm_client import LLMClient, get_llm_settings
    from analyzer.llm_prompts import CLASSIFICATION_SYSTEM_PROMPT

    prompt_text = args.prompt_file.read_text(encoding="utf-8")
    df = pd.read_csv(args.dataset, encoding="utf-8-sig")
    df["dataset_row_id"] = pd.to_numeric(df["dataset_row_id"], errors="raise").astype(int)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

    factor_keys = [item["key"] for item in FACTOR_CONFIG]
    factor_names = {item["key"]: item["name"] for item in FACTOR_CONFIG}
    eval_ids = load_eval_ids(df, args.ids_from, args.limit)

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
            text=text[: max(int(args.text_limit), 200)],
            url=clean_text(row.get("url", "")),
        )

    news_by_id = {int(row["dataset_row_id"]): make_news(row) for _, row in df.iterrows()}

    def news_payload(batch_news: list[SimpleNamespace]) -> list[dict[str, Any]]:
        return [
            {
                "dataset_row_id": int(item.id),
                "date": item.published_at.date().isoformat(),
                "title": item.title,
                "text": item.text[: max(int(args.text_limit), 200)],
                "url": item.url,
            }
            for item in batch_news
        ]

    settings = get_llm_settings()
    settings = replace(
        settings,
        enabled=True,
        timeout_seconds=max(float(settings.timeout_seconds), args.timeout),
        max_tokens=max(int(settings.max_tokens), args.max_tokens),
        num_ctx=max(int(settings.num_ctx), int(args.num_ctx or 0)),
        model=args.model or settings.model,
        batch_news_size=args.batch_size,
        think=args.think,
    )
    print(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "prompt_file": str(args.prompt_file),
                "n_eval_ids": len(eval_ids),
                "gold_pairs": int(y_true.sum()),
                "mean_gold_labels": float(y_true.sum(axis=1).mean()),
                "model": settings.model,
                "think": settings.think,
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

    suffix = f"v4_prompt_think_{str(args.think).lower()}"
    if args.tag:
        suffix = f"{suffix}_{args.tag}"
    final_path = OUTPUT_DIR / f"v4_social_signal_experiment_{suffix}.csv"
    partial_path = OUTPUT_DIR / f"v4_social_signal_experiment_{suffix}.partial.csv"
    raw_path = OUTPUT_DIR / f"v4_social_signal_experiment_{suffix}.raw.jsonl"
    metrics_path = OUTPUT_DIR / f"v4_social_signal_experiment_metrics_{suffix}.csv"

    if args.force:
        for path in [final_path, partial_path, raw_path, metrics_path]:
            if path.exists():
                path.unlink()

    def normalize_rows(response: dict[str, Any], batch_news: list[SimpleNamespace], raw_text: str, status: str) -> list[dict[str, Any]]:
        raw_by_id = {item_id(raw_item): raw_item for raw_item in response.get("items", []) if isinstance(raw_item, dict)}
        rows: list[dict[str, Any]] = []
        valid_factor_keys = set(factor_keys)
        for news_item in batch_news:
            raw_item = raw_by_id.get(str(news_item.id)) or {}
            raw_factor_map: dict[str, dict[str, Any]] = {}
            for raw_factor in item_factors(raw_item):
                factor_key = str(raw_factor.get("factor_id") or "").strip()
                if factor_key in valid_factor_keys:
                    raw_factor_map[factor_key] = raw_factor
            exclude = bool(raw_item.get("exclude", False))
            exclude_reason = trim_words(raw_item.get("exclude_reason"), 24)
            no_factor_reason = trim_words(raw_item.get("no_factor_reason"), 24)
            for factor_key in factor_keys:
                raw_factor = raw_factor_map.get(factor_key)
                relevance = 0.0 if exclude else clamp(raw_factor.get("relevance") if raw_factor else 0.0, 0.0, 1.0)
                sentiment_score = sentiment_to_score(raw_factor.get("sentiment") if raw_factor else 0.0)
                sentiment_label = sentiment_to_label(raw_factor.get("sentiment") if raw_factor else None, sentiment_score)
                if raw_factor and raw_factor.get("label"):
                    sentiment_label = sentiment_to_label(raw_factor.get("label"), sentiment_score)
                if relevance <= 0:
                    sentiment_score = 0.0
                    sentiment_label = "neutral"
                rows.append(
                    {
                        "dataset_row_id": int(news_item.id),
                        "factor_key": factor_key,
                        "factor_name": factor_names[factor_key],
                        "relevance": relevance,
                        "sentiment_score": sentiment_score,
                        "pressure": 0.0 if relevance <= 0 else clamp(raw_factor.get("pressure") if raw_factor else 0.0, 0.0, 1.0),
                        "confidence": 0.0 if relevance <= 0 else clamp(raw_factor.get("confidence") if raw_factor else 0.0, 0.0, 1.0),
                        "sentiment_label": sentiment_label,
                        "is_relevant": bool(relevance >= 0.3),
                        "evidence": trim_words(raw_factor.get("evidence"), 12) if raw_factor else "",
                        "reason": trim_words(raw_factor.get("reason"), 12) if raw_factor else "",
                        "raw_factor_returned": bool(raw_factor),
                        "exclude": exclude,
                        "exclude_reason": exclude_reason,
                        "no_factor_reason": no_factor_reason,
                        "call_status": status,
                        "raw_text_chars": len(raw_text),
                    }
                )
        return rows

    def request_batch(client: LLMClient, batch_ids: list[int], status: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        batch_news = [news_by_id[item_id] for item_id in batch_ids]
        user_prompt = build_prompt(prompt_text, news_payload(batch_news))
        try:
            raw_text = client.complete(CLASSIFICATION_SYSTEM_PROMPT, user_prompt)
            response = normalize_response_shape(parse_llm_json(raw_text), batch_ids)
            validate_response_schema(response, batch_ids)
            rows = normalize_rows(response, batch_news, raw_text, status)
            row_ids = {int(row["dataset_row_id"]) for row in rows if row["raw_factor_returned"] or row["exclude"] or row["no_factor_reason"]}
            missing_ids = sorted(set(batch_ids) - {int(item_id(item)) for item in response.get("items", []) if item_id(item)})
            append_jsonl(raw_path, {"ids": batch_ids, "status": "ok", "missing_ids": missing_ids, "raw_text": raw_text})
            if missing_ids and len(batch_ids) > 1:
                good_rows = [row for row in rows if int(row["dataset_row_id"]) not in missing_ids]
                fallback_rows: list[dict[str, Any]] = []
                fallback_errors: list[dict[str, Any]] = []
                for missing_id in missing_ids:
                    one_rows, one_errors = request_batch(client, [missing_id], "single_fallback")
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
                    "raw_text": locals().get("raw_text", ""),
                },
            )
            if len(batch_ids) > 1:
                fallback_rows: list[dict[str, Any]] = []
                fallback_errors: list[dict[str, Any]] = []
                for item_id_value in batch_ids:
                    one_rows, one_errors = request_batch(client, [item_id_value], "single_after_error")
                    fallback_rows.extend(one_rows)
                    fallback_errors.extend(one_errors)
                return fallback_rows, fallback_errors
            neutral_rows = normalize_rows({"items": [{"dataset_row_id": int(batch_ids[0]), "annotations": []}]}, [news_by_id[int(batch_ids[0])]], "", "neutral_after_error")
            return neutral_rows, [{"ids": batch_ids, "error_type": type(exc).__name__, "error": str(exc)}]

    if final_path.exists() and not args.force:
        predictions = pd.read_csv(final_path)
        print(f"using cached {final_path}", flush=True)
    else:
        completed_ids: set[int] = set()
        if partial_path.exists():
            partial = pd.read_csv(partial_path, usecols=["dataset_row_id", "factor_key"])
            completed_ids = set(
                partial.groupby("dataset_row_id")["factor_key"]
                .nunique()
                .loc[lambda counts: counts == len(factor_keys)]
                .index.astype(int)
            )
        pending_ids = [item_id_value for item_id_value in eval_ids if item_id_value not in completed_ids]
        client = LLMClient(settings)
        started = time.time()
        errors: list[dict[str, Any]] = []
        print(f"v4_prompt: completed={len(completed_ids)}, pending={len(pending_ids)}", flush=True)
        for start in range(0, len(pending_ids), args.batch_size):
            batch_ids = pending_ids[start : start + args.batch_size]
            rows, batch_errors = request_batch(client, batch_ids, "batch")
            pd.DataFrame(rows).to_csv(
                partial_path,
                mode="a",
                header=not partial_path.exists(),
                index=False,
                encoding="utf-8",
            )
            errors.extend(batch_errors)
            done = len(completed_ids) + min(start + len(batch_ids), len(pending_ids))
            print(f"v4_prompt: {done:4d}/{len(eval_ids)} news; elapsed={time.time()-started:,.1f}s; errors={len(errors)}", flush=True)
        predictions = pd.read_csv(partial_path)
        predictions = predictions.drop_duplicates(["dataset_row_id", "factor_key"], keep="last")
        predictions = predictions[predictions["dataset_row_id"].isin(eval_ids) & predictions["factor_key"].isin(factor_keys)]
        predictions = predictions.sort_values(["dataset_row_id", "factor_key"])
        expected = len(eval_ids) * len(factor_keys)
        if len(predictions) != expected:
            raise RuntimeError(f"v4_prompt incomplete: rows={len(predictions)}, expected={expected}")
        predictions.to_csv(final_path, index=False, encoding="utf-8-sig")
        print(f"saved={final_path}", flush=True)

    def evaluate(pred: pd.DataFrame, threshold: float | None) -> dict[str, Any]:
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
            threshold_label = "default"
        else:
            labels = (scores >= threshold).astype(int)
            threshold_label = f"{threshold:.2f}"
        micro = precision_recall_fscore_support(y_true, labels, average="micro", zero_division=0)
        any_metric = precision_recall_fscore_support(y_true.sum(axis=1) > 0, labels.sum(axis=1) > 0, average="binary", zero_division=0)
        return {
            "variant": "v4_prompt",
            "threshold": threshold_label,
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
            "excluded_news": int(predictions.groupby("dataset_row_id")["exclude"].max().sum()) if "exclude" in predictions else 0,
            "error_rows": int(predictions["call_status"].eq("neutral_after_error").sum()) if "call_status" in predictions else 0,
        }

    thresholds = [None, 0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70]
    metrics = pd.DataFrame([evaluate(predictions, threshold) for threshold in thresholds]).sort_values(
        ["micro_f1", "micro_recall"], ascending=False
    )
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(metrics.head(20).round(4).to_string(index=False), flush=True)
    print(f"metrics saved={metrics_path}", flush=True)


if __name__ == "__main__":
    main()
