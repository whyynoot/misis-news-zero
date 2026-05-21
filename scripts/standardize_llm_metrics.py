from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis_outputs"

SCOPES = ["all", "direct", "context", "weak", "direct_context"]
SCOPE_METRICS = {
    "micro_precision": "precision",
    "micro_recall": "recall",
    "micro_f1": "micro_f1",
    "macro_f1_supported": "macro_f1_supported",
    "sample_f1_empty_correct": "sample_f1",
    "sample_f1": "sample_f1",
    "any_precision": "any_precision",
    "any_recall": "any_recall",
    "any_f1": "any_f1",
    "gold_pairs": "gold_pairs",
    "pred_pairs": "pred_pairs",
    "tp_pairs": "tp_pairs",
    "fp_pairs_vs_scope": "fp_pairs",
    "fn_pairs": "fn_pairs",
    "mean_pred_labels": "mean_pred_labels",
    "gold_news": "gold_news",
    "pred_any_news": "pred_any_news",
    "true_hit_news_coverage": "true_hit_news_coverage",
    "false_relevant_news": "false_relevant_news",
    "missed_all_scope_news": "missed_all_news",
}
PROMPT_VERSION_TOKENS = [
    "social_signal_v9_f1_balanced",
    "high_recall_minimal_v3",
    "high_recall_minimal_v2",
    "high_recall_minimal_v1",
    "direct_business_migration_fixed",
    "direct_core_gold_calibrated_v4",
    "direct_core_clean_fewshot_v6",
    "direct_core_gold_fewshot_v5",
    "direct_core_balanced_v8",
    "direct_core_targeted_v7",
    "direct_core_filtered_v3",
    "direct_core_recall_v2",
    "direct_brief_thinking",
    "gold_mimic_broad_v1",
    "social-risk-classification-v1",
    "balanced_recall_v2",
    "few_shot_major_v3",
    "hard_negative_v2",
    "primary_first_v2",
    "latent_candidate_v2",
    "latent_candidate_v3",
    "grouped_event_v1",
    "recall_max_v1",
    "direct_checklist",
    "direct_precision",
    "direct_negative",
    "direct_taxonomy",
    "direct_fewshot",
    "direct_minimal",
    "direct_recall",
    "direct_strict",
    "broad_balanced",
    "broad_recall",
    "broad_weak_context",
    "production_v1",
    "v4_prompt",
    "social_signal_v5",
    "social_signal_v4",
    "social_signal_v3",
    "social_signal_v2",
    "social_signal_v1",
]
PRIMARY_COLUMNS = [
    "run_id",
    "run_family",
    "model",
    "prompt_version",
    "prompt_reported",
    "thinking",
    "context_window_tokens",
    "max_output_tokens",
    "text_limit_chars_or_tokens",
    "batch_size",
    "requested_n_news",
    "is_partial",
    "threshold",
    "selection_metric",
    "selection_score",
    "wide_n_news",
    "wide_all_precision",
    "wide_all_recall",
    "wide_all_micro_f1",
    "wide_direct_precision",
    "wide_direct_recall",
    "wide_direct_micro_f1",
    "wide_context_precision",
    "wide_context_recall",
    "wide_context_micro_f1",
    "wide_weak_precision",
    "wide_weak_recall",
    "wide_weak_micro_f1",
    "wide_direct_context_precision",
    "wide_direct_context_recall",
    "wide_direct_context_micro_f1",
    "wide_pressure_mae_on_predicted_true_pairs",
    "wide_pressure_rmse_on_predicted_true_pairs",
    "wide_sentiment_macro_f1_on_predicted_true_pairs",
    "v4_strength_n_news",
    "v4_strength_all_precision",
    "v4_strength_all_recall",
    "v4_strength_all_micro_f1",
    "v4_strength_direct_precision",
    "v4_strength_direct_recall",
    "v4_strength_direct_micro_f1",
    "v4_strength_context_precision",
    "v4_strength_context_recall",
    "v4_strength_context_micro_f1",
    "v4_strength_weak_precision",
    "v4_strength_weak_recall",
    "v4_strength_weak_micro_f1",
    "old_gold_n_news",
    "old_gold_precision",
    "old_gold_recall",
    "old_gold_micro_f1",
    "old_gold_macro_f1_supported",
    "old_gold_any_relevant_f1",
]
INT_COLUMNS = [
    "context_window_tokens",
    "max_output_tokens",
    "text_limit_chars_or_tokens",
    "batch_size",
    "requested_n_news",
    "wide_n_news",
    "wide_n_rows",
    "v4_strength_n_news",
    "v4_strength_n_rows",
    "old_gold_n_news",
    "wide_all_gold_pairs",
    "wide_all_pred_pairs",
    "v4_strength_all_gold_pairs",
    "v4_strength_all_pred_pairs",
    "old_gold_gold_pairs",
    "old_gold_pred_pairs",
]


def normalize_threshold(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if text in {"", "nan", "None"}:
        return ""
    if text in {"default", "ensemble"}:
        return text
    try:
        return f"{float(text):.2f}"
    except ValueError:
        return text


def first_present(row: pd.Series, columns: list[str], default: Any = "") -> Any:
    for col in columns:
        if col not in row:
            continue
        value = row[col]
        if pd.notna(value) and str(value) != "":
            return value
    return default


def parse_k_number(match: re.Match[str] | None) -> int | None:
    if not match:
        return None
    value = int(match.group(1))
    return value * 1024 if match.group(2).lower() == "k" else value


def parse_model(name: str, reported: str = "") -> str:
    lowered = name.lower().replace("-", "_")
    if "manual_dataset_bert" in lowered:
        return "rubert-nli-zero-shot"
    if "qwen36_35b_iq4" in lowered:
        return "qwen3.6:35b-iq4"
    if "qwen36_35b_iq3" in lowered or "qwen3.6_35b" in lowered:
        return "qwen3.6:35b-iq3"
    if "qwen36_27b" in lowered or "qwen3.6_27b" in lowered:
        return "qwen3.6:27b"
    if "qwen35_27b" in lowered or "qwen3.5_27b" in lowered:
        return "qwen3.5:27b"
    if "qwen35_9b" in lowered or "qwen3.5_9b" in lowered:
        return "qwen3.5:9b"
    if "qwen3_14b" in lowered:
        return "qwen3:14b"
    if "gemma4_26b" in lowered or "gemma4_27b" in lowered:
        return "gemma4:26b"
    if "gemma4_e4b" in lowered or "gemma4:e4b" in lowered:
        return "gemma4:e4b"
    if "gemma4_e2b" in lowered or "gemma4:e2b" in lowered:
        return "gemma4:e2b"
    if "gemma3n_e4b" in lowered or "gemma3n:e4b" in lowered:
        return "gemma3n:e4b"
    if reported and reported not in {"unknown", "gemma"}:
        return reported
    if lowered.startswith(("v3_social_signal_experiment_", "v4_social_signal_experiment_", "thinking_recall_experiment_", "grouped_event_")):
        return "gemma4:e2b"
    return reported if reported else "unknown"


def parse_prompt_version(name: str, reported: str = "") -> str:
    lowered = name.lower()
    if "manual_dataset_bert" in lowered:
        return "bert_full_text_baseline"
    for token in PROMPT_VERSION_TOKENS:
        if token in lowered:
            return token.replace("-", "_")
    if reported and reported != "unknown":
        return str(reported).replace("-", "_")
    return "unknown"


def parse_prompt_family(name: str, prompt_version: str) -> str:
    lowered = name.lower()
    if prompt_version == "bert_full_text_baseline":
        return "bert_baseline"
    if prompt_version.startswith("direct_") or prompt_version.startswith("gold_mimic") or prompt_version.startswith("broad_"):
        return "direct_search"
    if prompt_version.startswith("social_signal"):
        return "social_signal"
    if lowered.startswith("prompt_search_") or "balanced_recall" in prompt_version:
        return "prompt_search"
    if lowered.startswith("thinking_recall_experiment_") or prompt_version.startswith("latent_candidate") or prompt_version == "recall_max_v1":
        return "thinking_recall"
    if lowered.startswith("grouped_event_") or prompt_version.startswith("grouped_event"):
        return "grouped_event"
    if "social_risk_classification" in prompt_version:
        return "manual_llm"
    return "unknown"


def parse_run_family(name: str, reported: str = "") -> str:
    lowered = name.lower()
    if lowered.startswith("manual_dataset_bert"):
        return "bert_baseline"
    if lowered.startswith("v4_social_signal_experiment_"):
        return "v4_social_signal"
    if lowered.startswith("v3_social_signal_experiment_"):
        return "v3_social_signal"
    if lowered.startswith("prompt_search_") or lowered.startswith("manual_dataset_prompt_search"):
        return "prompt_search"
    if lowered.startswith("thinking_recall_experiment_"):
        return "thinking_recall"
    if lowered.startswith("grouped_event_"):
        return "grouped_event"
    if lowered.startswith("manual_dataset_llm_predictions"):
        return "manual_llm"
    return reported if reported else "unknown"


def parse_bool_thinking(name: str) -> str:
    lowered = name.lower()
    if "think_false" in lowered or "nothink" in lowered:
        return "false"
    if "think_true" in lowered or re.search(r"(^|_)think($|_ctx|_b|_n|_tok)", lowered):
        return "true"
    return "unknown"


def parse_int_token(name: str, prefix: str) -> int | None:
    return parse_k_number(re.search(rf"{prefix}(\d+)(k?)", name.lower()))


def parse_text_limit(name: str) -> int | None:
    lowered = name.lower()
    return parse_k_number(re.search(r"(?:txt|text)(\d+)(k?)", lowered))


def parse_batch_size(name: str) -> int | None:
    match = re.search(r"_b(\d+)(?:_|$|\s|\.)", name.lower())
    return int(match.group(1)) if match else None


def parse_requested_n(name: str) -> int | None:
    match = re.search(r"_n(\d+)(?:_|$)", name.lower())
    return int(match.group(1)) if match else None


def load_scoped_metrics(path: Path, prefix: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "run_id" not in df.columns or "scope" not in df.columns:
        return pd.DataFrame()
    df["run_id"] = df["run_id"].astype(str)
    df["threshold_key"] = df["threshold"].map(normalize_threshold)
    rows: list[dict[str, Any]] = []
    group_cols = ["run_id", "threshold_key"]
    for (run_id, threshold), group in df.groupby(group_cols, dropna=False):
        first = group.iloc[0]
        row: dict[str, Any] = {
            "run_id": run_id,
            "threshold": threshold,
            f"{prefix}_source_file": first.get("file_name", ""),
            f"{prefix}_run_family_reported": first.get("family", ""),
            f"{prefix}_model_reported": first.get("model", ""),
            f"{prefix}_prompt_reported": first.get("prompt", ""),
            f"{prefix}_is_partial": bool(first.get("is_partial", str(run_id).endswith(".partial"))),
            f"{prefix}_n_news": int(first.get("n_news", 0)) if pd.notna(first.get("n_news", None)) else None,
            f"{prefix}_n_rows": int(first.get("n_rows", 0)) if pd.notna(first.get("n_rows", None)) else None,
            f"{prefix}_complete_36x_rows": first.get("complete_36x_rows", ""),
            f"{prefix}_json_errors": first.get("json_errors", ""),
        }
        for scope in SCOPES:
            scope_df = group[group["scope"].astype(str).eq(scope)]
            if scope_df.empty:
                continue
            scope_row = scope_df.iloc[0]
            for source_col, target_col in SCOPE_METRICS.items():
                if source_col in scope_row:
                    row[f"{prefix}_{scope}_{target_col}"] = scope_row[source_col]
        rows.append(row)
    return pd.DataFrame(rows)


def load_old_gold_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "run_name" not in df.columns:
        return pd.DataFrame()
    df["run_id"] = df["run_name"].astype(str)
    df["threshold"] = df["threshold"].map(normalize_threshold)
    rows = []
    for _, item in df.iterrows():
        rows.append(
            {
                "run_id": item["run_id"],
                "threshold": item["threshold"],
                "old_gold_source_file": item.get("source_file", ""),
                "old_gold_run_family_reported": item.get("run_family", ""),
                "old_gold_model_reported": item.get("model_guess", ""),
                "old_gold_n_news": item.get("n_news", pd.NA),
                "old_gold_gold_pairs": item.get("gold_positive_pairs", pd.NA),
                "old_gold_pred_pairs": item.get("pred_positive_pairs", pd.NA),
                "old_gold_mean_pred_labels": item.get("pred_mean_labels", pd.NA),
                "old_gold_precision": item.get("micro_precision", pd.NA),
                "old_gold_recall": item.get("micro_recall", pd.NA),
                "old_gold_micro_f1": item.get("micro_f1", pd.NA),
                "old_gold_macro_f1_supported": item.get("macro_f1_supported", pd.NA),
                "old_gold_sample_f1": item.get("sample_f1_empty_correct", pd.NA),
                "old_gold_any_relevant_precision": item.get("any_relevant_precision", pd.NA),
                "old_gold_any_relevant_recall": item.get("any_relevant_recall", pd.NA),
                "old_gold_any_relevant_f1": item.get("any_relevant_f1", pd.NA),
                "old_gold_false_relevant_news": item.get("false_relevant_news", pd.NA),
                "old_gold_missed_all_relevant_news": item.get("missed_all_relevant_news", pd.NA),
                "old_gold_error_rows": item.get("error_rows", pd.NA),
            }
        )
    return pd.DataFrame(rows)


def load_sentiment_pressure(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "run_id" not in df.columns:
        return pd.DataFrame()
    df["run_id"] = df["run_id"].astype(str)
    df["threshold"] = df["threshold"].map(normalize_threshold)
    keep = [
        "run_id",
        "threshold",
        "true_sentiment_pairs",
        "predicted_true_factor_pairs",
        "sentiment_pair_coverage",
        "sentiment_accuracy_on_predicted_true_pairs",
        "sentiment_macro_f1_on_predicted_true_pairs",
        "sentiment_weighted_f1_on_predicted_true_pairs",
        "pressure_eval_pairs",
        "pressure_mae_on_predicted_true_pairs",
        "pressure_rmse_on_predicted_true_pairs",
        "pressure_corr_on_predicted_true_pairs",
    ]
    df = df[[col for col in keep if col in df.columns]].copy()
    return df.rename(columns={col: f"wide_{col}" for col in df.columns if col not in {"run_id", "threshold"}})


def outer_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if left.empty:
        return right
    if right.empty:
        return left
    return left.merge(right, on=["run_id", "threshold"], how="outer")


def add_standard_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    reported_model_cols = [col for col in ["wide_model_reported", "v4_strength_model_reported", "old_gold_model_reported"] if col in df.columns]
    reported_prompt_cols = [col for col in ["wide_prompt_reported", "v4_strength_prompt_reported"] if col in df.columns]
    reported_family_cols = [
        col
        for col in ["wide_run_family_reported", "v4_strength_run_family_reported", "old_gold_run_family_reported"]
        if col in df.columns
    ]
    source_cols = [col for col in ["wide_source_file", "v4_strength_source_file", "old_gold_source_file"] if col in df.columns]

    metadata_rows = []
    for _, row in df.iterrows():
        run_id = str(row["run_id"])
        source_file = str(first_present(row, source_cols, ""))
        model_reported = str(first_present(row, reported_model_cols, ""))
        prompt_reported = str(first_present(row, reported_prompt_cols, ""))
        family_reported = str(first_present(row, reported_family_cols, ""))
        name = f"{run_id} {source_file}"
        prompt_version = parse_prompt_version(name, prompt_reported)
        metadata_rows.append(
            {
                "source_file": source_file,
                "run_family": parse_run_family(run_id, family_reported),
                "model": parse_model(name, model_reported),
                "prompt_reported": prompt_reported if prompt_reported else "unknown",
                "prompt_version": prompt_version,
                "prompt_family": parse_prompt_family(run_id, prompt_version),
                "thinking": parse_bool_thinking(name),
                "context_window_tokens": parse_int_token(name, "ctx"),
                "max_output_tokens": parse_int_token(name, "tok"),
                "text_limit_chars_or_tokens": parse_text_limit(name),
                "batch_size": parse_batch_size(name),
                "requested_n_news": parse_requested_n(name),
                "is_partial": ".partial" in run_id or ".partial" in source_file,
            }
        )
    meta = pd.DataFrame(metadata_rows, index=df.index)
    return pd.concat([df[["run_id", "threshold"]], meta, df.drop(columns=["run_id", "threshold"])], axis=1)


def select_best_by_run(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    scored = df.copy()
    metric_priority = [
        ("old_gold_micro_f1", "old_gold_micro_f1"),
        ("wide_all_micro_f1", "wide_all_micro_f1"),
        ("v4_strength_all_micro_f1", "v4_strength_all_micro_f1"),
    ]
    selection_metric = []
    selection_score = []
    for _, row in scored.iterrows():
        metric_name = ""
        metric_value = float("-inf")
        for col, name in metric_priority:
            if col in row and pd.notna(row[col]):
                metric_name = name
                metric_value = float(row[col])
                break
        selection_metric.append(metric_name)
        selection_score.append(metric_value if metric_name else pd.NA)
    scored["selection_metric"] = selection_metric
    scored["selection_score"] = selection_score
    sort_cols = ["run_id", "selection_score"]
    if "wide_n_news" in scored.columns:
        sort_cols.append("wide_n_news")
    if "old_gold_n_news" in scored.columns:
        sort_cols.append("old_gold_n_news")
    ascending = [True, False] + [False] * (len(sort_cols) - 2)
    return (
        scored.sort_values(sort_cols, ascending=ascending, na_position="last")
        .groupby("run_id", as_index=False)
        .head(1)
        .sort_values(["selection_score", "wide_n_news", "old_gold_n_news"], ascending=[False, False, False], na_position="last")
    )


def best_rows_for_metric(df: pd.DataFrame, metric_col: str, recall_col: str, n_col: str) -> pd.DataFrame:
    if df.empty or metric_col not in df.columns:
        return pd.DataFrame()
    cols = [metric_col]
    if recall_col in df.columns:
        cols.append(recall_col)
    if n_col in df.columns:
        cols.append(n_col)
    candidates = df[df[metric_col].notna()].copy()
    if candidates.empty:
        return candidates
    sort_cols = [metric_col]
    ascending = [False]
    if recall_col in candidates.columns:
        sort_cols.append(recall_col)
        ascending.append(False)
    if n_col in candidates.columns:
        sort_cols.append(n_col)
        ascending.append(False)
    return candidates.sort_values(sort_cols, ascending=ascending).groupby("run_id", as_index=False).head(1)


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    front = [col for col in PRIMARY_COLUMNS if col in df.columns]
    rest = [col for col in df.columns if col not in front]
    return df[front + rest]


def coerce_int_columns(df: pd.DataFrame) -> pd.DataFrame:
    converted = df.copy()
    for col in INT_COLUMNS:
        if col in converted.columns:
            converted[col] = pd.to_numeric(converted[col], errors="coerce").astype("Int64")
    return converted


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int = 12, digits: int = 4) -> str:
    subset = df[[col for col in columns if col in df.columns]].head(limit).copy()
    if subset.empty:
        return "No rows."
    for col in subset.columns:
        if pd.api.types.is_float_dtype(subset[col]):
            subset[col] = subset[col].round(digits)
    printable = subset.astype("object").where(pd.notna(subset), "")
    rows = [list(printable.columns)] + printable.astype(str).values.tolist()
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    header = "| " + " | ".join(rows[0][i].ljust(widths[i]) for i in range(len(widths))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(widths))) + " |" for row in rows[1:]]
    return "\n".join([header, sep] + body)


def thinking_run_label(name: str) -> str:
    model = parse_model(name)
    ctx = parse_int_token(name, "ctx")
    tok = parse_int_token(name, "tok")
    batch = parse_batch_size(name)
    parts = []
    if "oldgold" in name.lower():
        parts.append("old_gold")
    parts.append(model)
    if batch:
        parts.append(f"b{batch}")
    if ctx:
        parts.append(f"ctx{ctx}")
    if tok:
        parts.append(f"tok{tok}")
    if "guard2" in name.lower():
        parts.append("guard2")
    elif "guard" in name.lower():
        parts.append("guard")
    return " ".join(parts)


def load_thinking_trace_summary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT.glob("*think_nojson*.ollama_response.jsonl")):
        run_id = path.name.replace(".ollama_response.jsonl", "")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                rows.append(
                    {
                        "run_id": run_id,
                        "run_label": thinking_run_label(run_id),
                        "status": "jsonl_parse_error",
                        "done_reason": "",
                        "content_chars": 0,
                        "thinking_chars": 0,
                        "duration_s": 0.0,
                        "wait_count": 0,
                        "final_check_count": 0,
                        "markdown_count": 0,
                    }
                )
                continue
            raw = item.get("raw_response") or {}
            message = raw.get("message") if isinstance(raw, dict) else {}
            if not isinstance(message, dict):
                message = {}
            thinking = str(message.get("thinking") or "")
            content = str(message.get("content") or raw.get("response") or "") if isinstance(raw, dict) else ""
            rows.append(
                {
                    "run_id": run_id,
                    "run_label": thinking_run_label(run_id),
                    "status": item.get("status", ""),
                    "done_reason": item.get("done_reason") or (raw.get("done_reason") if isinstance(raw, dict) else ""),
                    "content_chars": item.get("content_chars", len(content)),
                    "thinking_chars": item.get("thinking_chars", len(thinking)),
                    "eval_count": item.get("eval_count") or (raw.get("eval_count") if isinstance(raw, dict) else pd.NA),
                    "duration_s": float(item.get("total_duration") or (raw.get("total_duration") if isinstance(raw, dict) else 0) or 0)
                    / 1_000_000_000,
                    "wait_count": len(re.findall(r"\bWait\b|\*Wait|wait,", thinking, flags=re.IGNORECASE)),
                    "final_check_count": thinking.count("Final check"),
                    "markdown_count": thinking.count("Do not use markdown") + thinking.count("Не используй markdown"),
                }
            )
    if not rows:
        return pd.DataFrame()
    traces = pd.DataFrame(rows)
    traces["content_chars"] = pd.to_numeric(traces["content_chars"], errors="coerce").fillna(0)
    traces["thinking_chars"] = pd.to_numeric(traces["thinking_chars"], errors="coerce").fillna(0)
    traces["duration_s"] = pd.to_numeric(traces["duration_s"], errors="coerce").fillna(0)
    grouped = traces.groupby(["run_id", "run_label"], as_index=False).agg(
        response_rows=("status", "size"),
        ok_rows=("status", lambda value: int((value == "ok").sum())),
        error_rows=("status", lambda value: int((value != "ok").sum())),
        length_rows=("done_reason", lambda value: int((value == "length").sum())),
        empty_content_rows=("content_chars", lambda value: int((value <= 0).sum())),
        avg_thinking_chars=("thinking_chars", "mean"),
        max_thinking_chars=("thinking_chars", "max"),
        avg_duration_s=("duration_s", "mean"),
        wait_count=("wait_count", "sum"),
        final_check_count=("final_check_count", "sum"),
        markdown_count=("markdown_count", "sum"),
    )
    for col in ["avg_thinking_chars", "avg_duration_s"]:
        grouped[col] = grouped[col].round(1)
    grouped["max_thinking_chars"] = grouped["max_thinking_chars"].astype(int)
    return grouped.sort_values(["run_label", "run_id"])


def write_summary(all_rows: pd.DataFrame, best: pd.DataFrame) -> None:
    lines = ["# Standardized LLM Metrics", ""]
    lines.append(f"Rows in all-threshold table: {len(all_rows)}")
    lines.append(f"Runs in best-by-run table: {best['run_id'].nunique() if not best.empty else 0}")
    lines.append(f"Rows with wide metrics: {int(all_rows.get('wide_all_micro_f1', pd.Series(dtype=float)).notna().sum())}")
    lines.append(f"Rows with v4 strength metrics: {int(all_rows.get('v4_strength_all_micro_f1', pd.Series(dtype=float)).notna().sum())}")
    lines.append(f"Rows with old gold metrics: {int(all_rows.get('old_gold_micro_f1', pd.Series(dtype=float)).notna().sum())}")
    lines.append("")
    lines.append("## Top V4 Dataset, Together")
    v4_best = best_rows_for_metric(all_rows, "v4_strength_all_micro_f1", "v4_strength_all_recall", "v4_strength_n_news")
    v4_top = v4_best[
        v4_best.get("v4_strength_all_micro_f1", pd.Series(dtype=float)).notna()
        & pd.to_numeric(v4_best.get("v4_strength_n_news", pd.Series(dtype=float)), errors="coerce").ge(30)
    ].sort_values(
        ["v4_strength_all_micro_f1", "v4_strength_all_recall", "v4_strength_n_news"], ascending=False
    )
    lines.append(
        markdown_table(
            v4_top,
            [
                "model",
                "thinking",
                "context_window_tokens",
                "max_output_tokens",
                "batch_size",
                "prompt_version",
                "threshold",
                "v4_strength_n_news",
                "v4_strength_all_precision",
                "v4_strength_all_recall",
                "v4_strength_all_micro_f1",
                "v4_strength_json_errors",
            ],
        )
    )
    lines.append("")
    lines.append("## Latest V4 Pilot60 Grid")
    lines.append("Default-label pilot metrics on the same deterministic 60 v4 rows; partial rows are failed/aborted thinking runs.")
    latest_prompt_mask = all_rows["prompt_version"].astype(str).isin({"social_signal_v9_f1_balanced", "high_recall_minimal_v3"})
    pilot_mask = (
        all_rows["run_id"].astype(str).str.contains("v4pilot60", case=False, na=False)
        & latest_prompt_mask
        & all_rows["threshold"].astype(str).eq("default")
    )
    pilot = all_rows[pilot_mask].copy()
    if not pilot.empty:
        pilot = pilot.sort_values(["prompt_version", "model", "thinking"])
    lines.append(
        markdown_table(
            pilot,
            [
                "model",
                "thinking",
                "prompt_version",
                "v4_strength_n_news",
                "is_partial",
                "v4_strength_all_precision",
                "v4_strength_all_recall",
                "v4_strength_all_micro_f1",
                "v4_strength_all_pred_pairs",
                "v4_strength_json_errors",
            ],
            limit=12,
        )
    )
    lines.append("")
    lines.append("## Latest V4 Full Winner")
    full_mask = all_rows["run_id"].astype(str).str.contains(
        "v4full965_gemma4_26b_social_signal_v9_f1_balanced", case=False, na=False
    )
    full_thresholds = all_rows["threshold"].astype(str).isin({"default", "0.60", "0.75", "0.85"})
    full = all_rows[full_mask & full_thresholds].copy()
    if not full.empty:
        threshold_order = {"default": 0, "0.60": 1, "0.75": 2, "0.85": 3}
        full["_threshold_order"] = full["threshold"].map(threshold_order).fillna(99)
        full = full.sort_values("_threshold_order").drop(columns=["_threshold_order"])
    lines.append(
        markdown_table(
            full,
            [
                "model",
                "thinking",
                "prompt_version",
                "threshold",
                "v4_strength_n_news",
                "v4_strength_all_precision",
                "v4_strength_all_recall",
                "v4_strength_all_micro_f1",
                "old_gold_n_news",
                "old_gold_precision",
                "old_gold_recall",
                "old_gold_micro_f1",
            ],
            limit=8,
        )
    )
    lines.append("")
    lines.append("## Top Wide Dataset, Together")
    wide_best = best_rows_for_metric(all_rows, "wide_all_micro_f1", "wide_all_recall", "wide_n_news")
    wide_top = wide_best[wide_best.get("wide_all_micro_f1", pd.Series(dtype=float)).notna()].sort_values(
        ["wide_all_micro_f1", "wide_all_recall", "wide_n_news"], ascending=False
    )
    lines.append(
        markdown_table(
            wide_top,
            [
                "model",
                "thinking",
                "context_window_tokens",
                "max_output_tokens",
                "prompt_version",
                "threshold",
                "wide_n_news",
                "wide_all_precision",
                "wide_all_recall",
                "wide_all_micro_f1",
            ],
        )
    )
    for scope in ["direct", "context", "weak"]:
        lines.append("")
        lines.append(f"## Top Wide Dataset, {scope}")
        col = f"wide_{scope}_micro_f1"
        recall_col = f"wide_{scope}_recall"
        scope_best = best_rows_for_metric(all_rows, col, recall_col, "wide_n_news")
        scope_top = scope_best[scope_best.get(col, pd.Series(dtype=float)).notna()].sort_values([col, recall_col, "wide_n_news"], ascending=False)
        lines.append(
            markdown_table(
                scope_top,
                [
                    "model",
                    "thinking",
                    "prompt_version",
                    "threshold",
                    "wide_n_news",
                    f"wide_{scope}_precision",
                    f"wide_{scope}_recall",
                    f"wide_{scope}_micro_f1",
                ],
            )
        )
    lines.append("")
    lines.append("## Top Old Gold Only")
    old_best = best_rows_for_metric(all_rows, "old_gold_micro_f1", "old_gold_recall", "old_gold_n_news")
    old_top = old_best[old_best.get("old_gold_micro_f1", pd.Series(dtype=float)).notna()].sort_values(
        ["old_gold_micro_f1", "old_gold_recall", "old_gold_n_news"], ascending=False
    )
    lines.append(
        markdown_table(
            old_top,
            [
                "model",
                "thinking",
                "prompt_version",
                "threshold",
                "old_gold_n_news",
                "old_gold_precision",
                "old_gold_recall",
                "old_gold_micro_f1",
            ],
        )
    )
    latest_mask = old_best["run_id"].astype(str).str.contains("gemma4_26b_direct_core_targeted_v7_nothink", na=False) if not old_best.empty else pd.Series(dtype=bool)
    latest = old_best[latest_mask].copy() if not old_best.empty else pd.DataFrame()
    if not latest.empty:
        lines.append("")
        lines.append("## Latest Old Gold Check")
        lines.append(
            markdown_table(
                latest.sort_values(["old_gold_micro_f1", "old_gold_recall"], ascending=False),
                [
                    "model",
                    "thinking",
                    "context_window_tokens",
                    "max_output_tokens",
                    "batch_size",
                    "prompt_version",
                    "threshold",
                    "old_gold_n_news",
                    "old_gold_precision",
                    "old_gold_recall",
                    "old_gold_micro_f1",
                    "wide_all_micro_f1",
                    "v4_strength_direct_micro_f1",
                ],
                limit=5,
            )
        )

    oldgold_thinking_mask = (
        best["run_id"].astype(str).str.contains("oldgold", case=False, na=False)
        & best["run_id"].astype(str).str.contains("think_nojson", case=False, na=False)
    ) if not best.empty else pd.Series(dtype=bool)
    oldgold_thinking = best[oldgold_thinking_mask].copy() if not best.empty else pd.DataFrame()
    if not oldgold_thinking.empty:
        lines.append("")
        lines.append("## Thinking Old Gold Pilots")
        lines.append(
            markdown_table(
                oldgold_thinking.sort_values(["old_gold_micro_f1", "old_gold_n_news"], ascending=False),
                [
                    "model",
                    "thinking",
                    "context_window_tokens",
                    "max_output_tokens",
                    "batch_size",
                    "prompt_version",
                    "threshold",
                    "old_gold_n_news",
                    "old_gold_precision",
                    "old_gold_recall",
                    "old_gold_micro_f1",
                    "old_gold_error_rows",
                ],
                limit=20,
            )
        )

    traces = load_thinking_trace_summary()
    if not traces.empty:
        lines.append("")
        lines.append("## Thinking Trace Diagnostics")
        lines.append(
            markdown_table(
                traces,
                [
                    "run_label",
                    "response_rows",
                    "ok_rows",
                    "error_rows",
                    "length_rows",
                    "empty_content_rows",
                    "avg_thinking_chars",
                    "max_thinking_chars",
                    "avg_duration_s",
                    "wait_count",
                    "final_check_count",
                    "markdown_count",
                ],
                limit=20,
            )
        )
    lines.append("")
    lines.append("## Column Standard")
    lines.append("- `wide_all_*` is the together score on the broad weak/direct/context dataset.")
    lines.append("- `wide_direct_*`, `wide_context_*`, and `wide_weak_*` are separated strength scores.")
    lines.append("- `v4_strength_*` is the same strength split on the normalized v4 benchmark subset.")
    lines.append("- `old_gold_*` is evaluated only on the old multilabel gold dataset.")
    lines.append("- `context_window_tokens` and `max_output_tokens` are parsed from `ctx*` and `tok*` filename tokens.")
    (OUT / "standardized_llm_metrics_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    wide = load_scoped_metrics(OUT / "v3_all_runs_threshold_strength_metrics.csv", "wide")
    sentiment_pressure = load_sentiment_pressure(OUT / "v3_all_runs_sentiment_pressure_metrics.csv")
    if not wide.empty and not sentiment_pressure.empty:
        wide = wide.merge(sentiment_pressure, on=["run_id", "threshold"], how="left")
    v4_strength = load_scoped_metrics(OUT / "v4_strength_metrics.csv", "v4_strength")
    old_gold = load_old_gold_metrics(OUT / "old_gold_all_runs_metrics.csv")

    all_rows = outer_merge(outer_merge(wide, v4_strength), old_gold)
    all_rows = add_standard_metadata(all_rows)
    best = select_best_by_run(all_rows)
    all_rows = coerce_int_columns(all_rows)
    best = coerce_int_columns(best)
    all_rows = order_columns(all_rows)
    best = order_columns(best)

    all_path = OUT / "standardized_llm_metrics_all_thresholds.csv"
    best_path = OUT / "standardized_llm_metrics_all.csv"
    all_rows.to_csv(all_path, index=False, encoding="utf-8-sig")
    best.to_csv(best_path, index=False, encoding="utf-8-sig")
    write_summary(all_rows, best)

    print(f"saved={best_path}")
    print(f"saved={all_path}")
    print(f"saved={OUT / 'standardized_llm_metrics_summary.md'}")
    print(f"runs={best['run_id'].nunique()} all_threshold_rows={len(all_rows)}")
    preview_cols = [
        "model",
        "thinking",
        "context_window_tokens",
        "max_output_tokens",
        "prompt_version",
        "threshold",
        "wide_n_news",
        "wide_all_precision",
        "wide_all_recall",
        "wide_all_micro_f1",
        "old_gold_n_news",
        "old_gold_precision",
        "old_gold_recall",
        "old_gold_micro_f1",
    ]
    print(best[[col for col in preview_cols if col in best.columns]].head(20).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
