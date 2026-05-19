from __future__ import annotations

import json
import math
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyzer.factors import FACTOR_CONFIG


OUT = ROOT / "analysis_outputs"
DATASET = OUT / "v3_clean_broad_weak_dataset_1000.csv"

THRESHOLDS = [None] + [round(float(value), 2) for value in np.arange(0.05, 1.00, 0.05)]
SCOPES = ["all", "direct", "context", "weak", "direct_context"]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return series.fillna("").astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "да"})


def normalize_sentiment_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if label in SENTIMENT_LABELS:
        return label
    if label in {"pos", "positive_signal", "позитив", "положительный"}:
        return "positive"
    if label in {"neg", "negative_signal", "негатив", "отрицательный"}:
        return "negative"
    if label in {"neu", "none", "not_applicable", "not relevant", "not_relevant", "нерелевантно"}:
        return "neutral"
    return ""


def sentiment_from_score(value: Any) -> str:
    score = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(score):
        return ""
    if float(score) > 0.2:
        return "positive"
    if float(score) < -0.2:
        return "negative"
    return "neutral"


def run_family(name: str) -> str:
    if name.startswith("prompt_search_") or name.startswith("manual_dataset_prompt_search"):
        return "prompt_search"
    if name.startswith("thinking_recall_experiment"):
        return "thinking_recall"
    if name.startswith("grouped_event"):
        return "grouped_event"
    if name.startswith("v3_social_signal"):
        return "v3_social_signal"
    if name.startswith("v4_social_signal"):
        return "v4_social_signal"
    if name.startswith("manual_dataset_llm_predictions"):
        return "manual_llm"
    return "other"


def infer_model(name: str) -> str:
    lowered = name.lower()
    if "manual_dataset_bert" in lowered:
        return "rubert-nli-zero-shot"
    if "qwen36_35b" in lowered or "qwen3.6-35b" in lowered:
        return "batiai/qwen3.6-35b:iq3"
    if "qwen35_9b" in lowered or "qwen3.5" in lowered:
        return "qwen3.5:9b"
    if "gemma4_e4b" in lowered or "gemma4:e4b" in lowered:
        return "gemma4:e4b"
    if "gemma4-e2b" in lowered or "gemma4_e2b" in lowered or "gemma4:e2b" in lowered:
        return "gemma4:e2b"
    if "gemma3n" in lowered:
        return "gemma3n:e4b"
    # Older experiment runners did not put the default model in prediction
    # filenames. At the time these artifacts were produced, .env used gemma4:e2b.
    if (
        lowered.startswith("v3_social_signal_experiment_")
        or lowered.startswith("v4_social_signal_experiment_")
        or lowered.startswith("grouped_event")
        or lowered.startswith("thinking_recall_experiment")
        or lowered.startswith("manual_dataset_prompt_search")
    ):
        return "gemma4:e2b"
    return "unknown"


def infer_prompt(name: str) -> str:
    lowered = name.lower()
    for token in [
        "balanced_recall_v2",
        "production_v1",
        "few_shot_major_v3",
        "hard_negative_v2",
        "primary_first_v2",
        "recall_max_v1",
        "latent_candidate_v2",
        "latent_candidate_v3",
        "grouped_event_v1",
        "social_signal_v1",
        "social_signal_v2",
        "social_signal_v3",
        "social_signal_v4",
        "social_signal_v5",
        "v4_prompt",
    ]:
        if token in lowered:
            return token
    return "unknown"


def candidate_prediction_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(OUT.glob("*.csv")):
        if path.name.endswith(".partial.csv"):
            final_path = path.with_name(path.name.replace(".partial.csv", ".csv"))
            if final_path.exists():
                continue
        try:
            columns = set(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        if {"dataset_row_id", "factor_key"}.issubset(columns) and ({"relevance", "is_relevant"} & columns):
            files.append(path)
    return files


def candidate_file_audit() -> pd.DataFrame:
    rows = []
    included_paths = {path.name for path in candidate_prediction_files()}
    for path in sorted(OUT.glob("*.csv")):
        try:
            columns = set(pd.read_csv(path, nrows=0).columns)
        except Exception as exc:
            rows.append(
                {
                    "file_name": path.name,
                    "is_prediction_shape": False,
                    "included": False,
                    "reason": f"header_read_error: {exc!r}",
                    "model": infer_model(path.name),
                    "prompt": infer_prompt(path.name),
                }
            )
            continue
        is_prediction_shape = {"dataset_row_id", "factor_key"}.issubset(columns) and ({"relevance", "is_relevant"} & columns)
        reason = ""
        if not is_prediction_shape:
            reason = "not_prediction_file"
        elif path.name.endswith(".partial.csv") and path.with_name(path.name.replace(".partial.csv", ".csv")).exists():
            reason = "skipped_partial_duplicate_final_exists"
        else:
            reason = "included"
        rows.append(
            {
                "file_name": path.name,
                "is_prediction_shape": bool(is_prediction_shape),
                "included": path.name in included_paths,
                "reason": reason,
                "model": infer_model(path.name),
                "prompt": infer_prompt(path.name),
            }
        )
    return pd.DataFrame(rows)


def raw_error_count(path: Path) -> int | None:
    candidates = []
    if path.name.endswith(".partial.csv"):
        candidates.append(path.with_name(path.name.replace(".partial.csv", ".raw.jsonl")))
    candidates.append(path.with_suffix(".raw.jsonl"))
    for raw_path in candidates:
        if not raw_path.exists():
            continue
        statuses = []
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    statuses.append(json.loads(line).get("status"))
                except json.JSONDecodeError:
                    statuses.append("jsonl_parse_error")
        return statuses.count("error") + statuses.count("jsonl_parse_error")
    return None


def sample_f1(gold: np.ndarray, pred: np.ndarray) -> float:
    scores = []
    for gold_row, pred_row in zip(gold, pred):
        true_sum = int(gold_row.sum())
        pred_sum = int(pred_row.sum())
        tp = int(((gold_row == 1) & (pred_row == 1)).sum())
        if true_sum == 0 and pred_sum == 0:
            scores.append(1.0)
        elif true_sum == 0 or pred_sum == 0:
            scores.append(0.0)
        else:
            scores.append(2 * tp / (true_sum + pred_sum))
    return float(np.mean(scores)) if scores else 0.0


def safe_micro(gold: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(gold, pred, average="micro", zero_division=0)
    return float(precision), float(recall), float(f1)


def safe_macro_supported(gold: np.ndarray, pred: np.ndarray) -> float:
    supported = gold.sum(axis=0) > 0
    if not supported.any():
        return 0.0
    return float(precision_recall_fscore_support(gold[:, supported], pred[:, supported], average="macro", zero_division=0)[2])


def load_prediction_matrix(path: Path, eval_ids: list[int], factor_keys: list[str]) -> dict[str, Any]:
    pred = pd.read_csv(path)
    pred["dataset_row_id"] = pd.to_numeric(pred["dataset_row_id"], errors="coerce")
    pred = pred[pred["dataset_row_id"].notna()].copy()
    pred["dataset_row_id"] = pred["dataset_row_id"].astype(int)
    pred["factor_key"] = pred["factor_key"].astype(str)
    pred = pred[pred["dataset_row_id"].isin(eval_ids) & pred["factor_key"].isin(factor_keys)].copy()
    pred = pred.drop_duplicates(["dataset_row_id", "factor_key"], keep="last")

    if "relevance" not in pred.columns:
        pred["relevance"] = normalize_bool_series(pred["is_relevant"]).astype(float)
    pred["relevance"] = pd.to_numeric(pred["relevance"], errors="coerce").fillna(0.0)
    if "is_relevant" in pred.columns:
        pred["is_relevant_norm"] = normalize_bool_series(pred["is_relevant"])
    else:
        pred["is_relevant_norm"] = pred["relevance"].ge(0.3)

    covered_ids = sorted(set(pred["dataset_row_id"].tolist()) & set(eval_ids))
    if not covered_ids:
        raise ValueError(f"{path.name}: no covered v3 rows")

    scores = (
        pred.pivot(index="dataset_row_id", columns="factor_key", values="relevance")
        .reindex(index=covered_ids, columns=factor_keys)
        .fillna(0.0)
        .to_numpy(float)
    )
    default_labels = (
        pred.pivot(index="dataset_row_id", columns="factor_key", values="is_relevant_norm")
        .reindex(index=covered_ids, columns=factor_keys)
        .fillna(False)
        .astype(int)
        .to_numpy()
    )

    if "sentiment_label" in pred.columns:
        pred["sentiment_norm"] = pred["sentiment_label"].map(normalize_sentiment_label)
    elif "sentiment_score" in pred.columns:
        pred["sentiment_norm"] = pred["sentiment_score"].map(sentiment_from_score)
    else:
        pred["sentiment_norm"] = ""
    sentiment = (
        pred.pivot(index="dataset_row_id", columns="factor_key", values="sentiment_norm")
        .reindex(index=covered_ids, columns=factor_keys)
        .fillna("")
        .to_numpy(str)
    )

    if "pressure" in pred.columns:
        pressure = (
            pred.assign(pressure_num=pd.to_numeric(pred["pressure"], errors="coerce"))
            .pivot(index="dataset_row_id", columns="factor_key", values="pressure_num")
            .reindex(index=covered_ids, columns=factor_keys)
            .to_numpy(float)
        )
    else:
        pressure = np.full_like(scores, np.nan, dtype=float)

    return {
        "covered_ids": covered_ids,
        "scores": scores,
        "default_labels": default_labels,
        "sentiment": sentiment,
        "pressure": pressure,
        "rows": len(pred),
        "raw_positive_rows": int(default_labels.sum()),
    }


def gold_scope_matrix(all_gold: np.ndarray, strength: np.ndarray, scope: str) -> np.ndarray:
    if scope == "all":
        return all_gold.astype(int)
    if scope == "direct_context":
        return ((strength == "direct") | (strength == "context")).astype(int)
    return (strength == scope).astype(int)


def evaluate_scope(
    run_meta: dict[str, Any],
    threshold_label: str,
    threshold: float | None,
    labels: np.ndarray,
    gold_all: np.ndarray,
    strength: np.ndarray,
    scope: str,
) -> dict[str, Any]:
    gold = gold_scope_matrix(gold_all, strength, scope)
    precision, recall, f1 = safe_micro(gold, labels)
    any_metric = precision_recall_fscore_support(gold.sum(axis=1) > 0, labels.sum(axis=1) > 0, average="binary", zero_division=0)
    true_hit_by_news = ((gold == 1) & (labels == 1)).sum(axis=1) > 0
    gold_news = gold.sum(axis=1) > 0
    tp = int(((gold == 1) & (labels == 1)).sum())
    fp = int(((gold == 0) & (labels == 1)).sum())
    fn = int(((gold == 1) & (labels == 0)).sum())

    return {
        **run_meta,
        "threshold": threshold_label,
        "threshold_value": np.nan if threshold is None else float(threshold),
        "scope": scope,
        "gold_pairs": int(gold.sum()),
        "pred_pairs": int(labels.sum()),
        "tp_pairs": tp,
        "fp_pairs_vs_scope": fp,
        "fn_pairs": fn,
        "mean_pred_labels": float(labels.sum(axis=1).mean()),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "macro_f1_supported": safe_macro_supported(gold, labels),
        "sample_f1_empty_correct": sample_f1(gold, labels),
        "any_precision": float(any_metric[0]),
        "any_recall": float(any_metric[1]),
        "any_f1": float(any_metric[2]),
        "gold_news": int(gold_news.sum()),
        "pred_any_news": int((labels.sum(axis=1) > 0).sum()),
        "news_with_true_hit": int((gold_news & true_hit_by_news).sum()),
        "true_hit_news_coverage": float((gold_news & true_hit_by_news).sum() / max(1, gold_news.sum())),
        "false_relevant_news": int(((gold.sum(axis=1) == 0) & (labels.sum(axis=1) > 0)).sum()),
        "missed_all_scope_news": int(((gold.sum(axis=1) > 0) & (labels.sum(axis=1) == 0)).sum()),
    }


def evaluate_per_factor(
    run_meta: dict[str, Any],
    threshold_label: str,
    threshold: float | None,
    labels: np.ndarray,
    gold_all: np.ndarray,
    strength: np.ndarray,
    factor_keys: list[str],
    scope: str,
) -> list[dict[str, Any]]:
    gold = gold_scope_matrix(gold_all, strength, scope)
    rows = []
    for j, key in enumerate(factor_keys):
        precision, recall, f1, _ = precision_recall_fscore_support(gold[:, j], labels[:, j], average="binary", zero_division=0)
        rows.append(
            {
                **run_meta,
                "threshold": threshold_label,
                "threshold_value": np.nan if threshold is None else float(threshold),
                "scope": scope,
                "factor_key": key,
                "support": int(gold[:, j].sum()),
                "predicted": int(labels[:, j].sum()),
                "tp": int(((gold[:, j] == 1) & (labels[:, j] == 1)).sum()),
                "fp": int(((gold[:, j] == 0) & (labels[:, j] == 1)).sum()),
                "fn": int(((gold[:, j] == 1) & (labels[:, j] == 0)).sum()),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return rows


def evaluate_sentiment_pressure(
    run_meta: dict[str, Any],
    threshold_label: str,
    threshold: float | None,
    labels: np.ndarray,
    gold_all: np.ndarray,
    gold_sentiment: np.ndarray,
    pred_sentiment: np.ndarray,
    gold_pressure: np.ndarray,
    pred_pressure: np.ndarray,
) -> dict[str, Any]:
    true_pair_mask = (gold_all == 1) & np.isin(gold_sentiment, SENTIMENT_LABELS)
    true_positive_mask = true_pair_mask & (labels == 1) & np.isin(pred_sentiment, SENTIMENT_LABELS)
    y_true = gold_sentiment[true_positive_mask]
    y_pred = pred_sentiment[true_positive_mask]
    if len(y_true):
        sentiment_accuracy = float(accuracy_score(y_true, y_pred))
        sentiment_macro_f1 = float(precision_recall_fscore_support(y_true, y_pred, labels=SENTIMENT_LABELS, average="macro", zero_division=0)[2])
        sentiment_weighted_f1 = float(precision_recall_fscore_support(y_true, y_pred, labels=SENTIMENT_LABELS, average="weighted", zero_division=0)[2])
    else:
        sentiment_accuracy = 0.0
        sentiment_macro_f1 = 0.0
        sentiment_weighted_f1 = 0.0

    pressure_mask = true_positive_mask & np.isfinite(gold_pressure) & np.isfinite(pred_pressure)
    if pressure_mask.any():
        diff = pred_pressure[pressure_mask] - gold_pressure[pressure_mask]
        pressure_mae = float(np.abs(diff).mean())
        pressure_rmse = float(math.sqrt((diff * diff).mean()))
        if pressure_mask.sum() >= 2 and np.nanstd(gold_pressure[pressure_mask]) > 0 and np.nanstd(pred_pressure[pressure_mask]) > 0:
            pressure_corr = float(np.corrcoef(gold_pressure[pressure_mask], pred_pressure[pressure_mask])[0, 1])
        else:
            pressure_corr = np.nan
    else:
        pressure_mae = np.nan
        pressure_rmse = np.nan
        pressure_corr = np.nan

    return {
        **run_meta,
        "threshold": threshold_label,
        "threshold_value": np.nan if threshold is None else float(threshold),
        "true_sentiment_pairs": int(true_pair_mask.sum()),
        "predicted_true_factor_pairs": int(true_positive_mask.sum()),
        "sentiment_pair_coverage": float(true_positive_mask.sum() / max(1, true_pair_mask.sum())),
        "sentiment_accuracy_on_predicted_true_pairs": sentiment_accuracy,
        "sentiment_macro_f1_on_predicted_true_pairs": sentiment_macro_f1,
        "sentiment_weighted_f1_on_predicted_true_pairs": sentiment_weighted_f1,
        "pressure_eval_pairs": int(pressure_mask.sum()),
        "pressure_mae_on_predicted_true_pairs": pressure_mae,
        "pressure_rmse_on_predicted_true_pairs": pressure_rmse,
        "pressure_corr_on_predicted_true_pairs": pressure_corr,
    }


def main() -> None:
    df = pd.read_csv(DATASET, encoding="utf-8-sig")
    df["dataset_row_id"] = pd.to_numeric(df["dataset_row_id"], errors="raise").astype(int)
    factor_keys = [item["key"] for item in FACTOR_CONFIG]
    eval_ids = sorted(df["dataset_row_id"].unique().astype(int).tolist())
    indexed = df.set_index("dataset_row_id").loc[eval_ids]

    gold_all = np.column_stack(
        [
            pd.to_numeric(indexed[f"factor__{key}"], errors="coerce").fillna(0).astype(int).clip(0, 1).to_numpy()
            for key in factor_keys
        ]
    )
    strength = np.column_stack([indexed[f"strength__{key}"].fillna("").astype(str).str.lower().to_numpy() for key in factor_keys])
    gold_sentiment = np.column_stack(
        [indexed[f"sentiment__{key}"].fillna("").astype(str).str.strip().str.lower().to_numpy() for key in factor_keys]
    )
    gold_pressure = np.column_stack(
        [pd.to_numeric(indexed[f"pressure__{key}"], errors="coerce").to_numpy(float) for key in factor_keys]
    )

    metric_rows: list[dict[str, Any]] = []
    per_factor_rows: list[dict[str, Any]] = []
    sentiment_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    label_cache: list[dict[str, Any]] = []
    gold_by_coverage: dict[tuple[int, ...], dict[str, np.ndarray]] = {}

    files = candidate_prediction_files()
    audit = candidate_file_audit()
    for path in files:
        try:
            matrices = load_prediction_matrix(path, eval_ids, factor_keys)
        except Exception as exc:
            manifest_rows.append(
                {
                    "run_id": path.stem,
                    "path": str(path),
                    "family": run_family(path.name),
                    "model": infer_model(path.name),
                    "prompt": infer_prompt(path.name),
                    "is_partial": path.name.endswith(".partial.csv"),
                    "load_error": repr(exc),
                }
            )
            continue

        covered_ids = matrices["covered_ids"]
        positions = [eval_ids.index(item_id) for item_id in covered_ids]
        gold_sub = gold_all[positions]
        strength_sub = strength[positions]
        gold_sentiment_sub = gold_sentiment[positions]
        gold_pressure_sub = gold_pressure[positions]
        coverage_key = tuple(covered_ids)
        gold_by_coverage[coverage_key] = {
            "gold": gold_sub,
            "strength": strength_sub,
            "sentiment": gold_sentiment_sub,
            "pressure": gold_pressure_sub,
        }

        run_id = path.stem
        meta = {
            "run_id": run_id,
            "file_name": path.name,
            "family": run_family(path.name),
            "model": infer_model(path.name),
            "prompt": infer_prompt(path.name),
            "is_partial": path.name.endswith(".partial.csv"),
            "n_news": len(covered_ids),
            "n_rows": matrices["rows"],
            "complete_36x_rows": matrices["rows"] == len(covered_ids) * len(factor_keys),
            "json_errors": raw_error_count(path),
        }
        manifest_rows.append(
            {
                **meta,
                "path": str(path),
                "gold_pairs_all": int(gold_sub.sum()),
                "mean_gold_labels": float(gold_sub.sum(axis=1).mean()),
                "raw_positive_rows_default": matrices["raw_positive_rows"],
                "load_error": "",
            }
        )

        for threshold in THRESHOLDS:
            if threshold is None:
                threshold_label = "default"
                labels = matrices["default_labels"]
            else:
                threshold_label = f"{threshold:.2f}"
                labels = (matrices["scores"] >= threshold).astype(int)

            label_cache.append(
                {
                    "config_id": f"{run_id}@{threshold_label}",
                    "run_id": run_id,
                    "threshold": threshold_label,
                    "coverage_key": coverage_key,
                    "labels": labels.copy(),
                    "meta": meta,
                }
            )

            for scope in SCOPES:
                metric_rows.append(evaluate_scope(meta, threshold_label, threshold, labels, gold_sub, strength_sub, scope))
                per_factor_rows.extend(evaluate_per_factor(meta, threshold_label, threshold, labels, gold_sub, strength_sub, factor_keys, scope))

            sentiment_rows.append(
                evaluate_sentiment_pressure(
                    meta,
                    threshold_label,
                    threshold,
                    labels,
                    gold_sub,
                    gold_sentiment_sub,
                    matrices["sentiment"],
                    gold_pressure_sub,
                    matrices["pressure"],
                )
            )

    manifest = pd.DataFrame(manifest_rows).sort_values(["family", "model", "prompt", "n_news", "file_name"], ascending=[True, True, True, False, True])
    metrics = pd.DataFrame(metric_rows).sort_values(["scope", "micro_f1", "micro_recall"], ascending=[True, False, False])
    per_factor = pd.DataFrame(per_factor_rows)
    sentiment_pressure = pd.DataFrame(sentiment_rows).sort_values(
        ["sentiment_macro_f1_on_predicted_true_pairs", "sentiment_pair_coverage"], ascending=False
    )

    individual_all = metrics[metrics["scope"].eq("all")].copy()
    individual_all["config_id"] = individual_all["run_id"] + "@" + individual_all["threshold"]
    cache_by_id = {item["config_id"]: item for item in label_cache}
    ensemble_rows: list[dict[str, Any]] = []
    ensemble_limit_per_group = 40
    coverage_to_config_ids: dict[tuple[int, ...], list[str]] = {}
    for item in label_cache:
        coverage_to_config_ids.setdefault(item["coverage_key"], []).append(item["config_id"])

    for coverage_key, config_ids in coverage_to_config_ids.items():
        group_metrics = individual_all[individual_all["config_id"].isin(config_ids)].copy()
        if len(group_metrics) < 2:
            continue
        top_f1 = group_metrics.sort_values(["micro_f1", "micro_recall"], ascending=False).head(25)
        top_recall = group_metrics.sort_values(["micro_recall", "micro_f1"], ascending=False).head(20)
        candidate_ids = list(dict.fromkeys(top_f1["config_id"].tolist() + top_recall["config_id"].tolist()))[:ensemble_limit_per_group]
        gold_pack = gold_by_coverage[coverage_key]
        for left_id, right_id in combinations(candidate_ids, 2):
            left = cache_by_id[left_id]
            right = cache_by_id[right_id]
            for op_name, labels in [
                ("OR", (left["labels"] | right["labels"]).astype(int)),
                ("AND", (left["labels"] & right["labels"]).astype(int)),
            ]:
                left_short = re.sub(r"[^A-Za-z0-9_.:-]+", "_", left_id)[-90:]
                right_short = re.sub(r"[^A-Za-z0-9_.:-]+", "_", right_id)[-90:]
                run_id = f"ensemble_{op_name}__{left_short}__{right_short}"
                meta = {
                    "run_id": run_id,
                    "file_name": "",
                    "family": "pairwise_ensemble",
                    "model": "mixed",
                    "prompt": "mixed",
                    "is_partial": bool(left["meta"].get("is_partial")) or bool(right["meta"].get("is_partial")),
                    "n_news": len(coverage_key),
                    "n_rows": np.nan,
                    "complete_36x_rows": True,
                    "json_errors": np.nan,
                    "ensemble_op": op_name,
                    "ensemble_left": left_id,
                    "ensemble_right": right_id,
                }
                for scope in SCOPES:
                    ensemble_rows.append(
                        evaluate_scope(
                            meta,
                            "ensemble",
                            None,
                            labels,
                            gold_pack["gold"],
                            gold_pack["strength"],
                            scope,
                        )
                    )

    ensemble_metrics = pd.DataFrame(ensemble_rows)
    if not ensemble_metrics.empty:
        ensemble_metrics = ensemble_metrics.sort_values(["scope", "micro_f1", "micro_recall"], ascending=[True, False, False])

    best_by_run = (
        metrics[metrics["scope"].eq("all")]
        .sort_values(["run_id", "micro_f1", "micro_recall", "sample_f1_empty_correct"], ascending=[True, False, False, False])
        .groupby("run_id", as_index=False)
        .first()
        .sort_values(["n_news", "micro_f1", "micro_recall"], ascending=[False, False, False])
    )

    top_rows = []
    for scope in SCOPES:
        scope_df = metrics[metrics["scope"].eq(scope)]
        top_f1 = scope_df.sort_values(["micro_f1", "micro_recall", "n_news"], ascending=[False, False, False]).head(50).copy()
        top_f1["ranking"] = f"{scope}_micro_f1"
        top_recall = scope_df.sort_values(["micro_recall", "micro_f1", "n_news"], ascending=[False, False, False]).head(50).copy()
        top_recall["ranking"] = f"{scope}_micro_recall"
        top_rows.extend(top_f1.to_dict("records"))
        top_rows.extend(top_recall.to_dict("records"))
    top_configs = pd.DataFrame(top_rows)
    if not ensemble_metrics.empty:
        top_ensemble_rows = []
        for scope in SCOPES:
            scope_df = ensemble_metrics[ensemble_metrics["scope"].eq(scope)]
            top_ensemble_rows.extend(
                scope_df.sort_values(["micro_f1", "micro_recall", "n_news"], ascending=[False, False, False]).head(50).assign(
                    ranking=f"{scope}_ensemble_micro_f1"
                ).to_dict("records")
            )
            top_ensemble_rows.extend(
                scope_df.sort_values(["micro_recall", "micro_f1", "n_news"], ascending=[False, False, False]).head(50).assign(
                    ranking=f"{scope}_ensemble_micro_recall"
                ).to_dict("records")
            )
        top_ensembles = pd.DataFrame(top_ensemble_rows)
    else:
        top_ensembles = pd.DataFrame()

    summary_md = ["# V3 All LLM Runs Summary", ""]
    summary_md.append(f"Prediction files evaluated: {len(manifest[manifest['load_error'].eq('')])}")
    summary_md.append("")
    summary_md.append("## Run Inventory")
    loaded_manifest = manifest[manifest["load_error"].eq("")].copy()
    inventory = (
        loaded_manifest.groupby(["model", "prompt"], as_index=False)
        .agg(runs=("run_id", "count"), min_news=("n_news", "min"), max_news=("n_news", "max"))
        .sort_values(["model", "prompt"])
    )
    summary_md.append(inventory.to_markdown(index=False))
    summary_md.append("")
    summary_md.append("## Candidate File Audit")
    audit_summary = audit.groupby("reason", as_index=False).size().rename(columns={"size": "files"}).sort_values("reason")
    summary_md.append(audit_summary.to_markdown(index=False))
    summary_md.append("")
    summary_md.append("## Top All-Scope Micro-F1")
    top_all = metrics[metrics["scope"].eq("all")].sort_values(["micro_f1", "micro_recall", "n_news"], ascending=False).head(20)
    summary_md.append(
        top_all[
            [
                "run_id",
                "family",
                "model",
                "prompt",
                "n_news",
                "threshold",
                "micro_precision",
                "micro_recall",
                "micro_f1",
                "mean_pred_labels",
                "json_errors",
            ]
        ]
        .round(4)
        .to_markdown(index=False)
    )
    for scope in ["direct", "context", "weak"]:
        summary_md.append("")
        summary_md.append(f"## Top {scope} Recall")
        top_scope = metrics[metrics["scope"].eq(scope)].sort_values(["micro_recall", "micro_f1", "n_news"], ascending=False).head(15)
        summary_md.append(
            top_scope[
                [
                    "run_id",
                    "family",
                    "model",
                    "prompt",
                    "n_news",
                    "threshold",
                    "micro_precision",
                    "micro_recall",
                    "micro_f1",
                    "true_hit_news_coverage",
                    "mean_pred_labels",
                ]
            ]
            .round(4)
            .to_markdown(index=False)
        )
    if not ensemble_metrics.empty:
        summary_md.append("")
        summary_md.append("## Top Pairwise Ensembles By All-Scope Micro-F1")
        top_ens_all = ensemble_metrics[ensemble_metrics["scope"].eq("all")].sort_values(
            ["micro_f1", "micro_recall", "n_news"], ascending=False
        ).head(15)
        summary_md.append(
            top_ens_all[
                [
                    "run_id",
                    "n_news",
                    "ensemble_op",
                    "micro_precision",
                    "micro_recall",
                    "micro_f1",
                    "mean_pred_labels",
                    "ensemble_left",
                    "ensemble_right",
                ]
            ]
            .round(4)
            .to_markdown(index=False)
        )

    manifest.to_csv(OUT / "v3_all_runs_manifest.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(OUT / "v3_all_runs_candidate_files_audit.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "v3_all_runs_threshold_strength_metrics.csv", index=False, encoding="utf-8-sig")
    per_factor.to_csv(OUT / "v3_all_runs_per_factor_by_strength_metrics.csv", index=False, encoding="utf-8-sig")
    sentiment_pressure.to_csv(OUT / "v3_all_runs_sentiment_pressure_metrics.csv", index=False, encoding="utf-8-sig")
    best_by_run.to_csv(OUT / "v3_all_runs_best_by_run.csv", index=False, encoding="utf-8-sig")
    top_configs.to_csv(OUT / "v3_all_runs_top_configs.csv", index=False, encoding="utf-8-sig")
    ensemble_metrics.to_csv(OUT / "v3_all_runs_pairwise_ensemble_metrics.csv", index=False, encoding="utf-8-sig")
    top_ensembles.to_csv(OUT / "v3_all_runs_top_pairwise_ensembles.csv", index=False, encoding="utf-8-sig")
    (OUT / "v3_all_runs_summary.md").write_text("\n".join(summary_md), encoding="utf-8")

    print(f"prediction_files={len(files)}")
    print(f"loaded_runs={len(manifest[manifest['load_error'].eq('')])}")
    print("saved:")
    for name in [
        "v3_all_runs_manifest.csv",
        "v3_all_runs_candidate_files_audit.csv",
        "v3_all_runs_threshold_strength_metrics.csv",
        "v3_all_runs_per_factor_by_strength_metrics.csv",
        "v3_all_runs_sentiment_pressure_metrics.csv",
        "v3_all_runs_best_by_run.csv",
        "v3_all_runs_top_configs.csv",
        "v3_all_runs_pairwise_ensemble_metrics.csv",
        "v3_all_runs_top_pairwise_ensembles.csv",
        "v3_all_runs_summary.md",
    ]:
        print(f"  {OUT / name}")
    print("\nTop all-scope:")
    print(
        top_all[
            [
                "run_id",
                "family",
                "model",
                "prompt",
                "n_news",
                "threshold",
                "micro_precision",
                "micro_recall",
                "micro_f1",
                "mean_pred_labels",
            ]
        ]
        .head(12)
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
