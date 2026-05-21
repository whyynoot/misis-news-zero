from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyzer.factors import FACTOR_CONFIG

OUT = ROOT / "analysis_outputs"
DATASET = OUT / "FINAL_v4_clean_social_signal_dataset_965_normalized.csv"
THRESHOLDS = [None] + [round(float(value), 2) for value in np.arange(0.05, 1.00, 0.05)]
SCOPES = ["all", "direct", "context", "weak", "direct_context"]
SEARCH_PROMPT_TOKENS = [
    "social_signal_v9_f1_balanced",
    "social_signal_v5",
    "high_recall_minimal_v3",
    "high_recall_minimal_v2",
    "high_recall_minimal_v1",
    "direct_strict",
    "direct_recall",
    "direct_checklist",
    "direct_negative",
    "direct_fewshot",
    "direct_minimal",
    "direct_brief_thinking",
    "direct_business_migration_fixed",
    "direct_core_recall_v2",
    "direct_core_filtered_v3",
    "direct_core_gold_calibrated_v4",
    "direct_core_gold_fewshot_v5",
    "direct_core_clean_fewshot_v6",
    "direct_core_targeted_v7",
    "direct_core_balanced_v8",
    "gold_mimic_broad_v1",
    "direct_taxonomy",
    "direct_precision",
    "broad_balanced",
    "broad_recall",
    "broad_weak_context",
]


def df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        rows = [list(df.columns)] + df.fillna("").astype(str).values.tolist()
        if not rows:
            return ""
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        header = "| " + " | ".join(rows[0][i].ljust(widths[i]) for i in range(len(widths))) + " |"
        sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
        body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(widths))) + " |" for row in rows[1:]]
        return "\n".join([header, sep] + body)


def infer_model(name: str) -> str:
    lowered = name.lower()
    if "qwen36_35b_iq4" in lowered:
        return "batiai/qwen3.6-35b:iq4"
    if "qwen36_35b" in lowered or "qwen3.6-35b" in lowered:
        return "batiai/qwen3.6-35b:iq3"
    if "qwen36_27b" in lowered or "qwen3.6_27b" in lowered:
        return "qwen3.6:27b"
    if "qwen35_27b" in lowered or "qwen3.5_27b" in lowered:
        return "qwen3.5:27b"
    if "qwen35_9b" in lowered or "qwen3.5" in lowered:
        return "qwen3.5:9b"
    if "qwen3_14b" in lowered or "qwen3-14b" in lowered:
        return "qwen3:14b"
    if "gemma4_e4b" in lowered or "gemma4-e4b" in lowered:
        return "gemma4:e4b"
    if "gemma4_e2b" in lowered or "gemma4-e2b" in lowered:
        return "gemma4:e2b"
    if "gemma4_26b" in lowered or "gemma4-26b" in lowered:
        return "gemma4:26b"
    if "gemma3n_e4b" in lowered or "gemma3n-e4b" in lowered:
        return "gemma3n:e4b"
    return "unknown"


def infer_prompt(name: str) -> str:
    lowered = name.lower()
    known = [
        "social_signal_v9_f1_balanced",
        "social_signal_v5",
        "high_recall_minimal_v3",
        "high_recall_minimal_v2",
        "high_recall_minimal_v1",
        "direct_strict",
        "direct_recall",
        "direct_checklist",
        "direct_negative",
        "direct_fewshot",
        "direct_minimal",
        "direct_brief_thinking",
        "direct_business_migration_fixed",
        "direct_core_recall_v2",
        "direct_core_filtered_v3",
        "direct_core_gold_calibrated_v4",
        "direct_core_gold_fewshot_v5",
        "direct_core_clean_fewshot_v6",
        "direct_core_targeted_v7",
        "direct_core_balanced_v8",
        "gold_mimic_broad_v1",
        "direct_taxonomy",
        "direct_precision",
        "broad_balanced",
        "broad_recall",
        "broad_weak_context",
        "v4_prompt",
    ]
    for item in known:
        if item in lowered:
            return item
    return "unknown"


def normalize_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return series.fillna("").astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "да"})


def raw_error_count(path: Path) -> int | None:
    raw_path = path.with_suffix(".raw.jsonl")
    if not raw_path.exists():
        return None
    errors = 0
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            if json.loads(line).get("status") != "ok":
                errors += 1
        except json.JSONDecodeError:
            errors += 1
    return errors


def candidate_prediction_files() -> list[Path]:
    files = []
    for path in sorted(OUT.glob("v4_social_signal_experiment_*.csv")):
        if path.name.startswith("v4_social_signal_experiment_metrics_"):
            continue
        if not any(token in path.name.lower() for token in SEARCH_PROMPT_TOKENS):
            continue
        if path.name.endswith(".partial.csv") and path.with_name(path.name.replace(".partial.csv", ".csv")).exists():
            continue
        try:
            columns = set(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        if {"dataset_row_id", "factor_key"}.issubset(columns) and ({"relevance", "is_relevant"} & columns):
            files.append(path)
    return files


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


def gold_scope_matrix(all_gold: np.ndarray, strength: np.ndarray, scope: str) -> np.ndarray:
    if scope == "all":
        return all_gold.astype(int)
    if scope == "direct_context":
        return ((strength == "direct") | (strength == "context")).astype(int)
    return (strength == scope).astype(int)


def evaluate_scope(meta: dict[str, Any], threshold: str, labels: np.ndarray, gold_all: np.ndarray, strength: np.ndarray, scope: str) -> dict[str, Any]:
    gold = gold_scope_matrix(gold_all, strength, scope)
    precision, recall, f1, _ = precision_recall_fscore_support(gold, labels, average="micro", zero_division=0)
    supported = gold.sum(axis=0) > 0
    macro = precision_recall_fscore_support(gold[:, supported], labels[:, supported], average="macro", zero_division=0)[2] if supported.any() else 0.0
    gold_any = gold.sum(axis=1) > 0
    pred_any = labels.sum(axis=1) > 0
    any_p, any_r, any_f1, _ = precision_recall_fscore_support(gold_any, pred_any, average="binary", zero_division=0)
    tp = int(((gold == 1) & (labels == 1)).sum())
    fp = int(((gold == 0) & (labels == 1)).sum())
    fn = int(((gold == 1) & (labels == 0)).sum())
    return {
        **meta,
        "threshold": threshold,
        "scope": scope,
        "gold_pairs": int(gold.sum()),
        "pred_pairs": int(labels.sum()),
        "tp_pairs": tp,
        "fp_pairs_vs_scope": fp,
        "fn_pairs": fn,
        "mean_pred_labels": float(labels.sum(axis=1).mean()),
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "micro_f1": float(f1),
        "macro_f1_supported": float(macro),
        "sample_f1": sample_f1(gold, labels),
        "any_precision": float(any_p),
        "any_recall": float(any_r),
        "any_f1": float(any_f1),
        "gold_news": int(gold_any.sum()),
        "pred_any_news": int(pred_any.sum()),
        "missed_all_scope_news": int((gold_any & ~pred_any).sum()),
        "false_relevant_news": int((~gold_any & pred_any).sum()),
    }


def main() -> None:
    df = pd.read_csv(DATASET, encoding="utf-8-sig")
    factor_keys = [item["key"] for item in FACTOR_CONFIG]
    df["dataset_row_id"] = pd.to_numeric(df["dataset_row_id"], errors="raise").astype(int)
    eval_ids = sorted(df.loc[df.get("use_for_benchmark", 1).astype(int).eq(1), "dataset_row_id"].tolist())
    indexed = df.set_index("dataset_row_id").loc[eval_ids]
    gold_all = np.column_stack([
        pd.to_numeric(indexed[f"factor__{key}"], errors="coerce").fillna(0).astype(int).clip(0, 1).to_numpy()
        for key in factor_keys
    ])
    strength = np.column_stack([
        indexed[f"strength__{key}"].fillna("").astype(str).str.lower().to_numpy()
        for key in factor_keys
    ])

    rows = []
    manifest = []
    for path in candidate_prediction_files():
        pred = pd.read_csv(path)
        pred["dataset_row_id"] = pd.to_numeric(pred["dataset_row_id"], errors="coerce")
        pred = pred[pred["dataset_row_id"].notna()].copy()
        pred["dataset_row_id"] = pred["dataset_row_id"].astype(int)
        pred = pred[pred["dataset_row_id"].isin(eval_ids) & pred["factor_key"].isin(factor_keys)].copy()
        pred = pred.drop_duplicates(["dataset_row_id", "factor_key"], keep="last")
        covered_ids = sorted(set(pred["dataset_row_id"].tolist()) & set(eval_ids))
        if not covered_ids:
            continue
        positions = [eval_ids.index(item) for item in covered_ids]
        scores = (
            pred.assign(relevance=pd.to_numeric(pred["relevance"], errors="coerce").fillna(0.0))
            .pivot(index="dataset_row_id", columns="factor_key", values="relevance")
            .reindex(index=covered_ids, columns=factor_keys)
            .fillna(0.0)
            .to_numpy(float)
        )
        if "is_relevant" in pred.columns:
            pred["is_rel"] = normalize_bool(pred["is_relevant"])
        else:
            pred["is_rel"] = pred["relevance"].ge(0.3)
        default_labels = (
            pred.pivot(index="dataset_row_id", columns="factor_key", values="is_rel")
            .reindex(index=covered_ids, columns=factor_keys)
            .fillna(False)
            .astype(int)
            .to_numpy()
        )
        meta = {
            "run_id": path.stem,
            "file_name": path.name,
            "model": infer_model(path.name),
            "prompt": infer_prompt(path.name),
            "n_news": len(covered_ids),
            "n_rows": len(pred),
            "complete_36x_rows": len(pred) == len(covered_ids) * len(factor_keys),
            "json_errors": raw_error_count(path),
        }
        manifest.append({**meta, "path": str(path)})
        for threshold in THRESHOLDS:
            if threshold is None:
                threshold_label = "default"
                labels = default_labels
            else:
                threshold_label = f"{threshold:.2f}"
                labels = (scores >= threshold).astype(int)
            for scope in SCOPES:
                rows.append(evaluate_scope(meta, threshold_label, labels, gold_all[positions], strength[positions], scope))

    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        metrics = metrics.sort_values(["scope", "micro_f1", "micro_recall"], ascending=[True, False, False])
    manifest_df = pd.DataFrame(manifest)
    if not manifest_df.empty:
        manifest_df = manifest_df.sort_values(["model", "prompt", "n_news", "file_name"])
    metrics.to_csv(OUT / "v4_strength_metrics.csv", index=False, encoding="utf-8-sig")
    manifest_df.to_csv(OUT / "v4_strength_manifest.csv", index=False, encoding="utf-8-sig")

    summary = ["# V4 Strength Metrics", "", f"Runs: {len(manifest_df)}", ""]
    for scope in SCOPES:
        summary.append(f"## Top {scope} micro-F1")
        if metrics.empty:
            summary.append("No completed direct-search runs yet.")
        else:
            top = metrics[metrics["scope"].eq(scope)].head(20)
            summary.append(
                df_to_markdown(
                    top[
                        [
                            "model",
                            "prompt",
                            "n_news",
                            "threshold",
                            "micro_precision",
                            "micro_recall",
                            "micro_f1",
                            "mean_pred_labels",
                            "file_name",
                        ]
                    ].round(4)
                )
            )
        summary.append("")
    (OUT / "v4_strength_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"runs={len(manifest_df)}")
    print(f"saved={OUT / 'v4_strength_metrics.csv'}")
    if not metrics.empty:
        print(metrics[metrics["scope"].eq("direct")].head(12).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
