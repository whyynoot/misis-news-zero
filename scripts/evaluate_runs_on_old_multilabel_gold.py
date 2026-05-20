from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyzer.factors import FACTOR_CONFIG  # noqa: E402

OUTPUT_DIR = ROOT / "analysis_outputs"
OLD_GOLD_PATH = OUTPUT_DIR / "interfax_news_multilabel_factor_dataset_1000_wide.csv"
FACTOR_KEYS = [item["key"] for item in FACTOR_CONFIG]
THRESHOLDS: list[float | None] = [None] + [round(x / 100, 2) for x in range(5, 100, 5)]


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


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
    return float(np.mean(scores)) if scores else 0.0


def model_guess(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    if "qwen36_35b_iq3" in normalized or "qwen3.6_35b" in normalized:
        return "qwen3.6-35b:iq3"
    if "qwen35_9b" in normalized:
        return "qwen35-9b"
    if "gemma4_e4b" in normalized:
        return "gemma4:e4b"
    if "gemma4_e2b" in normalized or "gemma4_e2b" in normalized or "gemma4-e2b" in name.lower():
        return "gemma4:e2b"
    if "gemma" in normalized:
        return "gemma"
    return "unknown"


def run_family(name: str) -> str:
    if name.startswith("v4_social_signal_experiment_"):
        return "v4_social_signal"
    if name.startswith("v3_social_signal_experiment_"):
        return "v3_social_signal"
    if name.startswith("prompt_search_") or name.startswith("manual_dataset_prompt_search_"):
        return "prompt_search"
    if name.startswith("thinking_recall_experiment_"):
        return "thinking_recall"
    if name.startswith("grouped_event_"):
        return "grouped_event"
    return "other"


def stable_hash(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def is_candidate(path: Path) -> bool:
    name = path.name
    if not name.endswith(".csv"):
        return False
    if name.startswith("old_gold_"):
        return False
    if any(token in name for token in ("metrics", "manifest", "report", "audit", "summary", "choices", "errors", "dashboard")):
        return False
    try:
        columns = set(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)
    except Exception:
        return False
    return {"dataset_row_id", "factor_key", "relevance"}.issubset(columns) and (
        "is_relevant" in columns or "is_relevant_bool" in columns
    )


def choose_candidate_files() -> list[Path]:
    all_candidates = sorted([path for path in OUTPUT_DIR.glob("*.csv") if is_candidate(path)])
    names = {path.name for path in all_candidates}
    chosen: list[Path] = []
    for path in all_candidates:
        if path.name.endswith(".partial.csv") and path.name.replace(".partial.csv", ".csv") in names:
            continue
        chosen.append(path)
    return chosen


def load_old_gold() -> pd.DataFrame:
    old = pd.read_csv(OLD_GOLD_PATH, encoding="utf-8-sig")
    old["dataset_row_id"] = pd.to_numeric(old.iloc[:, 0], errors="raise").astype(int)
    missing = [f"factor__{key}" for key in FACTOR_KEYS if f"factor__{key}" not in old.columns]
    if missing:
        raise KeyError(f"Old gold dataset missing factor columns: {missing[:10]}")
    return pd.DataFrame(
        {
            key: pd.to_numeric(old[f"factor__{key}"], errors="coerce").fillna(0).astype(int).clip(0, 1).to_numpy()
            for key in FACTOR_KEYS
        },
        index=old["dataset_row_id"],
    )


def evaluate_file(path: Path, y_true_by_id: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pred_raw = pd.read_csv(path, encoding="utf-8-sig")
    pred_raw["dataset_row_id"] = pd.to_numeric(pred_raw["dataset_row_id"], errors="coerce")
    pred_raw = pred_raw.dropna(subset=["dataset_row_id", "factor_key"]).copy()
    pred_raw["dataset_row_id"] = pred_raw["dataset_row_id"].astype(int)
    pred_raw["factor_key"] = pred_raw["factor_key"].astype(str)
    pred_raw = pred_raw[
        pred_raw["dataset_row_id"].isin(y_true_by_id.index) & pred_raw["factor_key"].isin(FACTOR_KEYS)
    ].copy()

    is_relevant_col = "is_relevant_bool" if "is_relevant_bool" in pred_raw.columns else "is_relevant"
    pred_raw["is_relevant_norm"] = as_bool(pred_raw[is_relevant_col])
    pred_raw["relevance_norm"] = pd.to_numeric(pred_raw["relevance"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if "call_status" in pred_raw.columns:
        pred_raw["error_norm"] = pred_raw["call_status"].astype(str).eq("neutral_after_error")
    else:
        pred_raw["error_norm"] = False

    pred = (
        pred_raw.groupby(["dataset_row_id", "factor_key"], as_index=False)
        .agg(relevance=("relevance_norm", "max"), is_relevant=("is_relevant_norm", "max"), error_row=("error_norm", "max"))
        .copy()
    )
    counts = pred.groupby("dataset_row_id")["factor_key"].nunique()
    complete_ids = sorted(counts[counts == len(FACTOR_KEYS)].index.astype(int).tolist())
    pred = pred[pred["dataset_row_id"].isin(complete_ids)].copy()

    y_true = y_true_by_id.loc[complete_ids, FACTOR_KEYS].to_numpy(int)
    supported = y_true.sum(axis=0) > 0
    scores = (
        pred.pivot(index="dataset_row_id", columns="factor_key", values="relevance")
        .reindex(index=complete_ids, columns=FACTOR_KEYS)
        .fillna(0.0)
        .to_numpy(float)
    )
    default_labels = (
        pred.pivot(index="dataset_row_id", columns="factor_key", values="is_relevant")
        .reindex(index=complete_ids, columns=FACTOR_KEYS)
        .fillna(False)
        .astype(int)
        .to_numpy()
    )

    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        if threshold is None:
            labels = default_labels
            threshold_label = "default"
        else:
            labels = (scores >= threshold).astype(int)
            threshold_label = f"{threshold:.2f}"

        micro = precision_recall_fscore_support(y_true, labels, average="micro", zero_division=0)
        any_metric = precision_recall_fscore_support(
            y_true.sum(axis=1) > 0, labels.sum(axis=1) > 0, average="binary", zero_division=0
        )
        rows.append(
            {
                "run_name": path.stem,
                "source_file": path.name,
                "run_family": run_family(path.name),
                "model_guess": model_guess(path.name),
                "file_hash": stable_hash(path),
                "threshold": threshold_label,
                "n_news": len(complete_ids),
                "gold_positive_pairs": int(y_true.sum()),
                "pred_positive_pairs": int(labels.sum()),
                "pred_mean_labels": float(labels.sum(axis=1).mean()) if len(complete_ids) else 0.0,
                "micro_precision": float(micro[0]),
                "micro_recall": float(micro[1]),
                "micro_f1": float(micro[2]),
                "macro_f1_supported": float(
                    f1_score(y_true[:, supported], labels[:, supported], average="macro", zero_division=0)
                )
                if supported.any()
                else 0.0,
                "sample_f1_empty_correct": sample_f1_empty_correct(y_true, labels),
                "any_relevant_precision": float(any_metric[0]),
                "any_relevant_recall": float(any_metric[1]),
                "any_relevant_f1": float(any_metric[2]),
                "false_relevant_news": int(((y_true.sum(axis=1) == 0) & (labels.sum(axis=1) > 0)).sum()),
                "missed_all_relevant_news": int(((y_true.sum(axis=1) > 0) & (labels.sum(axis=1) == 0)).sum()),
                "error_rows": int(pred["error_row"].sum()),
            }
        )

    manifest = {
        "source_file": path.name,
        "run_name": path.stem,
        "run_family": run_family(path.name),
        "model_guess": model_guess(path.name),
        "file_hash": stable_hash(path),
        "n_prediction_rows_raw": len(pred_raw),
        "n_prediction_rows_grouped": len(pred),
        "n_news_complete": len(complete_ids),
        "gold_positive_pairs_complete": int(y_true.sum()) if len(complete_ids) else 0,
        "mtime": path.stat().st_mtime,
        "size_bytes": path.stat().st_size,
    }
    return rows, manifest


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    y_true_by_id = load_old_gold()
    candidate_files = choose_candidate_files()

    all_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for path in candidate_files:
        try:
            rows, manifest = evaluate_file(path, y_true_by_id)
        except Exception as exc:
            manifest_rows.append({"source_file": path.name, "error": repr(exc), "size_bytes": path.stat().st_size})
            continue
        all_rows.extend(rows)
        manifest_rows.append(manifest)

    metrics = pd.DataFrame(all_rows)
    manifest = pd.DataFrame(manifest_rows)
    if metrics.empty:
        raise RuntimeError("No compatible prediction runs were evaluated")

    sort_cols = ["micro_f1", "micro_recall", "micro_precision", "n_news", "pred_positive_pairs"]
    metrics = metrics.sort_values(sort_cols, ascending=[False, False, False, False, False])
    best_by_run = (
        metrics.sort_values(["run_name"] + sort_cols, ascending=[True, False, False, False, False, False])
        .groupby("run_name", as_index=False)
        .head(1)
        .sort_values(sort_cols, ascending=[False, False, False, False, False])
    )

    metrics.to_csv(OUTPUT_DIR / "old_gold_all_runs_metrics.csv", index=False, encoding="utf-8-sig")
    best_by_run.to_csv(OUTPUT_DIR / "old_gold_all_runs_best_by_run.csv", index=False, encoding="utf-8-sig")
    manifest.sort_values(["n_news_complete", "mtime"], ascending=[False, False]).to_csv(
        OUTPUT_DIR / "old_gold_all_runs_manifest.csv", index=False, encoding="utf-8-sig"
    )

    fullish = best_by_run[best_by_run["n_news"] >= 200].copy()
    pilots = best_by_run[best_by_run["n_news"] < 200].copy()
    fullish.to_csv(OUTPUT_DIR / "old_gold_all_runs_best_fullish.csv", index=False, encoding="utf-8-sig")
    pilots.to_csv(OUTPUT_DIR / "old_gold_all_runs_best_pilots.csv", index=False, encoding="utf-8-sig")

    columns = [
        "run_name",
        "model_guess",
        "threshold",
        "n_news",
        "gold_positive_pairs",
        "pred_positive_pairs",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "any_relevant_f1",
    ]
    print(f"candidate_files={len(candidate_files)} evaluated_runs={best_by_run['run_name'].nunique()}")
    print("\nTOP full-ish runs (n_news >= 200)")
    print(fullish[columns].head(25).round(4).to_string(index=False))
    print("\nTOP pilots (n_news < 200)")
    print(pilots[columns].head(20).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
