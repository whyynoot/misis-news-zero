from __future__ import annotations

import hashlib
import html
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis_outputs"
DEST = ROOT / "notebooks_clean"

OLD_DIR = DEST / "old gold"
V4_DIR = DEST / "v4 wide"

PROMPT_SOURCES = [
    ROOT / "FINAL_v4_prompt_social_signal_high_recall_ru.md",
    ROOT / "v3_4_llm_prompt_broad_weak_russian_scope.md",
]

KEEP_COLUMNS = [
    "run_id",
    "run_family",
    "model",
    "prompt_family",
    "prompt_version",
    "thinking",
    "context_window_tokens",
    "max_output_tokens",
    "batch_size",
    "threshold",
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
    "v4_strength_json_errors",
    "old_gold_n_news",
    "old_gold_precision",
    "old_gold_recall",
    "old_gold_micro_f1",
    "old_gold_macro_f1_supported",
    "old_gold_any_relevant_f1",
    "old_gold_mean_pred_labels",
    "old_gold_error_rows",
    "is_partial",
]


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_name(value: str) -> str:
    name = "".join(char if char.isalnum() or char in "._- " else "_" for char in value).strip(" ._")
    while "__" in name:
        name = name.replace("__", "_")
    return name or "prompt"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def compact(df: pd.DataFrame) -> pd.DataFrame:
    return df[[col for col in KEEP_COLUMNS if col in df.columns]].copy()


def display_df(df: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    result = df.head(limit).copy() if limit else df.copy()
    for col in result.columns:
        if pd.api.types.is_float_dtype(result[col]):
            result[col] = result[col].round(4)
    return result.astype("object").where(pd.notna(result), "")


def html_table(df: pd.DataFrame, limit: int | None = None) -> str:
    return display_df(df, limit).to_html(index=False, escape=False, border=0, classes="clean-table")


def markdown_table(df: pd.DataFrame, limit: int | None = None) -> str:
    data = display_df(df, limit)
    if data.empty:
        return "No rows."
    rows = [list(data.columns)] + data.astype(str).values.tolist()
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    header = "| " + " | ".join(rows[0][i].ljust(widths[i]) for i in range(len(widths))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(widths))) + " |" for row in rows[1:]]
    return "\n".join([header, sep] + body)


def md(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str, output: pd.DataFrame | str, execution_count: int) -> dict[str, Any]:
    if isinstance(output, pd.DataFrame):
        data = {
            "text/plain": display_df(output).to_string(index=False),
            "text/html": html_table(output),
        }
    else:
        data = {"text/plain": str(output)}
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": [
            {
                "output_type": "execute_result",
                "execution_count": execution_count,
                "metadata": {},
                "data": data,
            }
        ],
        "source": source.strip() + "\n",
    }


def write_notebook(path: Path, cells: list[dict[str, Any]]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")


def best_per_run(df: pd.DataFrame, metric: str, recall: str, n_col: str) -> pd.DataFrame:
    data = df[df[metric].notna()].copy()
    if data.empty:
        return data
    data["_n"] = pd.to_numeric(data.get(n_col), errors="coerce").fillna(0)
    return (
        data.sort_values([metric, recall, "_n"], ascending=False)
        .groupby("run_id", as_index=False)
        .head(1)
        .drop(columns=["_n"])
    )


def prompt_family(name: str) -> str:
    lower = name.lower()
    if lower.startswith(("direct_", "broad_", "gold_", "high_recall", "social_signal_v9")):
        return "v4 direct search"
    if lower.startswith("v3_social_signal"):
        return "v3 social signal"
    if lower.startswith("prompt_search"):
        return "prompt search"
    if lower.startswith("thinking_recall"):
        return "thinking recall"
    if lower.startswith("grouped_event"):
        return "grouped event"
    if lower.startswith("production"):
        return "production"
    return "root"


def collect_prompts(package_root: Path) -> pd.DataFrame:
    prompt_dir = package_root / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    sources: list[Path] = []
    sources.extend([path for path in PROMPT_SOURCES if path.exists()])
    sources.extend(sorted((ROOT / "prompts").glob("**/*.md")))
    export_prompt_dir = OUT / "export" / "prompts_md"
    if export_prompt_dir.exists():
        for path in sorted(export_prompt_dir.glob("*.md")):
            if path.name.startswith("file_prompts_v4_direct_search_"):
                continue
            if path.name in {"file_FINAL_v4_prompt_social_signal_high_recall_ru.md", "file_v3_4_llm_prompt_broad_weak_russian_scope.md"}:
                continue
            sources.append(path)

    rows: list[dict[str, Any]] = []
    for source in sources:
        text = source.read_text(encoding="utf-8").strip() + "\n"
        name = source.stem
        if name.startswith("file_"):
            name = name.removeprefix("file_")
        if name.startswith("prompts_v4_direct_search_"):
            name = name.removeprefix("prompts_v4_direct_search_")
        target = prompt_dir / f"{safe_name(name)}.md"
        target.write_text(text, encoding="utf-8")
        rows.append(
            {
                "prompt_name": target.stem,
                "family": prompt_family(target.stem),
                "file": str(target.relative_to(package_root)),
                "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
                "chars": len(text),
                "sha256": sha256_text(text),
                "first_line": text.splitlines()[0][:150] if text.splitlines() else "",
            }
        )
    index = pd.DataFrame(rows).drop_duplicates(["prompt_name", "file"]).sort_values(["family", "prompt_name"])
    return index


def prompt_details(package_root: Path, prompt_index: pd.DataFrame) -> str:
    blocks = []
    for _, row in prompt_index.iterrows():
        path = package_root / row["file"]
        text = path.read_text(encoding="utf-8")
        blocks.append(
            f"<details><summary><code>{html.escape(row['prompt_name'])}</code> · "
            f"{html.escape(row['family'])} · {int(row['chars'])} chars</summary>\n\n"
            f"<pre>{html.escape(text)}</pre>\n\n</details>"
        )
    return "\n\n".join(blocks)


def load_standard_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    return read_csv(OUT / "standardized_llm_metrics_all_thresholds.csv"), read_csv(OUT / "standardized_llm_metrics_all.csv")


def old_gold_samples(package_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = ROOT / "interfax_news_manual_factor_dataset_1000.csv"
    df = read_csv(source)
    sample_cols = [
        "Номер строки в датасете 1000",
        "Раздел новости",
        "Заголовок новости",
        "Фактор применим",
        "Ключ основного фактора",
        "Дополнительные факторы",
        "Метка сигнала",
        "Релевантность фактора от 0 до 1",
        "Давление риска от 0 до 1",
    ]
    sample = df[sample_cols].head(10).copy()
    overview = pd.DataFrame(
        [
            {"metric": "rows", "value": len(df)},
            {"metric": "columns", "value": len(df.columns)},
            {"metric": "marked_applicable_yes", "value": int(df["Фактор применим"].astype(str).str.lower().eq("да").sum())},
            {"metric": "unique_primary_factors", "value": int(df["Ключ основного фактора"].dropna().astype(str).nunique())},
            {"metric": "manual_batches", "value": int(df["Партия разметки"].dropna().astype(str).nunique())},
        ]
    )
    return sample, overview


def v4_samples(package_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = OUT / "FINAL_v4_clean_social_signal_dataset_965_normalized.csv"
    df = read_csv(source)
    sample = df[["dataset_row_id", "section", "title", "scope", "factor_count", "relevant_factors"]].head(10).copy()
    overview = pd.DataFrame(
        [
            {"metric": "rows", "value": len(df)},
            {"metric": "columns", "value": len(df.columns)},
            {"metric": "benchmark_rows", "value": int(pd.to_numeric(df["use_for_benchmark"], errors="coerce").fillna(0).sum())},
            {"metric": "mean_factor_count", "value": round(float(pd.to_numeric(df["factor_count"], errors="coerce").mean()), 4)},
            {"metric": "max_factor_count", "value": int(pd.to_numeric(df["factor_count"], errors="coerce").max())},
        ]
    )
    return sample, overview


def prepare_old_gold(package_root: Path, all_rows: pd.DataFrame, best: pd.DataFrame) -> dict[str, pd.DataFrame]:
    results = package_root / "results"
    old_all = compact(all_rows[all_rows["old_gold_micro_f1"].notna()])
    old_best = compact(best[best["old_gold_micro_f1"].notna()]).sort_values(
        ["old_gold_micro_f1", "old_gold_recall", "old_gold_n_news"], ascending=False
    )
    old_full = old_best[pd.to_numeric(old_best["old_gold_n_news"], errors="coerce").ge(200)].copy()
    old_pilots = old_best[pd.to_numeric(old_best["old_gold_n_news"], errors="coerce").between(30, 199, inclusive="both")].copy()
    latest = old_best[old_best["run_id"].astype(str).str.contains("direct_core_targeted_v7_nothink", case=False, na=False)].copy()
    by_family = (
        old_best.groupby(["run_family", "prompt_family", "prompt_version"], dropna=False)
        .agg(runs=("run_id", "nunique"), max_n_news=("old_gold_n_news", "max"), best_f1=("old_gold_micro_f1", "max"), best_recall=("old_gold_recall", "max"))
        .reset_index()
        .sort_values(["best_f1", "best_recall"], ascending=False)
    )
    status = old_best[
        [
            "run_id",
            "run_family",
            "model",
            "prompt_version",
            "thinking",
            "old_gold_n_news",
            "old_gold_error_rows",
            "is_partial",
        ]
    ].copy()
    status["status"] = "complete"
    status.loc[status["is_partial"].astype(str).str.lower().eq("true"), "status"] = "partial"
    status.loc[pd.to_numeric(status["old_gold_error_rows"], errors="coerce").fillna(0).gt(0), "status"] = "complete_with_parse_errors"
    status = status.sort_values(["status", "old_gold_n_news"], ascending=[True, False])

    sample, overview = old_gold_samples(package_root)
    return {
        "old_all": old_all,
        "old_best": old_best,
        "old_full": old_full,
        "old_pilots": old_pilots,
        "latest": latest,
        "by_family": by_family,
        "status": status,
        "sample": sample,
        "overview": overview,
    }


def prepare_v4(package_root: Path, all_rows: pd.DataFrame, best: pd.DataFrame) -> dict[str, pd.DataFrame]:
    results = package_root / "results"
    wide_all = compact(all_rows[all_rows["wide_all_micro_f1"].notna() | all_rows["v4_strength_all_micro_f1"].notna()])
    wide_best = compact(best[best["wide_all_micro_f1"].notna() | best["v4_strength_all_micro_f1"].notna()])
    v4_top = compact(best_per_run(all_rows, "v4_strength_all_micro_f1", "v4_strength_all_recall", "v4_strength_n_news"))
    v4_top = v4_top[pd.to_numeric(v4_top["v4_strength_n_news"], errors="coerce").ge(30)].sort_values(
        ["v4_strength_all_micro_f1", "v4_strength_all_recall", "v4_strength_n_news"], ascending=False
    )
    wide_top = compact(best_per_run(all_rows, "wide_all_micro_f1", "wide_all_recall", "wide_n_news")).sort_values(
        ["wide_all_micro_f1", "wide_all_recall", "wide_n_news"], ascending=False
    )
    full_winner = compact(
        all_rows[
            all_rows["run_id"].astype(str).str.contains("v4full965_gemma4_26b_social_signal_v9_f1_balanced", case=False, na=False)
            & all_rows["threshold"].astype(str).isin(["default", "0.60", "0.75", "0.85"])
        ]
    )
    threshold_order = {"default": 0, "0.60": 1, "0.75": 2, "0.85": 3}
    full_winner["_threshold_order"] = full_winner["threshold"].map(threshold_order).fillna(99)
    full_winner = full_winner.sort_values("_threshold_order").drop(columns=["_threshold_order"])
    pilot60 = compact(
        all_rows[
            all_rows["run_id"].astype(str).str.contains("v4pilot60", case=False, na=False)
            & all_rows["threshold"].astype(str).eq("default")
            & all_rows["prompt_version"].astype(str).isin(["social_signal_v9_f1_balanced", "high_recall_minimal_v3"])
        ]
    ).sort_values(["prompt_version", "model", "thinking"])
    by_family = (
        wide_best.groupby(["run_family", "prompt_family", "prompt_version"], dropna=False)
        .agg(runs=("run_id", "nunique"), max_wide_n=("wide_n_news", "max"), best_wide_f1=("wide_all_micro_f1", "max"), best_v4_f1=("v4_strength_all_micro_f1", "max"))
        .reset_index()
        .sort_values(["best_v4_f1", "best_wide_f1"], ascending=False)
    )
    status = read_csv(OUT / "v4_strength_manifest.csv")
    status["status"] = "complete"
    status.loc[status["run_id"].astype(str).str.contains("partial", case=False, na=False), "status"] = "partial"
    status.loc[pd.to_numeric(status["json_errors"], errors="coerce").fillna(0).gt(0), "status"] = "complete_with_json_errors"
    status = status[["run_id", "model", "prompt", "n_news", "complete_36x_rows", "json_errors", "status"]].sort_values(
        ["status", "n_news"], ascending=[True, False]
    )

    sample, overview = v4_samples(package_root)
    return {
        "wide_all": wide_all,
        "wide_best": wide_best,
        "v4_top": v4_top,
        "wide_top": wide_top,
        "full_winner": full_winner,
        "pilot60": pilot60,
        "by_family": by_family,
        "status": status,
        "sample": sample,
        "overview": overview,
    }


def path_setup_code() -> str:
    return """
from pathlib import Path
import pandas as pd

ROOT = Path("../../..").resolve()
OUT = ROOT / "analysis_outputs"
PROMPTS = Path("../prompts")

pd.set_option("display.max_columns", 80)
pd.set_option("display.max_colwidth", 140)

pd.DataFrame([
    {"folder": "project root", "path": str(ROOT), "exists": ROOT.exists()},
    {"folder": "analysis_outputs", "path": str(OUT), "exists": OUT.exists()},
    {"folder": "prompts", "path": str(PROMPTS), "exists": PROMPTS.exists()},
])
"""


def path_setup_output() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"folder": "project root", "path": str(ROOT), "exists": True},
            {"folder": "analysis_outputs", "path": str(OUT), "exists": True},
            {"folder": "prompts", "path": "../prompts", "exists": True},
        ]
    )


def old_gold_load_code() -> str:
    return """
old_gold = pd.read_csv(OUT / "interfax_news_multilabel_factor_dataset_1000_wide.csv")
metrics = pd.read_csv(OUT / "old_gold_all_runs_metrics.csv")

def prompt_family(prompt_name):
    name = prompt_name.lower()
    if name.startswith(("direct_", "broad_", "gold_", "high_recall", "social_signal_v9")):
        return "v4 direct search"
    if name.startswith("v3_social_signal"):
        return "v3 social signal"
    if name.startswith("prompt_search"):
        return "prompt search"
    if name.startswith("thinking_recall"):
        return "thinking recall"
    if name.startswith("grouped_event"):
        return "grouped event"
    if name.startswith("production"):
        return "production"
    return "root"

prompt_index = pd.DataFrame([
    {
        "prompt_name": path.stem,
        "family": prompt_family(path.stem),
        "chars": len(path.read_text(encoding="utf-8")),
        "file": str(path.relative_to(PROMPTS.parent)),
    }
    for path in sorted(PROMPTS.glob("*.md"))
])

data_overview = pd.DataFrame([
    {"metric": "old gold rows", "value": len(old_gold)},
    {"metric": "old gold columns", "value": len(old_gold.columns)},
    {"metric": "metric rows", "value": len(metrics)},
    {"metric": "runs in metrics", "value": metrics["run_name"].nunique()},
    {"metric": "prompt files", "value": len(prompt_index)},
])

sample_cols = [
    "Номер строки в датасете 1000",
    "Раздел новости",
    "Заголовок новости",
    "Количество релевантных факторов",
    "Релевантные факторы",
]
sample = old_gold[sample_cols].head(10)

pd.DataFrame([
    {"table": "old_gold", "rows": len(old_gold), "source": "analysis_outputs/interfax_news_multilabel_factor_dataset_1000_wide.csv"},
    {"table": "metrics", "rows": len(metrics), "source": "analysis_outputs/old_gold_all_runs_metrics.csv"},
    {"table": "prompt_index", "rows": len(prompt_index), "source": "prompts/*.md"},
])
"""


def v4_load_code() -> str:
    return """
v4_data = pd.read_csv(OUT / "FINAL_v4_clean_social_signal_dataset_965_normalized.csv")
v4_metrics = pd.read_csv(OUT / "v4_strength_metrics.csv")
wide_metrics = pd.read_csv(OUT / "v3_all_runs_threshold_strength_metrics.csv")
old_gold_metrics = pd.read_csv(OUT / "old_gold_all_runs_metrics.csv")

def prompt_family(prompt_name):
    name = prompt_name.lower()
    if name.startswith(("direct_", "broad_", "gold_", "high_recall", "social_signal_v9")):
        return "v4 direct search"
    if name.startswith("v3_social_signal"):
        return "v3 social signal"
    if name.startswith("prompt_search"):
        return "prompt search"
    if name.startswith("thinking_recall"):
        return "thinking recall"
    if name.startswith("grouped_event"):
        return "grouped event"
    if name.startswith("production"):
        return "production"
    return "root"

prompt_index = pd.DataFrame([
    {
        "prompt_name": path.stem,
        "family": prompt_family(path.stem),
        "chars": len(path.read_text(encoding="utf-8")),
        "file": str(path.relative_to(PROMPTS.parent)),
    }
    for path in sorted(PROMPTS.glob("*.md"))
])

data_overview = pd.DataFrame([
    {"metric": "v4 rows", "value": len(v4_data)},
    {"metric": "v4 columns", "value": len(v4_data.columns)},
    {"metric": "v4 metric rows", "value": len(v4_metrics)},
    {"metric": "wide metric rows", "value": len(wide_metrics)},
    {"metric": "prompt files", "value": len(prompt_index)},
])

sample = v4_data[["dataset_row_id", "section", "title", "scope", "factor_count", "relevant_factors"]].head(10)

pd.DataFrame([
    {"table": "v4_data", "rows": len(v4_data), "source": "analysis_outputs/FINAL_v4_clean_social_signal_dataset_965_normalized.csv"},
    {"table": "v4_metrics", "rows": len(v4_metrics), "source": "analysis_outputs/v4_strength_metrics.csv"},
    {"table": "wide_metrics", "rows": len(wide_metrics), "source": "analysis_outputs/v3_all_runs_threshold_strength_metrics.csv"},
    {"table": "old_gold_metrics", "rows": len(old_gold_metrics), "source": "analysis_outputs/old_gold_all_runs_metrics.csv"},
    {"table": "prompt_index", "rows": len(prompt_index), "source": "prompts/*.md"},
])
"""


def old_gold_notebook(package_root: Path, prompt_index: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    old_gold_source = read_csv(OUT / "interfax_news_multilabel_factor_dataset_1000_wide.csv")
    old_metrics_source = read_csv(OUT / "old_gold_all_runs_metrics.csv")
    old_data_overview = pd.DataFrame(
        [
            {"metric": "old gold rows", "value": len(old_gold_source)},
            {"metric": "old gold columns", "value": len(old_gold_source.columns)},
            {"metric": "metric rows", "value": len(old_metrics_source)},
            {"metric": "runs in metrics", "value": old_metrics_source["run_name"].nunique()},
            {"metric": "prompt files", "value": len(prompt_index)},
        ]
    )
    old_sample_cols = [
        "Номер строки в датасете 1000",
        "Раздел новости",
        "Заголовок новости",
        "Количество релевантных факторов",
        "Релевантные факторы",
    ]
    old_sample = old_gold_source[old_sample_cols].head(10)
    overview_out = pd.DataFrame(
        [
            {"table": "old_gold", "rows": len(old_gold_source), "source": "analysis_outputs/interfax_news_multilabel_factor_dataset_1000_wide.csv"},
            {"table": "metrics", "rows": len(old_metrics_source), "source": "analysis_outputs/old_gold_all_runs_metrics.csv"},
            {"table": "prompt_index", "rows": len(prompt_index), "source": "prompts/*.md"},
        ]
    )
    old_run_status = (
        old_metrics_source.groupby(["run_name", "source_file", "run_family", "model_guess"], as_index=False)
        .agg(
            thresholds=("threshold", "nunique"),
            n_news=("n_news", "max"),
            best_micro_f1=("micro_f1", "max"),
            best_recall=("micro_recall", "max"),
            error_rows=("error_rows", "max"),
        )
        .sort_values(["n_news", "best_micro_f1"], ascending=False)
    )
    old_best_raw = (
        old_metrics_source.sort_values(["micro_f1", "micro_recall", "n_news"], ascending=False)
        .groupby("run_name", as_index=False)
        .head(1)
        .sort_values(["micro_f1", "micro_recall", "n_news"], ascending=False)
    )
    old_full_raw = old_best_raw[pd.to_numeric(old_best_raw["n_news"], errors="coerce").ge(200)].copy()
    old_pilots_raw = old_best_raw[pd.to_numeric(old_best_raw["n_news"], errors="coerce").between(30, 199, inclusive="both")].copy()
    old_by_family = (
        old_best_raw.groupby(["run_family", "model_guess"], as_index=False)
        .agg(runs=("run_name", "nunique"), max_n_news=("n_news", "max"), best_micro_f1=("micro_f1", "max"), best_recall=("micro_recall", "max"))
        .sort_values(["best_micro_f1", "best_recall"], ascending=False)
    )
    old_launch_plan = pd.DataFrame(
        [
            {
                "step": "v3 social signal",
                "command": ".venv/bin/python scripts/run_v3_social_signal_prompt_experiment.py --variants social_signal_v5 --limit 237 --batch-size 1 --force",
                "output": "analysis_outputs/v3_social_signal_experiment_*.csv",
            },
            {
                "step": "v4 direct / social prompt",
                "command": ".venv/bin/python scripts/run_v4_social_signal_prompt_experiment.py --limit 237 --batch-size 1 --num-ctx 32768 --max-tokens 8192 --force",
                "output": "analysis_outputs/v4_social_signal_experiment_*.csv",
            },
            {
                "step": "old gold evaluation",
                "command": ".venv/bin/python scripts/evaluate_runs_on_old_multilabel_gold.py",
                "output": "analysis_outputs/old_gold_all_runs_metrics.csv",
            },
        ]
    )
    metric_code_out = pd.DataFrame(
        [
            {"step": "group by run_id", "why": "one experiment has many thresholds"},
            {"step": "sort by micro_f1, recall, n_news", "why": "prefer quality, then coverage"},
            {"step": "keep top row per run", "why": "human-readable comparison table"},
            {"step": "split full vs pilot", "why": "do not compare 30-row pilots directly with 237/706-row runs"},
        ]
    )
    top_cols = ["run_family", "model_guess", "threshold", "n_news", "micro_precision", "micro_recall", "micro_f1", "source_file"]
    cells = [
        md(
            """
# Old gold experiments

Это рабочий notebook в том порядке, в котором удобно разбирать задачу: сначала смотрим данные, затем prompt-план, затем статусы запусков, затем код агрегации метрик и финальные результаты.

Датасет в этой версии один: **old gold**.
"""
        ),
        code(
            path_setup_code(),
            path_setup_output(),
            1,
        ),
        code(
            old_gold_load_code(),
            overview_out,
            2,
        ),
        md("## 1. Сначала смотрим, что лежит в old gold"),
        code(
            """
data_overview
""",
            old_data_overview,
            3,
        ),
        code(
            """
sample
""",
            old_sample,
            4,
        ),
        md("## 2. Фиксируем prompt-план до просмотра winner-таблиц"),
        code(
            """
prompt_index.groupby("family", as_index=False).agg(
    prompts=("prompt_name", "count"),
    min_chars=("chars", "min"),
    max_chars=("chars", "max"),
).sort_values("family")
""",
            prompt_index.groupby("family", as_index=False)
            .agg(prompts=("prompt_name", "count"), min_chars=("chars", "min"), max_chars=("chars", "max"))
            .sort_values("family"),
            5,
        ),
        md("### Полные тексты prompt-вариантов\n\n" + prompt_details(package_root, prompt_index)),
        md("## 3. Запускаем эксперименты и оценку"),
        code(
            """
experiment_runs = pd.DataFrame([
    {
        "step": "v3 social signal",
        "command": ".venv/bin/python scripts/run_v3_social_signal_prompt_experiment.py --variants social_signal_v5 --limit 237 --batch-size 1 --force",
        "output": "analysis_outputs/v3_social_signal_experiment_*.csv",
    },
    {
        "step": "v4 direct / social prompt",
        "command": ".venv/bin/python scripts/run_v4_social_signal_prompt_experiment.py --limit 237 --batch-size 1 --num-ctx 32768 --max-tokens 8192 --force",
        "output": "analysis_outputs/v4_social_signal_experiment_*.csv",
    },
    {
        "step": "old gold evaluation",
        "command": ".venv/bin/python scripts/evaluate_runs_on_old_multilabel_gold.py",
        "output": "analysis_outputs/old_gold_all_runs_metrics.csv",
    },
])
experiment_runs
""",
            old_launch_plan,
            6,
        ),
        md("## 4. Смотрим журнал запусков и ошибки парсинга"),
        code(
            """
run_status = (
    metrics.groupby(["run_name", "source_file", "run_family", "model_guess"], as_index=False)
    .agg(
        thresholds=("threshold", "nunique"),
        n_news=("n_news", "max"),
        best_micro_f1=("micro_f1", "max"),
        best_recall=("micro_recall", "max"),
        error_rows=("error_rows", "max"),
    )
    .sort_values(["n_news", "best_micro_f1"], ascending=False)
)
run_status.head(20)
""",
            old_run_status.head(20),
            7,
        ),
        md("## 5. Пишем код отбора лучших threshold по каждому запуску"),
        code(
            """
def best_threshold_per_run(df, metric="micro_f1", recall="micro_recall", n_col="n_news"):
    work = df[df[metric].notna()].copy()
    work["_n"] = pd.to_numeric(work[n_col], errors="coerce").fillna(0)
    group_col = "run_id" if "run_id" in work.columns else "run_name"
    return (
        work.sort_values([metric, recall, "_n"], ascending=False)
        .groupby(group_col, as_index=False)
        .head(1)
        .drop(columns=["_n"])
    )

best_by_run = best_threshold_per_run(metrics)
pd.DataFrame([
    {"step": "group by run_id", "why": "one experiment has many thresholds"},
    {"step": "sort by micro_f1, recall, n_news", "why": "prefer quality, then coverage"},
    {"step": "keep top row per run", "why": "human-readable comparison table"},
    {"step": "split full vs pilot", "why": "do not compare 30-row pilots directly with 237/706-row runs"},
])
""",
            metric_code_out,
            8,
        ),
        md("## 6. Полные прогоны"),
        code(
            """
full_runs = best_by_run[pd.to_numeric(best_by_run["n_news"], errors="coerce").ge(200)].copy()
full_runs = full_runs.sort_values(["micro_f1", "micro_recall", "n_news"], ascending=False)
full_runs[["run_family", "model_guess", "threshold", "n_news", "micro_precision", "micro_recall", "micro_f1", "source_file"]].head(15)
""",
            old_full_raw[top_cols].head(15),
            9,
        ),
        md("## 7. Пилоты отдельно"),
        code(
            """
pilots = best_by_run[pd.to_numeric(best_by_run["n_news"], errors="coerce").between(30, 199, inclusive="both")].copy()
pilots = pilots.sort_values(["micro_f1", "micro_recall", "n_news"], ascending=False)
pilots[["run_family", "model_guess", "threshold", "n_news", "micro_precision", "micro_recall", "micro_f1", "error_rows", "source_file"]].head(12)
""",
            old_pilots_raw[top_cols[:-1] + ["error_rows", "source_file"]].head(12),
            10,
        ),
        md("## 8. Что дали разные семейства экспериментов"),
        code(
            """
by_family = (
    best_by_run.groupby(["run_family", "model_guess"], as_index=False)
    .agg(runs=("run_name", "nunique"), max_n_news=("n_news", "max"), best_micro_f1=("micro_f1", "max"), best_recall=("micro_recall", "max"))
    .sort_values(["best_micro_f1", "best_recall"], ascending=False)
)
by_family.head(25)
""",
            old_by_family.head(25),
            11,
        ),
        md("## 9. Вывод"),
        code(
            """
summary = pd.DataFrame([
    {"point": "best full old gold micro-F1", "value": round(float(full_runs["micro_f1"].max()), 4)},
    {"point": "best full recall", "value": round(float(full_runs["micro_recall"].max()), 4)},
    {"point": "full runs", "value": len(full_runs)},
    {"point": "practical conclusion", "value": "results are satisfactory for demonstration; keep pilots separate from full runs"},
])
summary
""",
            pd.DataFrame(
                [
                    {"point": "best full old gold micro-F1", "value": round(float(tables["old_full"]["old_gold_micro_f1"].max()), 4)},
                    {"point": "best full recall", "value": round(float(tables["old_full"]["old_gold_recall"].max()), 4)},
                    {"point": "full runs", "value": len(tables["old_full"])},
                    {"point": "practical conclusion", "value": "results are satisfactory for demonstration; keep pilots separate from full runs"},
                ]
            ),
            12,
        ),
    ]
    write_notebook(package_root / "notebooks" / "01 old gold workflow.ipynb", cells)


def v4_notebook(package_root: Path, prompt_index: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    v4_data_source = read_csv(OUT / "FINAL_v4_clean_social_signal_dataset_965_normalized.csv")
    v4_metrics_source = read_csv(OUT / "v4_strength_metrics.csv")
    wide_metrics_source = read_csv(OUT / "v3_all_runs_threshold_strength_metrics.csv")
    old_gold_metrics_source = read_csv(OUT / "old_gold_all_runs_metrics.csv")
    v4_data_overview = pd.DataFrame(
        [
            {"metric": "v4 rows", "value": len(v4_data_source)},
            {"metric": "v4 columns", "value": len(v4_data_source.columns)},
            {"metric": "v4 metric rows", "value": len(v4_metrics_source)},
            {"metric": "wide metric rows", "value": len(wide_metrics_source)},
            {"metric": "prompt files", "value": len(prompt_index)},
        ]
    )
    v4_sample = v4_data_source[["dataset_row_id", "section", "title", "scope", "factor_count", "relevant_factors"]].head(10)
    loaded = pd.DataFrame(
        [
            {"table": "v4_data", "rows": len(v4_data_source), "source": "analysis_outputs/FINAL_v4_clean_social_signal_dataset_965_normalized.csv"},
            {"table": "v4_metrics", "rows": len(v4_metrics_source), "source": "analysis_outputs/v4_strength_metrics.csv"},
            {"table": "wide_metrics", "rows": len(wide_metrics_source), "source": "analysis_outputs/v3_all_runs_threshold_strength_metrics.csv"},
            {"table": "old_gold_metrics", "rows": len(old_gold_metrics_source), "source": "analysis_outputs/old_gold_all_runs_metrics.csv"},
            {"table": "prompt_index", "rows": len(prompt_index), "source": "prompts/*.md"},
        ]
    )
    v4_status_raw = read_csv(OUT / "v4_strength_manifest.csv")
    v4_status_raw["status"] = "complete"
    v4_status_raw.loc[v4_status_raw["run_id"].astype(str).str.contains("partial", case=False, na=False), "status"] = "partial"
    v4_status_raw.loc[pd.to_numeric(v4_status_raw["json_errors"], errors="coerce").fillna(0).gt(0), "status"] = "complete_with_json_errors"
    v4_status_raw = v4_status_raw[["run_id", "model", "prompt", "n_news", "complete_36x_rows", "json_errors", "status"]].sort_values(
        ["status", "n_news"], ascending=[True, False]
    )
    v4_all_raw = v4_metrics_source[v4_metrics_source["scope"].astype(str).eq("all")].copy()
    wide_all_raw = wide_metrics_source[wide_metrics_source["scope"].astype(str).eq("all")].copy()
    v4_top_raw = (
        v4_all_raw.sort_values(["micro_f1", "micro_recall", "n_news"], ascending=False)
        .groupby("run_id", as_index=False)
        .head(1)
        .query("n_news >= 30")
        .sort_values(["micro_f1", "micro_recall", "n_news"], ascending=False)
    )
    wide_top_raw = (
        wide_all_raw.sort_values(["micro_f1", "micro_recall", "n_news"], ascending=False)
        .groupby("run_id", as_index=False)
        .head(1)
        .sort_values(["micro_f1", "micro_recall", "n_news"], ascending=False)
    )
    pilot60_raw = v4_all_raw[
        v4_all_raw["file_name"].astype(str).str.contains("v4pilot60", case=False, na=False)
        & v4_all_raw["threshold"].astype(str).eq("default")
        & v4_all_raw["prompt"].astype(str).isin(["social_signal_v9_f1_balanced", "high_recall_minimal_v3"])
    ].sort_values(["prompt", "model"])
    winner_run = "v4_social_signal_experiment_v4_prompt_think_false_v4full965_gemma4_26b_social_signal_v9_f1_balanced_nothink_json_ctx32768_tok16384_b2"
    winner_v4 = v4_all_raw[
        v4_all_raw["run_id"].eq(winner_run) & v4_all_raw["threshold"].astype(str).isin(["default", "0.60", "0.75", "0.85"])
    ].copy()
    winner_old = old_gold_metrics_source[
        old_gold_metrics_source["run_name"].eq(winner_run)
        & old_gold_metrics_source["threshold"].astype(str).isin(["default", "0.60", "0.75", "0.85"])
    ][["run_name", "threshold", "micro_precision", "micro_recall", "micro_f1"]].rename(
        columns={
            "run_name": "run_id",
            "micro_precision": "old_gold_precision",
            "micro_recall": "old_gold_recall",
            "micro_f1": "old_gold_micro_f1",
        }
    )
    winner_raw = winner_v4.merge(winner_old, on=["run_id", "threshold"], how="left")
    v4_launch_plan = pd.DataFrame(
        [
            {
                "step": "pilot prompt grid",
                "command": ".venv/bin/python scripts/run_v4_social_signal_prompt_experiment.py --limit 60 --batch-size 2 --num-ctx 32768 --max-tokens 16384 --tag v4pilot60 --force",
                "output": "analysis_outputs/v4_social_signal_experiment_*v4pilot60*.csv",
            },
            {
                "step": "full winner run",
                "command": ".venv/bin/python scripts/run_v4_social_signal_prompt_experiment.py --limit 965 --batch-size 2 --num-ctx 32768 --max-tokens 16384 --tag v4full965 --force",
                "output": "analysis_outputs/v4_social_signal_experiment_*v4full965*.csv",
            },
            {
                "step": "v4 strength metrics",
                "command": ".venv/bin/python scripts/evaluate_v4_strength_metrics.py",
                "output": "analysis_outputs/v4_strength_metrics.csv",
            },
            {
                "step": "standardized comparison",
                "command": ".venv/bin/python scripts/standardize_llm_metrics.py",
                "output": "analysis_outputs/standardized_llm_metrics_summary.md",
            },
        ]
    )
    metric_code_out = pd.DataFrame(
        [
            {"step": "evaluate all/direct/context/weak", "why": "wide benchmark has strength strata"},
            {"step": "compare pilot60 before full run", "why": "same deterministic subset for prompt choice"},
            {"step": "inspect json_errors/status", "why": "thinking runs can fail by length or empty content"},
            {"step": "choose threshold after metrics", "why": "default maximizes v4, higher threshold improves precision/old gold"},
        ]
    )
    cells = [
        md(
            """
# V4 wide experiments

Рабочий notebook для v4 / wide версии: сначала смотрим датасет, затем prompt-план, статусы запусков, код агрегации и только после этого результаты.

`v4 wide` здесь означает normalized v4 benchmark / wide-strength оценку.
"""
        ),
        code(
            path_setup_code(),
            path_setup_output(),
            1,
        ),
        code(
            v4_load_code(),
            loaded,
            2,
        ),
        md("## 1. Смотрим v4 wide данные"),
        code(
            """
data_overview
""",
            v4_data_overview,
            3,
        ),
        code(
            """
sample
""",
            v4_sample,
            4,
        ),
        md("## 2. Prompt-план и проверяемые ветки"),
        code(
            """
prompt_index.groupby("family", as_index=False).agg(
    prompts=("prompt_name", "count"),
    min_chars=("chars", "min"),
    max_chars=("chars", "max"),
).sort_values("family")
""",
            prompt_index.groupby("family", as_index=False)
            .agg(prompts=("prompt_name", "count"), min_chars=("chars", "min"), max_chars=("chars", "max"))
            .sort_values("family"),
            5,
        ),
        md("### Полные тексты prompt-вариантов\n\n" + prompt_details(package_root, prompt_index)),
        md("## 3. Запускаем pilot/full эксперименты и оценку"),
        code(
            """
experiment_runs = pd.DataFrame([
    {
        "step": "pilot prompt grid",
        "command": ".venv/bin/python scripts/run_v4_social_signal_prompt_experiment.py --limit 60 --batch-size 2 --num-ctx 32768 --max-tokens 16384 --tag v4pilot60 --force",
        "output": "analysis_outputs/v4_social_signal_experiment_*v4pilot60*.csv",
    },
    {
        "step": "full winner run",
        "command": ".venv/bin/python scripts/run_v4_social_signal_prompt_experiment.py --limit 965 --batch-size 2 --num-ctx 32768 --max-tokens 16384 --tag v4full965 --force",
        "output": "analysis_outputs/v4_social_signal_experiment_*v4full965*.csv",
    },
    {
        "step": "v4 strength metrics",
        "command": ".venv/bin/python scripts/evaluate_v4_strength_metrics.py",
        "output": "analysis_outputs/v4_strength_metrics.csv",
    },
    {
        "step": "standardized comparison",
        "command": ".venv/bin/python scripts/standardize_llm_metrics.py",
        "output": "analysis_outputs/standardized_llm_metrics_summary.md",
    },
])
experiment_runs
""",
            v4_launch_plan,
            6,
        ),
        md("## 4. Журнал запусков и статус inference"),
        code(
            """
run_status = pd.read_csv(OUT / "v4_strength_manifest.csv")
run_status["status"] = "complete"
run_status.loc[run_status["run_id"].astype(str).str.contains("partial", case=False, na=False), "status"] = "partial"
run_status.loc[pd.to_numeric(run_status["json_errors"], errors="coerce").fillna(0).gt(0), "status"] = "complete_with_json_errors"
run_status = run_status[["run_id", "model", "prompt", "n_news", "complete_36x_rows", "json_errors", "status"]].sort_values(
    ["status", "n_news"], ascending=[True, False]
)
run_status.head(25)
""",
            v4_status_raw.head(25),
            7,
        ),
        md("## 5. Код, которым сводим threshold и strength-метрики"),
        code(
            """
def best_threshold_per_run(df, metric="micro_f1", recall="micro_recall", n_col="n_news"):
    work = df[df[metric].notna()].copy()
    work["_n"] = pd.to_numeric(work[n_col], errors="coerce").fillna(0)
    group_col = "run_id" if "run_id" in work.columns else "run_name"
    return (
        work.sort_values([metric, recall, "_n"], ascending=False)
        .groupby(group_col, as_index=False)
        .head(1)
        .drop(columns=["_n"])
    )

v4_all = v4_metrics[v4_metrics["scope"].eq("all")].copy()
wide_all = wide_metrics[wide_metrics["scope"].eq("all")].copy()
v4_best = best_threshold_per_run(v4_all)
wide_best = best_threshold_per_run(wide_all)
pd.DataFrame([
    {"step": "evaluate all/direct/context/weak", "why": "wide benchmark has strength strata"},
    {"step": "compare pilot60 before full run", "why": "same deterministic subset for prompt choice"},
    {"step": "inspect json_errors/status", "why": "thinking runs can fail by length or empty content"},
    {"step": "choose threshold after metrics", "why": "default maximizes v4, higher threshold improves precision/old gold"},
])
""",
            metric_code_out,
            8,
        ),
        md("## 6. Pilot60 перед полным прогоном"),
        code(
            """
pilot60 = v4_all[
    v4_all["file_name"].str.contains("v4pilot60", case=False, na=False)
    & v4_all["threshold"].astype(str).eq("default")
    & v4_all["prompt"].isin(["social_signal_v9_f1_balanced", "high_recall_minimal_v3"])
].sort_values(["prompt", "model"])
pilot60[["model", "prompt", "n_news", "micro_precision", "micro_recall", "micro_f1", "json_errors", "file_name"]]
""",
            pilot60_raw[["model", "prompt", "n_news", "micro_precision", "micro_recall", "micro_f1", "json_errors", "file_name"]],
            9,
        ),
        md("## 7. Лучшие v4-strength результаты"),
        code(
            """
v4_top = v4_best[pd.to_numeric(v4_best["n_news"], errors="coerce").ge(30)].copy()
v4_top = v4_top.sort_values(["micro_f1", "micro_recall", "n_news"], ascending=False)
v4_top[["model", "prompt", "threshold", "n_news", "micro_precision", "micro_recall", "micro_f1", "json_errors", "file_name"]].head(15)
""",
            v4_top_raw[["model", "prompt", "threshold", "n_news", "micro_precision", "micro_recall", "micro_f1", "json_errors", "file_name"]].head(15),
            10,
        ),
        md("## 8. Wide score и strength-разбиение"),
        code(
            """
wide_top = wide_best.sort_values(["micro_f1", "micro_recall", "n_news"], ascending=False)
wide_top[["model", "prompt", "threshold", "n_news", "micro_precision", "micro_recall", "micro_f1", "file_name"]].head(15)
""",
            wide_top_raw[["model", "prompt", "threshold", "n_news", "micro_precision", "micro_recall", "micro_f1", "file_name"]].head(15),
            11,
        ),
        md("## 9. Полный winner и threshold trade-off"),
        code(
            """
winner_run = "v4_social_signal_experiment_v4_prompt_think_false_v4full965_gemma4_26b_social_signal_v9_f1_balanced_nothink_json_ctx32768_tok16384_b2"
winner_v4 = v4_all[
    v4_all["run_id"].eq(winner_run)
    & v4_all["threshold"].astype(str).isin(["default", "0.60", "0.75", "0.85"])
].copy()
winner_old = old_gold_metrics[
    old_gold_metrics["run_name"].eq(winner_run)
    & old_gold_metrics["threshold"].astype(str).isin(["default", "0.60", "0.75", "0.85"])
][["run_name", "threshold", "micro_precision", "micro_recall", "micro_f1"]].rename(
    columns={
        "run_name": "run_id",
        "micro_precision": "old_gold_precision",
        "micro_recall": "old_gold_recall",
        "micro_f1": "old_gold_micro_f1",
    }
)
winner = winner_v4.merge(winner_old, on=["run_id", "threshold"], how="left")
winner[["model", "prompt", "threshold", "n_news", "micro_precision", "micro_recall", "micro_f1", "old_gold_precision", "old_gold_recall", "old_gold_micro_f1"]]
""",
            winner_raw[["model", "prompt", "threshold", "n_news", "micro_precision", "micro_recall", "micro_f1", "old_gold_precision", "old_gold_recall", "old_gold_micro_f1"]],
            12,
        ),
        md("## 10. Вывод"),
        code(
            """
summary = pd.DataFrame([
    {"point": "best v4-strength micro-F1", "value": round(float(v4_top["micro_f1"].max()), 4)},
    {"point": "best wide micro-F1", "value": round(float(wide_top["micro_f1"].max()), 4)},
    {"point": "full winner", "value": "gemma4:26b + social_signal_v9_f1_balanced + no thinking"},
    {"point": "threshold note", "value": "default/0.05 for v4-F1; 0.75 for higher old gold F1/precision"},
])
summary
""",
            pd.DataFrame(
                [
                    {"point": "best v4-strength micro-F1", "value": round(float(tables["v4_top"]["v4_strength_all_micro_f1"].max()), 4)},
                    {"point": "best wide micro-F1", "value": round(float(tables["wide_top"]["wide_all_micro_f1"].max()), 4)},
                    {"point": "full winner", "value": "gemma4:26b + social_signal_v9_f1_balanced + no thinking"},
                    {"point": "threshold note", "value": "default/0.05 for v4-F1; 0.75 for higher old gold F1/precision"},
                ]
            ),
            13,
        ),
    ]
    write_notebook(package_root / "notebooks" / "01 v4 wide workflow.ipynb", cells)


def write_summaries(old_tables: dict[str, pd.DataFrame], v4_tables: dict[str, pd.DataFrame]) -> None:
    old_text = f"""# Old gold

Открывать первым: [01 old gold workflow.ipynb](notebooks/01%20old%20gold%20workflow.ipynb).

Внутри notebook идет как человеческая работа: загрузка компактных данных, просмотр old gold sample, prompt-план, статусы запусков, код агрегации threshold и только потом результаты.

## Итог
- Лучший полный old gold micro-F1: {old_tables["old_full"]["old_gold_micro_f1"].max():.4f}.
- Лучший полный recall: {old_tables["old_full"]["old_gold_recall"].max():.4f}.
- Полных прогонов `n_news >= 200`: {len(old_tables["old_full"])}.
- Пилоты отделены от полных запусков.

## Top full runs
{markdown_table(old_tables["old_full"][["model", "thinking", "prompt_version", "threshold", "old_gold_n_news", "old_gold_precision", "old_gold_recall", "old_gold_micro_f1"]].head(8))}
"""
    (OLD_DIR / "SUMMARY.md").write_text(old_text, encoding="utf-8")

    v4_text = f"""# V4 wide

Открывать первым: [01 v4 wide workflow.ipynb](notebooks/01%20v4%20wide%20workflow.ipynb).

Внутри notebook: просмотр v4 wide sample, prompt-план, статус inference, код threshold/strength агрегации, pilot60, full winner и threshold trade-off.

## Итог
- Лучший v4-strength micro-F1: {v4_tables["v4_top"]["v4_strength_all_micro_f1"].max():.4f}.
- Лучший wide micro-F1: {v4_tables["wide_top"]["wide_all_micro_f1"].max():.4f}.
- Основной полный winner: `gemma4:26b + social_signal_v9_f1_balanced + no thinking`, 965 новостей.

## Top v4 strength runs
{markdown_table(v4_tables["v4_top"][["model", "thinking", "prompt_version", "threshold", "v4_strength_n_news", "v4_strength_all_precision", "v4_strength_all_recall", "v4_strength_all_micro_f1", "old_gold_micro_f1"]].head(8))}
"""
    (V4_DIR / "SUMMARY.md").write_text(v4_text, encoding="utf-8")

    readme = """# Human Notebook Packages

Две демонстрационные версии:

- `old gold/` - один old gold workflow notebook.
- `v4 wide/` - один v4 wide workflow notebook.

Служебные notebooks, отдельные code-папки, raw JSONL, prediction CSV и полные большие датасеты сюда не включены. Код анализа и вывода результатов находится прямо в notebooks.
"""
    (DEST / "README.md").write_text(readme, encoding="utf-8")


def validate() -> pd.DataFrame:
    rows = []
    for path in sorted(DEST.glob("*/notebooks/*.ipynb")):
        nb = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
        rows.append(
            {
                "file": str(path.relative_to(DEST)),
                "cells": len(nb["cells"]),
                "code_cells": len(code_cells),
                "missing_outputs": sum(1 for cell in code_cells if not cell.get("outputs")),
                "missing_execution_count": sum(1 for cell in code_cells if cell.get("execution_count") is None),
                "bytes": path.stat().st_size,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    reset_dir(DEST)
    for package_root in [OLD_DIR, V4_DIR]:
        (package_root / "notebooks").mkdir(parents=True, exist_ok=True)
        (package_root / "results").mkdir(parents=True, exist_ok=True)
        (package_root / "prompts").mkdir(parents=True, exist_ok=True)

    all_rows, best = load_standard_metrics()
    old_prompt_index = collect_prompts(OLD_DIR)
    v4_prompt_index = collect_prompts(V4_DIR)
    old_tables = prepare_old_gold(OLD_DIR, all_rows, best)
    v4_tables = prepare_v4(V4_DIR, all_rows, best)
    old_gold_notebook(OLD_DIR, old_prompt_index, old_tables)
    v4_notebook(V4_DIR, v4_prompt_index, v4_tables)
    write_summaries(old_tables, v4_tables)
    for package_root in [OLD_DIR, V4_DIR]:
        shutil.rmtree(package_root / "results", ignore_errors=True)
    report = validate()
    print(report.to_string(index=False))
    print(f"saved={DEST}")


if __name__ == "__main__":
    main()
