from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis_outputs"
EXPORT_ROOT = OUT / "export"
PROMPTS_DIR = EXPORT_ROOT / "prompts_md"
METRICS_DIR = EXPORT_ROOT / "metrics"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NEWS_PLACEHOLDER = [
    {
        "news_id": "{NEWS_ID}",
        "date": "{DATE}",
        "title": "{TITLE}",
        "summary": "{SUMMARY}",
        "text": "{NEWS_TEXT}",
        "url": "{URL}",
    }
]

SUMMARY_INPUT_PLACEHOLDER = {
    "period_start": "{PERIOD_START}",
    "period_end": "{PERIOD_END}",
    "factors": [
        {
            "factor_id": "{FACTOR_ID}",
            "metrics": "{AGGREGATED_METRICS}",
            "top_news": ["{TOP_NEWS_ITEM}"],
        }
    ],
}

FACTOR_CATALOG_PLACEHOLDER = [
    {
        "factor_id": "{FACTOR_ID}",
        "name": "{FACTOR_NAME}",
        "description": "{FACTOR_DESCRIPTION}",
        "positive_signal": "{POSITIVE_SIGNAL}",
        "negative_signal": "{NEGATIVE_SIGNAL}",
    }
]

SOCIAL_SIGNAL_CATALOG_PLACEHOLDER = [
    {
        "factor_id": "{FACTOR_ID}",
        "dashboard_name": "{DASHBOARD_NAME}",
        "social_signal_meaning": "{SOCIAL_SIGNAL_MEANING}",
        "count_as_signal_when": "{COUNT_AS_SIGNAL_WHEN}",
        "do_not_use_when": "{DO_NOT_USE_WHEN}",
        "positive_signal": "{POSITIVE_SIGNAL}",
        "negative_signal": "{NEGATIVE_SIGNAL}",
    }
]

EXAMPLES_PLACEHOLDER = [
    {
        "news": "{EXAMPLE_NEWS_TITLE}",
        "expected_factors": ["{FACTOR_ID}"],
        "logic": "{EXAMPLE_LOGIC}",
    }
]


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9а-яА-ЯёЁ._-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("._")
    return value or "prompt"


def reset_dir(path: Path) -> None:
    if path.exists():
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def write_prompt(records: list[dict[str, str]], name: str, text: str, source: str, note: str = "") -> None:
    filename = safe_filename(name)
    if not filename.endswith(".md"):
        filename += ".md"
    path = PROMPTS_DIR / filename
    content = text.strip() + "\n"
    path.write_text(content, encoding="utf-8")
    records.append(
        {
            "file": str(path.relative_to(EXPORT_ROOT)),
            "source": source,
            "note": note,
            "chars": str(len(content)),
            "sha256": sha256_text(content),
        }
    )


def copy_prompt_md(records: list[dict[str, str]]) -> None:
    prompt_paths = sorted((ROOT / "prompts").glob("**/*.md"))
    root_prompt_paths = [
        ROOT / "FINAL_v4_prompt_social_signal_high_recall_ru.md",
        ROOT / "v3_4_llm_prompt_broad_weak_russian_scope.md",
    ]

    for source_path in prompt_paths:
        rel = source_path.relative_to(ROOT)
        target_name = "file__" + "__".join(rel.with_suffix("").parts) + ".md"
        write_prompt(
            records,
            target_name,
            source_path.read_text(encoding="utf-8"),
            str(rel),
            "existing markdown prompt",
        )

    for source_path in root_prompt_paths:
        if not source_path.exists():
            continue
        rel = source_path.relative_to(ROOT)
        write_prompt(
            records,
            "file__" + rel.stem + ".md",
            source_path.read_text(encoding="utf-8"),
            str(rel),
            "existing root markdown prompt",
        )


def import_module_from_path(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_lightweight_stubs() -> None:
    if "pandas" not in sys.modules:
        pandas_stub = types.ModuleType("pandas")
        pandas_stub.isna = lambda value: value is None
        sys.modules["pandas"] = pandas_stub

    if "numpy" not in sys.modules:
        numpy_stub = types.ModuleType("numpy")
        sys.modules["numpy"] = numpy_stub

    if "sklearn" not in sys.modules:
        sklearn_stub = types.ModuleType("sklearn")
        metrics_stub = types.ModuleType("sklearn.metrics")

        def unavailable(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("Metric functions are not needed for prompt export.")

        metrics_stub.f1_score = unavailable
        metrics_stub.precision_recall_fscore_support = unavailable
        sklearn_stub.metrics = metrics_stub
        sys.modules["sklearn"] = sklearn_stub
        sys.modules["sklearn.metrics"] = metrics_stub

    if "nbformat" not in sys.modules:
        nbformat_stub = types.ModuleType("nbformat")

        class V4:
            @staticmethod
            def new_markdown_cell(source: str) -> dict[str, str]:
                return {"cell_type": "markdown", "source": source}

            @staticmethod
            def new_code_cell(source: str) -> dict[str, str]:
                return {"cell_type": "code", "source": source}

            @staticmethod
            def new_notebook(cells: list[dict[str, str]]) -> dict[str, Any]:
                return {"cells": cells}

        nbformat_stub.v4 = V4
        nbformat_stub.write = lambda *_args, **_kwargs: None
        sys.modules["nbformat"] = nbformat_stub


def export_production_prompts(records: list[dict[str, str]]) -> None:
    from analyzer.llm_prompts import (
        CLASSIFICATION_SYSTEM_PROMPT,
        SUMMARY_SYSTEM_PROMPT,
        build_classification_user_prompt,
        build_summary_user_prompt,
    )

    write_prompt(
        records,
        "production__classification_system_prompt.md",
        CLASSIFICATION_SYSTEM_PROMPT,
        "analyzer/llm_prompts.py::CLASSIFICATION_SYSTEM_PROMPT",
    )
    write_prompt(
        records,
        "production__classification_user_prompt_template.md",
        build_classification_user_prompt(NEWS_PLACEHOLDER, FACTOR_CATALOG_PLACEHOLDER),
        "analyzer/llm_prompts.py::build_classification_user_prompt",
        "rendered with placeholder news/catalog JSON",
    )
    write_prompt(
        records,
        "production__summary_system_prompt.md",
        SUMMARY_SYSTEM_PROMPT,
        "analyzer/llm_prompts.py::SUMMARY_SYSTEM_PROMPT",
    )
    write_prompt(
        records,
        "production__summary_user_prompt_template.md",
        build_summary_user_prompt(SUMMARY_INPUT_PLACEHOLDER),
        "analyzer/llm_prompts.py::build_summary_user_prompt",
        "rendered with placeholder summary input JSON",
    )


def export_prompt_search_variants(records: list[dict[str, str]]) -> None:
    module = import_module_from_path(
        "export_build_prompt_search_notebook",
        ROOT / "scripts" / "build_prompt_search_notebook.py",
    )
    prompt_cell = next(cell["source"] for cell in module.cells if "def prompt_balanced_recall" in cell["source"])
    prompt_code = prompt_cell[
        prompt_cell.index("def prompt_production") : prompt_cell.index("PROMPT_VARIANTS =")
    ].strip()

    from analyzer.llm_prompts import build_classification_user_prompt

    namespace: dict[str, Any] = {
        "build_classification_user_prompt": build_classification_user_prompt,
        "compact_json": compact_json,
    }
    exec(prompt_code, namespace)

    variants: list[tuple[str, str, Callable[[list[dict[str, Any]], list[dict[str, Any]]], str]]] = [
        ("production_v1", "Текущий production prompt", namespace["prompt_production"]),
        ("balanced_recall_v2", "Больше recall по факторам, но с not_relevant gate", namespace["prompt_balanced_recall"]),
        ("primary_first_v2", "Сначала выбрать primary factor/not_relevant, затем факторы", namespace["prompt_primary_first"]),
        ("hard_negative_v2", "Больше hard negatives, чтобы не потерять specificity", namespace["prompt_hard_negative"]),
        ("few_shot_major_v3", "Явные мини-примеры по главным классам и hard negative", namespace["prompt_few_shot_major"]),
    ]
    for name, note, builder in variants:
        write_prompt(
            records,
            f"prompt_search__{name}.md",
            builder(NEWS_PLACEHOLDER, FACTOR_CATALOG_PLACEHOLDER),
            "scripts/build_prompt_search_notebook.py::Prompt variants",
            f"{note}; rendered with placeholder news/catalog JSON",
        )


def export_v3_social_signal_variants(records: list[dict[str, str]]) -> None:
    module = import_module_from_path(
        "export_run_v3_social_signal_prompt_experiment",
        ROOT / "scripts" / "run_v3_social_signal_prompt_experiment.py",
    )
    for variant in ["social_signal_v1", "social_signal_v2", "social_signal_v3", "social_signal_v4", "social_signal_v5"]:
        text = module.social_signal_prompt(
            variant=variant,
            news_items=NEWS_PLACEHOLDER,
            catalog=SOCIAL_SIGNAL_CATALOG_PLACEHOLDER,
            examples=EXAMPLES_PLACEHOLDER,
        )
        write_prompt(
            records,
            f"v3_social_signal__{variant}.md",
            text,
            "scripts/run_v3_social_signal_prompt_experiment.py::social_signal_prompt",
            "rendered with placeholder news/catalog/example JSON",
        )


def export_thinking_recall_variants(records: list[dict[str, str]]) -> None:
    module = import_module_from_path(
        "export_run_thinking_recall_experiment",
        ROOT / "scripts" / "run_thinking_recall_experiment.py",
    )
    builders: dict[str, Callable[[list[dict[str, Any]], list[dict[str, Any]], Callable[[Any], str]], str]] = {
        "recall_max_v1": module.high_recall_prompt,
        "latent_candidate_v2": module.latent_candidate_prompt,
        "latent_candidate_v3": module.latent_candidate_v3_prompt,
    }
    for base_name, builder in builders.items():
        text = builder(NEWS_PLACEHOLDER, SOCIAL_SIGNAL_CATALOG_PLACEHOLDER, compact_json)
        for think_flag in ["think_false", "think_true"]:
            write_prompt(
                records,
                f"thinking_recall__{base_name}_{think_flag}.md",
                text,
                f"scripts/run_thinking_recall_experiment.py::{builder.__name__}",
                f"same prompt text for {think_flag}; runtime thinking flag differs",
            )


def export_grouped_event_prompt(records: list[dict[str, str]]) -> None:
    module = import_module_from_path(
        "export_run_grouped_event_taxonomy_experiment",
        ROOT / "scripts" / "run_grouped_event_taxonomy_experiment.py",
    )
    text = module.build_group_prompt(
        group_name="{GROUP_ID}",
        group_title="{GROUP_TITLE}",
        news_items=NEWS_PLACEHOLDER,
        factors_payload=SOCIAL_SIGNAL_CATALOG_PLACEHOLDER,
        positive_examples=[{"title": "{POSITIVE_EXAMPLE_TITLE}", "expected_factors_in_this_group": ["{FACTOR_ID}"]}],
        negative_examples=[{"title": "{NEGATIVE_EXAMPLE_TITLE}", "expected_factors_in_this_group": []}],
    )
    write_prompt(
        records,
        "grouped_event__grouped_event_v1.md",
        text,
        "scripts/run_grouped_event_taxonomy_experiment.py::build_group_prompt",
        "rendered with placeholder group/news/catalog/example JSON",
    )


def metric_export_candidates() -> list[Path]:
    patterns = [
        "*metrics*.csv",
        "*metrics*.json",
        "*summary*.json",
        "*summary*.md",
        "*report*.csv",
        "*threshold*.csv",
        "*manifest*.csv",
        "*best*.csv",
        "*top*.csv",
        "*comparison*.csv",
        "*ensemble*.csv",
        "*sentiment*.csv",
        "*breakdown*.csv",
        "*audit*.csv",
        "*choices*.csv",
        "*diff*.csv",
        "*errors*.csv",
        "*raw_status*.csv",
    ]
    files: set[Path] = set()
    for pattern in patterns:
        files.update(path for path in OUT.glob(pattern) if path.is_file())
    excluded_suffixes = (".partial", ".raw", ".log", ".pid", ".aborted", ".stopped", ".pre_thinking_log")
    return sorted(path for path in files if not path.name.endswith(excluded_suffixes))


def export_metrics() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for source_path in metric_export_candidates():
        target_path = METRICS_DIR / source_path.name
        shutil.copy2(source_path, target_path)
        records.append(
            {
                "file": str(target_path.relative_to(EXPORT_ROOT)),
                "source": str(source_path.relative_to(ROOT)),
                "bytes": str(target_path.stat().st_size),
                "sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
            }
        )
    return records


def write_csv_index(path: Path, records: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_readme(prompt_records: list[dict[str, str]], metric_records: list[dict[str, str]]) -> None:
    prompt_rows = "\n".join(
        f"| `{record['file']}` | `{record['source']}` | {record.get('note', '')} |"
        for record in prompt_records
    )
    metric_rows = "\n".join(
        f"| `{record['file']}` | `{record['source']}` | {record.get('bytes', '')} |"
        for record in metric_records
    )
    readme = f"""# Prompts And Metrics Export

Generated by `scripts/export_prompts_and_metrics.py`.

Prompt files are in `prompts_md/`.
Metric files are in `metrics/`.

Dynamic prompts were rendered with placeholder JSON blocks such as `{{NEWS_ID}}`, `{{NEWS_TEXT}}`, `{{FACTOR_ID}}`, and `{{GROUP_ID}}`.
Runtime flags like `think_true` and `think_false` are represented as separate files when they appeared as separate run variants, even when the prompt text itself is identical.

## Prompts

Total: {len(prompt_records)}

| file | source | note |
|---|---|---|
{prompt_rows}

## Metrics

Total: {len(metric_records)}

| file | source | bytes |
|---|---|---:|
{metric_rows}
"""
    (EXPORT_ROOT / "README.md").write_text(readme.strip() + "\n", encoding="utf-8")


def main() -> None:
    reset_dir(PROMPTS_DIR)
    reset_dir(METRICS_DIR)
    install_lightweight_stubs()

    prompt_records: list[dict[str, str]] = []
    copy_prompt_md(prompt_records)
    export_production_prompts(prompt_records)
    export_prompt_search_variants(prompt_records)
    export_v3_social_signal_variants(prompt_records)
    export_thinking_recall_variants(prompt_records)
    export_grouped_event_prompt(prompt_records)

    prompt_records.sort(key=lambda item: item["file"])
    metric_records = export_metrics()
    metric_records.sort(key=lambda item: item["file"])

    write_csv_index(EXPORT_ROOT / "prompts_index.csv", prompt_records)
    write_csv_index(EXPORT_ROOT / "metrics_index.csv", metric_records)
    write_readme(prompt_records, metric_records)

    print(f"prompts={len(prompt_records)} -> {PROMPTS_DIR}")
    print(f"metrics={len(metric_records)} -> {METRICS_DIR}")
    print(f"index={EXPORT_ROOT / 'README.md'}")


if __name__ == "__main__":
    main()
