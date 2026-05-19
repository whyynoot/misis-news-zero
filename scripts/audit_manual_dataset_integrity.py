from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyzer.constants import normalize_factor_key  # noqa: E402
from analyzer.factors import FACTOR_CONFIG  # noqa: E402


FACTOR_KEYS = [item["key"] for item in FACTOR_CONFIG]
CLASS_LABELS = ["not_relevant"] + FACTOR_KEYS

DATASET_PATH = ROOT / "interfax_news_manual_factor_dataset_1000.csv"
MULTILABEL_DATASET_PATH = Path(r"C:\Users\whynot\Downloads\interfax_news_multilabel_factor_dataset_1000_wide.csv")
OUTPUT_DIR = ROOT / "analysis_outputs"
OUTPUT_PATH = OUTPUT_DIR / "manual_dataset_integrity_audit_summary.json"

AUDIT_PATHS = [
    DATASET_PATH,
    MULTILABEL_DATASET_PATH,
    ROOT / "analyzer" / "factors.py",
    ROOT / "analyzer" / "llm_prompts.py",
    ROOT / "scripts" / "build_prompt_search_notebook.py",
    ROOT / "scripts" / "build_label_audit_notebook.py",
    ROOT / "scripts" / "build_supervised_gate_notebook.py",
    ROOT / "notebooks" / "manual_dataset_llm_prompt_search.ipynb",
    ROOT / "notebooks" / "manual_dataset_label_audit_and_relabel_queue.ipynb",
    ROOT / "notebooks" / "manual_dataset_supervised_gate_probe.ipynb",
    ROOT / "notebooks" / "manual_dataset_multilabel_metrics_dashboard.ipynb",
    OUTPUT_DIR / "manual_dataset_relabel_queue.csv",
    OUTPUT_DIR / "manual_dataset_relabel_queue_priority_only.csv",
    OUTPUT_DIR / "manual_dataset_prompt_search_best_balanced_recall_v2.csv",
]


def cyrillic_count(text: str) -> int:
    return sum(1 for char in text if 0x0400 <= ord(char) <= 0x04FF)


def repair_cp1251_utf8(text: str) -> str | None:
    try:
        return text.encode("cp1251").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None


def mojibake_line_audit(text: str) -> tuple[int, list[dict[str, str]]]:
    count = 0
    examples: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        repaired = repair_cp1251_utf8(line)
        if repaired and cyrillic_count(repaired) > cyrillic_count(line) and any(
            token in line for token in ("Р", "С")
        ):
            count += 1
            if len(examples) < 3:
                examples.append({"bad": line[:160], "repaired": repaired[:160]})
    return count, examples


def decode_info(path: Path) -> dict:
    data = path.read_bytes()
    info: dict = {
        "path": str(path),
        "bytes": len(data),
        "bom_utf8": data.startswith(b"\xef\xbb\xbf"),
    }
    try:
        text = data.decode("utf-8-sig")
        info["utf8_ok"] = True
    except UnicodeDecodeError as exc:
        info["utf8_ok"] = False
        info["utf8_error"] = str(exc)
        return info

    bad_lines, examples = mojibake_line_audit(text)
    info.update(
        {
            "replacement_chars": text.count("\ufffd"),
            "cyrillic_chars": cyrillic_count(text),
            "mojibake_repairable_lines": bad_lines,
            "mojibake_examples": examples,
            "sample": text[:120].replace("\n", "\\n"),
        }
    )
    return info


def read_csv_utf8(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def parse_additional_factors(value) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    out: list[str] = []
    for part in re.split(r"[;,\n|]+", str(value)):
        key = normalize_factor_key(part.strip())
        if key and key in FACTOR_KEYS and key not in out:
            out.append(key)
    return out


def yes_no(value) -> bool:
    return str(value).strip().lower() in {"\u0434\u0430", "yes", "true", "1", "y"}


def audit_dataset() -> dict:
    df = read_csv_utf8(DATASET_PATH)
    cols = df.columns
    col_applicable = cols[35]
    col_primary = cols[36]
    col_additional = cols[40]

    applicable = df[col_applicable].map(yes_no)
    primary_raw = df[col_primary].fillna("").astype(str).str.strip()
    primary_norm = primary_raw.map(lambda value: normalize_factor_key(value) if value else None).fillna(
        "not_relevant"
    )
    primary_class = pd.Series(primary_norm.where(applicable, "not_relevant"))
    additional = df[col_additional].map(parse_additional_factors)

    additional_invalid_raw = []
    for value in df[col_additional]:
        if pd.isna(value) or not str(value).strip():
            continue
        for part in re.split(r"[;,\n|]+", str(value)):
            raw = part.strip()
            if not raw:
                continue
            normalized = normalize_factor_key(raw)
            if normalized not in FACTOR_KEYS:
                additional_invalid_raw.append(raw)

    manual_sets = []
    for is_applicable, primary, additional_items in zip(applicable, primary_norm, additional):
        items = []
        if is_applicable and primary in FACTOR_KEYS:
            items.append(primary)
        for key in additional_items:
            if key not in items:
                items.append(key)
        manual_sets.append(items)

    additional_all = [key for items in additional for key in items]
    used_factor_keys = sorted(set([key for key in primary_norm if key in FACTOR_KEYS]) | set(additional_all))

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names_cyrillic_chars": cyrillic_count("\n".join(map(str, df.columns))),
        "column_names_mojibake_repairable_lines": mojibake_line_audit("\n".join(map(str, df.columns)))[0],
        "applicable_yes": int(applicable.sum()),
        "not_relevant": int((~applicable).sum()),
        "primary_supported_class_count_with_not_relevant": int(primary_class.nunique()),
        "used_factor_key_count_primary_or_additional": len(used_factor_keys),
        "primary_invalid_normalized": sorted(set(primary_norm) - set(CLASS_LABELS)),
        "additional_invalid_raw": sorted(set(additional_invalid_raw)),
        "manual_factor_set_size_distribution": {
            str(key): int(value)
            for key, value in pd.Series([len(items) for items in manual_sets]).value_counts().sort_index().items()
        },
        "multi_label_rows": int(sum(len(items) > 1 for items in manual_sets)),
        "missing_factor_keys_from_dataset_labels": [key for key in FACTOR_KEYS if key not in used_factor_keys],
        "used_factor_keys": used_factor_keys,
        "primary_counts": {str(key): int(value) for key, value in primary_class.value_counts().items()},
    }


def audit_predictions() -> dict:
    pred_path = OUTPUT_DIR / "manual_dataset_prompt_search_best_balanced_recall_v2.csv"
    if not pred_path.exists():
        return {"path": str(pred_path), "exists": False}

    dataset_rows = len(read_csv_utf8(DATASET_PATH))
    pred = read_csv_utf8(pred_path)
    pred_factor_keys = sorted(set(pred["factor_key"].dropna().astype(str)))

    return {
        "path": str(pred_path),
        "exists": True,
        "rows": len(pred),
        "expected_rows_1000_x_factor_count": dataset_rows * len(FACTOR_KEYS),
        "factor_key_count": len(pred_factor_keys),
        "missing_factor_keys": [key for key in FACTOR_KEYS if key not in pred_factor_keys],
        "extra_factor_keys": sorted(set(pred_factor_keys) - set(FACTOR_KEYS)),
        "duplicate_news_factor_rows": int(pred.duplicated(["dataset_row_id", "factor_key"]).sum()),
        "news_id_count": int(pred["dataset_row_id"].nunique()),
    }


def audit_notebooks() -> list[dict]:
    out = []
    for path in [
        ROOT / "notebooks" / "manual_dataset_llm_prompt_search.ipynb",
        ROOT / "notebooks" / "manual_dataset_label_audit_and_relabel_queue.ipynb",
        ROOT / "notebooks" / "manual_dataset_supervised_gate_probe.ipynb",
        ROOT / "notebooks" / "manual_dataset_multilabel_metrics_dashboard.ipynb",
    ]:
        if not path.exists():
            continue
        notebook = nbformat.read(path, as_version=4)
        text = "\n".join("".join(cell.get("source", "")) for cell in notebook.cells)
        bad_lines, examples = mojibake_line_audit(text)
        out.append(
            {
                "path": str(path),
                "cells": len(notebook.cells),
                "cyrillic_chars": cyrillic_count(text),
                "mojibake_repairable_lines": bad_lines,
                "mojibake_examples": examples,
                "contains_replacement_char": "\ufffd" in text,
            }
        )
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary = {
        "encoding_files": [decode_info(path) for path in AUDIT_PATHS if path.exists()],
        "factor_catalog": {
            "factor_count": len(FACTOR_KEYS),
            "single_label_class_count_with_not_relevant": len(CLASS_LABELS),
            "duplicate_factor_keys": [key for key, count in Counter(FACTOR_KEYS).items() if count > 1],
            "display_order_count": len([item.get("display_order") for item in FACTOR_CONFIG]),
            "display_order_min": min(item.get("display_order") for item in FACTOR_CONFIG),
            "display_order_max": max(item.get("display_order") for item in FACTOR_CONFIG),
            "factor_keys": FACTOR_KEYS,
        },
        "dataset": audit_dataset(),
        "predictions": audit_predictions(),
        "notebooks": audit_notebooks(),
    }
    OUTPUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
