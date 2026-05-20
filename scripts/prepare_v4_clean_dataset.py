from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis_outputs"
DEFAULT_SOURCE = Path(r"C:\Users\whynot\Downloads\FINAL_v4_clean_social_signal_dataset_965.csv")
DEFAULT_TARGET = OUT / "FINAL_v4_clean_social_signal_dataset_965_normalized.csv"


def parse_annotations(value: Any) -> list[dict[str, Any]]:
    if pd.isna(value):
        return []
    try:
        data = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DEFAULT_SOURCE, encoding="utf-8-sig")
    if "dataset_row_id" not in df.columns:
        df.insert(0, "dataset_row_id", pd.to_numeric(df["row_id"], errors="raise").astype(int))
    if "v3_factor_count" not in df.columns and "factor_count" in df.columns:
        df["v3_factor_count"] = pd.to_numeric(df["factor_count"], errors="coerce").fillna(0).astype(int)

    factor_keys = [col.removeprefix("factor__") for col in df.columns if col.startswith("factor__")]
    for key in factor_keys:
        df[f"strength__{key}"] = ""
        df[f"sentiment__{key}"] = ""
        df[f"relevance__{key}"] = 0.0
        df[f"pressure__{key}"] = 0.0

    for idx, value in df["annotations_json"].items():
        for annotation in parse_annotations(value):
            key = str(annotation.get("factor_id") or "").strip()
            if key not in factor_keys:
                continue
            df.at[idx, f"strength__{key}"] = str(annotation.get("label_strength") or "").strip().lower()
            df.at[idx, f"sentiment__{key}"] = str(annotation.get("sentiment") or "").strip().lower()
            df.at[idx, f"relevance__{key}"] = annotation.get("relevance") or 0.0
            df.at[idx, f"pressure__{key}"] = annotation.get("pressure") or 0.0

    df.to_csv(DEFAULT_TARGET, index=False, encoding="utf-8-sig")
    print(f"saved={DEFAULT_TARGET}")
    print(f"rows={len(df)}")
    print(f"factors={len(factor_keys)}")
    print("strength counts:")
    strengths = []
    for key in factor_keys:
        strengths.extend([value for value in df[f"strength__{key}"].tolist() if value])
    print(pd.Series(strengths).value_counts().to_string())


if __name__ == "__main__":
    main()
