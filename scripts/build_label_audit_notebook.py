import textwrap
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "manual_dataset_label_audit_and_relabel_queue.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


cells = [
    md(
        """
        # Manual dataset label audit and relabel queue

        This notebook checks whether the low classifier scores are partly caused by the labeling scheme itself.

        Focus:
        - how much of the dataset is actually multi-label;
        - how much single-label scoring penalizes predictions that hit an additional factor;
        - which rows should be manually reviewed first;
        - a ready CSV queue for relabeling without overwriting the source dataset.
        """
    ),
    code(
        r"""
        from __future__ import annotations

        import json
        import re
        import sys
        from pathlib import Path

        import numpy as np
        import pandas as pd
        from IPython.display import Markdown, display
        from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
        from sklearn.preprocessing import MultiLabelBinarizer

        ROOT = Path.cwd()
        if not (ROOT / "manage.py").exists():
            ROOT = Path(r"C:\Users\whynot\VSCodeProjects\news-zero-shot")
        sys.path.insert(0, str(ROOT))

        DATASET_PATH = ROOT / "interfax_news_manual_factor_dataset_1000.csv"
        OUTPUT_DIR = ROOT / "analysis_outputs"
        OUTPUT_DIR.mkdir(exist_ok=True)

        LLM_BEST_PATH = OUTPUT_DIR / "manual_dataset_prompt_search_best_balanced_recall_v2.csv"
        SUPERVISED_PRED_PATH = OUTPUT_DIR / "manual_dataset_supervised_gate_predictions.csv"

        print(f"root={ROOT}")
        print(f"dataset={DATASET_PATH}")
        print(f"llm_best_exists={LLM_BEST_PATH.exists()}")
        print(f"supervised_predictions_exists={SUPERVISED_PRED_PATH.exists()}")
        """
    ),
    md("## 1. Parse current manual labels"),
    code(
        r"""
        from analyzer.constants import normalize_factor_key
        from analyzer.factors import FACTOR_CONFIG

        factor_keys = [item["key"] for item in FACTOR_CONFIG]
        factor_name_by_key = {item["key"]: item["name"] for item in FACTOR_CONFIG}
        class_labels = ["not_relevant"] + factor_keys

        raw_df = pd.read_csv(DATASET_PATH)
        cols = raw_df.columns

        COL_ROW_ID = cols[0]
        COL_PUBLISHED_AT = cols[7]
        COL_TITLE = cols[9]
        COL_DESCRIPTION = cols[10]
        COL_FIRST_PARAGRAPH = cols[11]
        COL_FULL_TEXT = cols[12]
        COL_URL = cols[15]
        COL_APPLICABLE = cols[35]
        COL_PRIMARY_FACTOR = cols[36]
        COL_ADDITIONAL_FACTORS = cols[40]
        COL_SIGNAL = cols[41]
        COL_RELEVANCE = cols[43]
        COL_PRESSURE = cols[44]
        COL_CONFIDENCE = cols[45]

        def clean_text(value) -> str:
            if pd.isna(value):
                return ""
            return re.sub(r"\s+", " ", str(value)).strip()

        def yes_no(value) -> bool:
            return str(value).strip().lower() in {"\u0434\u0430", "yes", "true", "1", "y"}

        def parse_additional(value) -> list[str]:
            if pd.isna(value) or not str(value).strip():
                return []
            out = []
            for part in re.split(r"[;,\n|]+", str(value)):
                key = normalize_factor_key(part.strip())
                if key and key in factor_keys and key not in out:
                    out.append(key)
            return out

        df = raw_df.copy()
        df["dataset_row_id"] = pd.to_numeric(df[COL_ROW_ID], errors="raise").astype(int)
        df["manual_applicable_bool"] = df[COL_APPLICABLE].map(yes_no)
        df["manual_primary_factor"] = (
            df[COL_PRIMARY_FACTOR]
            .map(lambda value: normalize_factor_key(str(value).strip()) if pd.notna(value) else None)
            .fillna("not_relevant")
        )
        df["manual_additional_factors"] = df[COL_ADDITIONAL_FACTORS].map(parse_additional)
        df["manual_primary_class"] = np.where(
            df["manual_applicable_bool"],
            df["manual_primary_factor"],
            "not_relevant",
        )

        def factor_set(row) -> list[str]:
            keys = []
            if row["manual_applicable_bool"] and row["manual_primary_factor"] in factor_keys:
                keys.append(row["manual_primary_factor"])
            for key in row["manual_additional_factors"]:
                if key not in keys:
                    keys.append(key)
            return keys

        df["manual_factor_set"] = df.apply(factor_set, axis=1)
        df["manual_factor_count"] = df["manual_factor_set"].map(len)
        df["text_excerpt"] = (
            df[COL_TITLE].map(clean_text)
            + " | "
            + df[COL_DESCRIPTION].map(clean_text)
            + " | "
            + df[COL_FIRST_PARAGRAPH].map(clean_text)
        ).str.slice(0, 900)

        label_stats = {
            "rows": int(len(df)),
            "applicable": int(df["manual_applicable_bool"].sum()),
            "not_relevant": int((~df["manual_applicable_bool"]).sum()),
            "additional_nonempty": int(df["manual_additional_factors"].map(bool).sum()),
            "multi_label_rows": int((df["manual_factor_count"] > 1).sum()),
            "supported_primary_classes": int(df["manual_primary_class"].nunique()),
        }
        print(json.dumps(label_stats, ensure_ascii=False, indent=2))
        display(df["manual_factor_count"].value_counts().sort_index().rename("rows").to_frame())
        display(df["manual_primary_class"].value_counts().rename("primary_support").to_frame().head(25))
        """
    ),
    md("## 2. Load model predictions"),
    code(
        r"""
        if not LLM_BEST_PATH.exists():
            raise FileNotFoundError(f"Missing LLM predictions: {LLM_BEST_PATH}")

        llm_pred = pd.read_csv(LLM_BEST_PATH)
        llm_pred["is_relevant"] = llm_pred["is_relevant"].astype(bool)
        llm_pred["relevance"] = pd.to_numeric(llm_pred["relevance"], errors="coerce").fillna(0.0)

        llm_top = (
            llm_pred.sort_values(["dataset_row_id", "relevance"], ascending=[True, False])
            .groupby("dataset_row_id")
            .first()[["factor_key", "relevance", "sentiment_label", "pressure", "confidence", "evidence", "reason"]]
            .add_prefix("llm_top_")
        )
        llm_any = llm_pred[llm_pred["is_relevant"]].groupby("dataset_row_id").size().gt(0).rename("llm_pred_any_relevant")

        def top_relevant_factors(group: pd.DataFrame, n: int = 5) -> str:
            rel = group[group["is_relevant"]].sort_values("relevance", ascending=False).head(n)
            return "; ".join(f"{row.factor_key}:{row.relevance:.2f}" for row in rel.itertuples())

        llm_top_relevant = llm_pred.groupby("dataset_row_id").apply(top_relevant_factors).rename("llm_top_relevant_factors")

        eval_df = (
            df.set_index("dataset_row_id")
            .join(llm_top)
            .join(llm_any)
            .join(llm_top_relevant)
            .reset_index()
        )
        eval_df["llm_pred_any_relevant"] = eval_df["llm_pred_any_relevant"].fillna(False).astype(bool)
        eval_df["llm_pred_primary_class"] = np.where(
            eval_df["llm_pred_any_relevant"],
            eval_df["llm_top_factor_key"],
            "not_relevant",
        )

        if SUPERVISED_PRED_PATH.exists():
            sup_pred = pd.read_csv(SUPERVISED_PRED_PATH)
            best_col = "pred_char_word_tfidf_linearsvc"
            if best_col not in sup_pred.columns:
                pred_cols = [col for col in sup_pred.columns if col.startswith("pred_")]
                best_col = pred_cols[0] if pred_cols else None
            if best_col:
                eval_df = eval_df.merge(
                    sup_pred[["dataset_row_id", best_col]].rename(columns={best_col: "supervised_pred_primary_class"}),
                    on="dataset_row_id",
                    how="left",
                )
        if "supervised_pred_primary_class" not in eval_df:
            eval_df["supervised_pred_primary_class"] = ""

        print(f"eval_rows={len(eval_df)}")
        display(eval_df[["dataset_row_id", "manual_primary_class", "manual_factor_set", "llm_pred_primary_class", "llm_top_relevant_factors", "supervised_pred_primary_class"]].head())
        """
    ),
    md("## 3. Single-label vs multi-label acceptance"),
    code(
        r"""
        y_true_primary = eval_df["manual_primary_class"].to_numpy()
        y_pred_primary = eval_df["llm_pred_primary_class"].to_numpy()

        primary_accuracy = accuracy_score(y_true_primary, y_pred_primary)
        primary_supported_macro_f1 = f1_score(
            y_true_primary,
            y_pred_primary,
            labels=[label for label in class_labels if (eval_df["manual_primary_class"] == label).any()],
            average="macro",
            zero_division=0,
        )
        primary_all37_macro_f1 = f1_score(
            y_true_primary,
            y_pred_primary,
            labels=class_labels,
            average="macro",
            zero_division=0,
        )

        def top1_accepted_by_factor_set(row) -> bool:
            if not row["llm_pred_any_relevant"]:
                return not row["manual_applicable_bool"]
            return str(row["llm_top_factor_key"]) in set(row["manual_factor_set"])

        eval_df["llm_top1_accepted_by_manual_set"] = eval_df.apply(top1_accepted_by_factor_set, axis=1)
        set_accuracy = float(eval_df["llm_top1_accepted_by_manual_set"].mean())

        mlb = MultiLabelBinarizer(classes=factor_keys)
        y_true_multi = mlb.fit_transform(eval_df["manual_factor_set"])
        y_pred_multi = (
            llm_pred.pivot(index="dataset_row_id", columns="factor_key", values="is_relevant")
            .reindex(index=eval_df["dataset_row_id"], columns=factor_keys)
            .fillna(False)
            .astype(int)
            .to_numpy()
        )
        factor_micro_precision, factor_micro_recall, factor_micro_f1, _ = precision_recall_fscore_support(
            y_true_multi,
            y_pred_multi,
            average="micro",
            zero_division=0,
        )
        factor_macro_precision, factor_macro_recall, factor_macro_f1, _ = precision_recall_fscore_support(
            y_true_multi,
            y_pred_multi,
            average="macro",
            zero_division=0,
        )

        eval_df["primary_wrong_but_additional_hit"] = (
            eval_df["manual_applicable_bool"]
            & eval_df["llm_pred_any_relevant"]
            & (eval_df["llm_pred_primary_class"] != eval_df["manual_primary_class"])
            & eval_df.apply(lambda row: row["llm_top_factor_key"] in set(row["manual_factor_set"]), axis=1)
        )

        metric_summary = {
            "llm_primary_accuracy": float(primary_accuracy),
            "llm_primary_supported_macro_f1": float(primary_supported_macro_f1),
            "llm_primary_macro_f1_all37": float(primary_all37_macro_f1),
            "llm_top1_factor_set_accuracy": set_accuracy,
            "single_label_penalty_cases": int(eval_df["primary_wrong_but_additional_hit"].sum()),
            "factor_micro_precision": float(factor_micro_precision),
            "factor_micro_recall": float(factor_micro_recall),
            "factor_micro_f1": float(factor_micro_f1),
            "factor_macro_precision": float(factor_macro_precision),
            "factor_macro_recall": float(factor_macro_recall),
            "factor_macro_f1": float(factor_macro_f1),
        }

        display(pd.Series(metric_summary).rename("value").to_frame())
        """
    ),
    md("## 4. Build manual review queue"),
    code(
        r"""
        primary_support = eval_df["manual_primary_class"].value_counts().to_dict()

        def issue_flags(row) -> list[str]:
            flags = []
            manual_class = row["manual_primary_class"]
            llm_class = row["llm_pred_primary_class"]
            sup_class = row.get("supervised_pred_primary_class", "")
            llm_top_relevance = float(row["llm_top_relevance"] or 0.0)

            if row["manual_factor_count"] > 1:
                flags.append("has_additional_factors")
            if row["primary_wrong_but_additional_hit"]:
                flags.append("llm_hit_additional_not_primary")
            if manual_class == "not_relevant" and llm_class != "not_relevant" and sup_class not in {"", "not_relevant"}:
                flags.append("manual_not_relevant_both_models_relevant")
            if manual_class != "not_relevant" and llm_class == "not_relevant" and sup_class == "not_relevant":
                flags.append("manual_relevant_both_models_not_relevant")
            if manual_class != "not_relevant" and llm_class != "not_relevant" and sup_class == llm_class and llm_class != manual_class:
                flags.append("models_agree_against_manual")
            if manual_class == "not_relevant" and llm_class != "not_relevant" and llm_top_relevance >= 0.6:
                flags.append("llm_high_relevance_against_not_relevant")
            if manual_class != "not_relevant" and primary_support.get(manual_class, 0) < 10:
                flags.append("rare_primary_class")
            if manual_class != "not_relevant" and llm_class != "not_relevant" and llm_class != manual_class:
                flags.append("primary_factor_conflict")
            return flags

        def priority_score(flags: list[str]) -> int:
            weights = {
                "models_agree_against_manual": 100,
                "manual_not_relevant_both_models_relevant": 95,
                "manual_relevant_both_models_not_relevant": 90,
                "llm_high_relevance_against_not_relevant": 80,
                "llm_hit_additional_not_primary": 75,
                "primary_factor_conflict": 65,
                "has_additional_factors": 50,
                "rare_primary_class": 35,
            }
            return max([weights.get(flag, 0) for flag in flags], default=0)

        eval_df["review_flags_list"] = eval_df.apply(issue_flags, axis=1)
        eval_df["review_flags"] = eval_df["review_flags_list"].map(lambda flags: ";".join(flags))
        eval_df["review_priority"] = eval_df["review_flags_list"].map(priority_score)
        eval_df["needs_manual_review"] = eval_df["review_priority"] > 0

        relabel_queue = eval_df[
            [
                "dataset_row_id",
                COL_PUBLISHED_AT,
                COL_TITLE,
                COL_DESCRIPTION,
                COL_URL,
                "text_excerpt",
                "manual_applicable_bool",
                "manual_primary_class",
                "manual_additional_factors",
                "manual_factor_set",
                COL_SIGNAL,
                COL_RELEVANCE,
                COL_PRESSURE,
                COL_CONFIDENCE,
                "llm_pred_primary_class",
                "llm_top_relevance",
                "llm_top_sentiment_label",
                "llm_top_pressure",
                "llm_top_confidence",
                "llm_top_evidence",
                "llm_top_reason",
                "llm_top_relevant_factors",
                "supervised_pred_primary_class",
                "review_priority",
                "review_flags",
                "needs_manual_review",
            ]
        ].copy().rename(
            columns={
                COL_PUBLISHED_AT: "published_at",
                COL_TITLE: "title",
                COL_DESCRIPTION: "description",
                COL_URL: "url",
                COL_SIGNAL: "manual_signal_label",
                COL_RELEVANCE: "manual_relevance",
                COL_PRESSURE: "manual_pressure",
                COL_CONFIDENCE: "manual_annotator_confidence",
            }
        )

        relabel_queue["review_decision"] = ""
        relabel_queue["new_applicable"] = ""
        relabel_queue["new_primary_factor"] = ""
        relabel_queue["new_additional_factors"] = ""
        relabel_queue["new_signal_label"] = ""
        relabel_queue["new_relevance"] = ""
        relabel_queue["new_pressure"] = ""
        relabel_queue["review_notes"] = ""

        relabel_queue = relabel_queue.sort_values(
            ["review_priority", "dataset_row_id"],
            ascending=[False, True],
        )

        print(f"needs_manual_review={int(relabel_queue['needs_manual_review'].sum())} of {len(relabel_queue)}")
        display(relabel_queue["review_flags"].value_counts().head(20).rename("rows").to_frame())
        display(relabel_queue.head(30))
        """
    ),
    md("## 5. Save audit artifacts"),
    code(
        r"""
        queue_path = OUTPUT_DIR / "manual_dataset_relabel_queue.csv"
        priority_queue_path = OUTPUT_DIR / "manual_dataset_relabel_queue_priority_only.csv"
        summary_path = OUTPUT_DIR / "manual_dataset_label_audit_summary.json"

        relabel_queue.to_csv(queue_path, index=False, encoding="utf-8-sig")
        relabel_queue[relabel_queue["needs_manual_review"]].to_csv(priority_queue_path, index=False, encoding="utf-8-sig")

        summary = {
            "label_stats": label_stats,
            "metric_summary": metric_summary,
            "needs_manual_review": int(relabel_queue["needs_manual_review"].sum()),
            "priority_queue_rows": int((relabel_queue["review_priority"] > 0).sum()),
            "top_review_flags": relabel_queue["review_flags"].value_counts().head(20).to_dict(),
            "output_paths": {
                "queue_all_rows": str(queue_path),
                "queue_priority_only": str(priority_queue_path),
                "summary": str(summary_path),
            },
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        display(
            Markdown(
                f'''
                ## Audit summary

                Current dataset has **{label_stats['multi_label_rows']}** multi-label rows out of **{label_stats['rows']}**.

                LLM top-1 primary accuracy: **{metric_summary['llm_primary_accuracy']:.3f}**.

                LLM top-1 accepted by primary+additional labels: **{metric_summary['llm_top1_factor_set_accuracy']:.3f}**.

                Single-label scoring penalty: **{metric_summary['single_label_penalty_cases']}** rows.

                Rows recommended for manual review: **{summary['needs_manual_review']}**.

                The source dataset was not overwritten. Use the relabel queue CSV to edit new labels.
                '''
            )
        )

        print("Saved artifacts:")
        for key, value in summary["output_paths"].items():
            print(f"- {key}: {value}")
        """
    ),
]


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(exist_ok=True)
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    nbf.write(notebook, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
