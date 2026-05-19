import textwrap
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "manual_dataset_multilabel_metrics_dashboard.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


cells = [
    md(
        """
        # Multilabel metrics dashboard for the 1000-news manual dataset

        This notebook evaluates the classifier against the new wide multilabel dataset.

        The old single-label metric is not the right headline for this task: one news item can have several relevant
        factors, and 13 of 36 factors may have zero support in a given 1000-row evaluation set.

        Recommended dashboard:
        - `factor_micro_f1`: main operational metric over all factor decisions.
        - `sample_f1_empty_correct`: per-news factor-set F1, with empty/empty treated as correct.
        - `any_relevant_f1`: relevance gate quality, meaning whether a news item has at least one relevant factor.
        - `supported_macro_f1`: class-balance diagnostic over factors present in the dataset.
        - `exact_match`: strict diagnostic, useful but intentionally not the headline.
        """
    ),
    code(
        r"""
        from __future__ import annotations

        import json
        import math
        import re
        import sys
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from IPython.display import Markdown, display
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            hamming_loss,
            precision_recall_fscore_support,
        )

        ROOT = Path.cwd()
        if not (ROOT / "manage.py").exists():
            ROOT = Path(r"C:\Users\whynot\VSCodeProjects\news-zero-shot")
        sys.path.insert(0, str(ROOT))

        DATASET_CANDIDATES = [
            ROOT / "interfax_news_multilabel_factor_dataset_1000_wide.csv",
            ROOT / "analysis_outputs" / "interfax_news_multilabel_factor_dataset_1000_wide.csv",
            Path(r"C:\Users\whynot\Downloads\interfax_news_multilabel_factor_dataset_1000_wide.csv"),
        ]
        DATASET_PATH = next((path for path in DATASET_CANDIDATES if path.exists()), None)
        if DATASET_PATH is None:
            raise FileNotFoundError("Cannot find interfax_news_multilabel_factor_dataset_1000_wide.csv")

        OUTPUT_DIR = ROOT / "analysis_outputs"
        OUTPUT_DIR.mkdir(exist_ok=True)

        LLM_PREDICTIONS_PATH = OUTPUT_DIR / "manual_dataset_prompt_search_best_balanced_recall_v2.csv"
        if not LLM_PREDICTIONS_PATH.exists():
            raise FileNotFoundError(f"Missing LLM predictions: {LLM_PREDICTIONS_PATH}")

        sns.set_theme(style="whitegrid", context="notebook")
        pd.set_option("display.max_columns", 140)
        pd.set_option("display.max_colwidth", 180)

        print(f"root={ROOT}")
        print(f"dataset={DATASET_PATH}")
        print(f"llm_predictions={LLM_PREDICTIONS_PATH}")
        """
    ),
    md("## 1. Load and validate the wide multilabel dataset"),
    code(
        r"""
        from analyzer.factors import FACTOR_CONFIG

        factor_keys = [item["key"] for item in FACTOR_CONFIG]
        factor_names = {item["key"]: item["name"] for item in FACTOR_CONFIG}

        def cyrillic_count(text: str) -> int:
            return sum(1 for char in text if 0x0400 <= ord(char) <= 0x04FF)

        raw_bytes = DATASET_PATH.read_bytes()
        dataset_encoding = {
            "bytes": len(raw_bytes),
            "bom_utf8": raw_bytes.startswith(b"\xef\xbb\xbf"),
            "utf8_ok": True,
            "replacement_chars": 0,
        }
        try:
            decoded = raw_bytes.decode("utf-8-sig")
            dataset_encoding["replacement_chars"] = decoded.count("\ufffd")
            dataset_encoding["cyrillic_chars"] = cyrillic_count(decoded)
        except UnicodeDecodeError as exc:
            dataset_encoding["utf8_ok"] = False
            dataset_encoding["utf8_error"] = str(exc)
            raise

        df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
        row_id_col = "Номер строки в датасете 1000"
        if row_id_col not in df.columns:
            row_id_col = df.columns[0]
        df["dataset_row_id"] = pd.to_numeric(df[row_id_col], errors="raise").astype(int)

        expected_factor_cols = [f"factor__{key}" for key in factor_keys]
        missing_factor_cols = [col for col in expected_factor_cols if col not in df.columns]
        extra_factor_cols = sorted(
            col.replace("factor__", "")
            for col in df.columns
            if col.startswith("factor__") and col.replace("factor__", "") not in factor_keys
        )
        if missing_factor_cols:
            raise ValueError(f"Missing factor columns: {missing_factor_cols}")
        if extra_factor_cols:
            raise ValueError(f"Unexpected factor columns: {extra_factor_cols}")

        Y = np.column_stack(
            [
                pd.to_numeric(df[f"factor__{key}"], errors="coerce")
                .fillna(0)
                .astype(int)
                .clip(0, 1)
                .to_numpy()
                for key in factor_keys
            ]
        )
        ids = df["dataset_row_id"].to_numpy()
        true_label_count = Y.sum(axis=1)
        factor_support = pd.Series(Y.sum(axis=0), index=factor_keys).sort_values(ascending=False)
        supported_factor_keys = [key for key in factor_keys if factor_support[key] > 0]
        head_factor_keys = [key for key in factor_keys if factor_support[key] >= 25]

        relevant_count_col = "Количество релевантных факторов"
        relevant_count_mismatch = None
        if relevant_count_col in df.columns:
            declared = pd.to_numeric(df[relevant_count_col], errors="coerce").fillna(0).astype(int).to_numpy()
            relevant_count_mismatch = int((declared != true_label_count).sum())

        dataset_stats = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "factor_count": int(len(factor_keys)),
            "positive_factor_pairs": int(Y.sum()),
            "not_relevant_news": int((true_label_count == 0).sum()),
            "relevant_news": int((true_label_count > 0).sum()),
            "multi_label_news": int((true_label_count > 1).sum()),
            "supported_factor_count": int((factor_support > 0).sum()),
            "zero_support_factor_count": int((factor_support == 0).sum()),
            "mean_labels_per_news": float(true_label_count.mean()),
            "mean_labels_per_relevant_news": float(true_label_count[true_label_count > 0].mean()),
            "relevant_count_mismatch": relevant_count_mismatch,
        }

        print(json.dumps(dataset_encoding, ensure_ascii=False, indent=2))
        print(json.dumps(dataset_stats, ensure_ascii=False, indent=2))
        display(pd.Series(true_label_count).value_counts().sort_index().rename("news_count").to_frame())
        display(factor_support.rename("support").to_frame())
        """
    ),
    md("## 2. Label distribution EDA"),
    code(
        r"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        label_count_dist = pd.Series(true_label_count).value_counts().sort_index()
        sns.barplot(x=label_count_dist.index.astype(str), y=label_count_dist.values, ax=axes[0], color="#2f6f73")
        axes[0].set_title("Labels per news item")
        axes[0].set_xlabel("number of relevant factors")
        axes[0].set_ylabel("news count")

        support_plot = factor_support.reset_index()
        support_plot.columns = ["factor", "support"]
        sns.barplot(data=support_plot, y="factor", x="support", ax=axes[1], color="#486a9a")
        axes[1].set_title("Factor support in manual labels")
        axes[1].set_xlabel("positive news count")
        axes[1].set_ylabel("")

        plt.tight_layout()
        plt.show()
        """
    ),
    md("## 3. Load LLM predictions and align matrices"),
    code(
        r"""
        pred_long = pd.read_csv(LLM_PREDICTIONS_PATH, encoding="utf-8-sig")
        if pred_long["is_relevant"].dtype == object:
            pred_long["is_relevant_bool"] = pred_long["is_relevant"].astype(str).str.lower().isin(["true", "1", "yes"])
        else:
            pred_long["is_relevant_bool"] = pred_long["is_relevant"].astype(bool)
        pred_long["relevance"] = pd.to_numeric(pred_long["relevance"], errors="coerce").fillna(0.0)

        pred_factor_keys = sorted(pred_long["factor_key"].dropna().astype(str).unique())
        prediction_integrity = {
            "rows": int(len(pred_long)),
            "expected_rows": int(len(df) * len(factor_keys)),
            "news_id_count": int(pred_long["dataset_row_id"].nunique()),
            "factor_key_count": int(len(pred_factor_keys)),
            "missing_factor_keys": [key for key in factor_keys if key not in pred_factor_keys],
            "extra_factor_keys": sorted(set(pred_factor_keys) - set(factor_keys)),
            "duplicate_news_factor_rows": int(pred_long.duplicated(["dataset_row_id", "factor_key"]).sum()),
        }
        print(json.dumps(prediction_integrity, ensure_ascii=False, indent=2))

        score_matrix = (
            pred_long.pivot(index="dataset_row_id", columns="factor_key", values="relevance")
            .reindex(index=ids, columns=factor_keys)
            .fillna(0.0)
            .to_numpy(float)
        )
        default_pred_matrix = (
            pred_long.pivot(index="dataset_row_id", columns="factor_key", values="is_relevant_bool")
            .reindex(index=ids, columns=factor_keys)
            .fillna(False)
            .astype(bool)
            .to_numpy(int)
        )

        print(f"score_matrix={score_matrix.shape}, default_pred_matrix={default_pred_matrix.shape}")
        """
    ),
    md("## 4. Multilabel metric definitions"),
    code(
        r"""
        def per_sample_f1_empty_correct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            intersection = (y_true & y_pred).sum(axis=1)
            true_count = y_true.sum(axis=1)
            pred_count = y_pred.sum(axis=1)
            denom = true_count + pred_count
            values = np.where(denom == 0, 1.0, 2 * intersection / np.maximum(denom, 1))
            return float(values.mean())

        def per_sample_jaccard_empty_correct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            intersection = (y_true & y_pred).sum(axis=1)
            union = (y_true | y_pred).sum(axis=1)
            values = np.where(union == 0, 1.0, intersection / np.maximum(union, 1))
            return float(values.mean())

        def evaluate_matrix(y_pred: np.ndarray, name: str) -> dict:
            y_pred = y_pred.astype(int)
            supported_idx = [factor_keys.index(key) for key in supported_factor_keys]
            head_idx = [factor_keys.index(key) for key in head_factor_keys]

            y_any = (Y.sum(axis=1) > 0).astype(int)
            p_any = (y_pred.sum(axis=1) > 0).astype(int)
            any_precision, any_recall, any_f1, _ = precision_recall_fscore_support(
                y_any,
                p_any,
                average="binary",
                zero_division=0,
            )
            empty_specificity = (
                ((y_any == 0) & (p_any == 0)).sum() / max(1, (y_any == 0).sum())
            )

            micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
                Y,
                y_pred,
                average="micro",
                zero_division=0,
            )

            return {
                "name": name,
                "pred_positive_pairs": int(y_pred.sum()),
                "pred_relevant_news": int(p_any.sum()),
                "mean_pred_labels_per_news": float(y_pred.sum(axis=1).mean()),
                "exact_match": float((Y == y_pred).all(axis=1).mean()),
                "sample_f1_empty_correct": per_sample_f1_empty_correct(Y, y_pred),
                "sample_jaccard_empty_correct": per_sample_jaccard_empty_correct(Y, y_pred),
                "factor_micro_f1": float(micro_f1),
                "factor_micro_precision": float(micro_precision),
                "factor_micro_recall": float(micro_recall),
                "factor_macro_f1_all36": float(f1_score(Y, y_pred, average="macro", zero_division=0)),
                "factor_macro_f1_supported": float(
                    f1_score(Y[:, supported_idx], y_pred[:, supported_idx], average="macro", zero_division=0)
                ),
                "factor_macro_f1_head_ge25": float(
                    f1_score(Y[:, head_idx], y_pred[:, head_idx], average="macro", zero_division=0)
                    if head_idx
                    else 0.0
                ),
                "factor_weighted_f1": float(f1_score(Y, y_pred, average="weighted", zero_division=0)),
                "hamming_loss": float(hamming_loss(Y, y_pred)),
                "any_relevant_precision": float(any_precision),
                "any_relevant_recall": float(any_recall),
                "any_relevant_f1": float(any_f1),
                "empty_specificity": float(empty_specificity),
            }

        metric_columns = [
            "name",
            "sample_f1_empty_correct",
            "sample_jaccard_empty_correct",
            "factor_micro_f1",
            "factor_micro_precision",
            "factor_micro_recall",
            "factor_macro_f1_supported",
            "factor_macro_f1_head_ge25",
            "factor_weighted_f1",
            "exact_match",
            "any_relevant_f1",
            "any_relevant_precision",
            "any_relevant_recall",
            "empty_specificity",
            "mean_pred_labels_per_news",
            "pred_positive_pairs",
        ]
        """
    ),
    md("## 5. Threshold sweep and recommended dashboard"),
    code(
        r"""
        rows = [evaluate_matrix(default_pred_matrix, "llm_default_is_relevant")]

        for threshold in np.round(np.arange(0.05, 0.96, 0.05), 2):
            rows.append(evaluate_matrix((score_matrix >= threshold).astype(int), f"threshold_{threshold:.2f}"))

        for k in [1, 2, 3, 5, 10]:
            forced = np.zeros_like(Y)
            order = np.argsort(-score_matrix, axis=1)[:, :k]
            for row_idx, factor_idx in enumerate(order):
                forced[row_idx, factor_idx] = 1
            rows.append(evaluate_matrix(forced, f"top{k}_forced"))

            positive_only = np.zeros_like(Y)
            for row_idx, factor_idx in enumerate(order):
                for idx in factor_idx:
                    if score_matrix[row_idx, idx] > 0:
                        positive_only[row_idx, idx] = 1
            rows.append(evaluate_matrix(positive_only, f"top{k}_positive_score"))

        metrics = pd.DataFrame(rows)
        threshold_metrics = metrics[metrics["name"].str.startswith("threshold_")].copy()
        default_metrics = metrics[metrics["name"] == "llm_default_is_relevant"].copy()

        # Primary recommendation: maximize factor_micro_f1; if tied, prefer the default threshold-like behavior.
        max_micro = threshold_metrics["factor_micro_f1"].max()
        best_threshold_candidates = threshold_metrics[np.isclose(threshold_metrics["factor_micro_f1"], max_micro)].copy()
        if "threshold_0.30" in set(best_threshold_candidates["name"]):
            best_row = best_threshold_candidates[best_threshold_candidates["name"] == "threshold_0.30"].iloc[0]
        else:
            best_row = best_threshold_candidates.sort_values("sample_f1_empty_correct", ascending=False).iloc[0]

        display(metrics[metric_columns].sort_values(["factor_micro_f1", "sample_f1_empty_correct"], ascending=False).round(4).head(20))

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        threshold_plot = threshold_metrics.copy()
        threshold_plot["threshold"] = threshold_plot["name"].str.replace("threshold_", "", regex=False).astype(float)
        sns.lineplot(data=threshold_plot, x="threshold", y="factor_micro_f1", marker="o", ax=axes[0], label="factor_micro_f1")
        sns.lineplot(data=threshold_plot, x="threshold", y="sample_f1_empty_correct", marker="o", ax=axes[0], label="sample_f1")
        axes[0].set_title("Threshold sweep")
        axes[0].set_ylabel("score")

        dashboard_names = ["llm_default_is_relevant", best_row["name"]]
        dashboard_plot = metrics[metrics["name"].isin(dashboard_names)].melt(
            id_vars=["name"],
            value_vars=["factor_micro_f1", "sample_f1_empty_correct", "any_relevant_f1", "exact_match"],
            var_name="metric",
            value_name="value",
        )
        sns.barplot(data=dashboard_plot, x="metric", y="value", hue="name", ax=axes[1])
        axes[1].set_title("Dashboard metrics")
        axes[1].set_xlabel("")
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].set_ylim(0, 1)

        plt.tight_layout()
        plt.show()

        print(f"recommended_threshold_variant={best_row['name']}")
        display(best_row[metric_columns].to_frame("value"))
        """
    ),
    md("## 6. Top-k coverage diagnostics"),
    code(
        r"""
        def topk_stats(k: int) -> dict:
            order = np.argsort(-score_matrix, axis=1)[:, :k]
            label_recalls = []
            hit_any = []
            precisions = []
            for row_idx, factor_idx in enumerate(order):
                true_set = set(np.where(Y[row_idx] == 1)[0])
                pred_set = set(factor_idx)
                if true_set:
                    label_recalls.append(len(true_set & pred_set) / len(true_set))
                    hit_any.append(1.0 if true_set & pred_set else 0.0)
                precisions.append(len(true_set & pred_set) / k)
            return {
                "k": k,
                "label_recall_at_k_on_relevant_news": float(np.mean(label_recalls)),
                "news_hit_any_at_k_on_relevant_news": float(np.mean(hit_any)),
                "precision_at_k_all_news": float(np.mean(precisions)),
            }

        topk_results = pd.DataFrame([topk_stats(k) for k in [1, 2, 3, 5, 10, 15]])
        display(topk_results.round(4))

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=topk_results, x="k", y="label_recall_at_k_on_relevant_news", marker="o", ax=ax, label="label recall@k")
        sns.lineplot(data=topk_results, x="k", y="news_hit_any_at_k_on_relevant_news", marker="o", ax=ax, label="hit any@k")
        ax.set_title("Does the model rank true factors near the top?")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.show()
        """
    ),
    md("## 7. Factor-level report for the recommended threshold"),
    code(
        r"""
        recommended_pred = (score_matrix >= float(str(best_row["name"]).replace("threshold_", ""))).astype(int)

        precision, recall, f1, support = precision_recall_fscore_support(
            Y,
            recommended_pred,
            average=None,
            zero_division=0,
        )
        factor_report = pd.DataFrame(
            {
                "factor_key": factor_keys,
                "factor_name": [factor_names[key] for key in factor_keys],
                "support": support.astype(int),
                "predicted_positive": recommended_pred.sum(axis=0).astype(int),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        ).sort_values(["support", "f1"], ascending=[False, False])

        display(factor_report.round(4))

        fig, ax = plt.subplots(figsize=(12, 8))
        plot_df = factor_report[factor_report["support"] > 0].copy()
        sns.barplot(data=plot_df, y="factor_key", x="f1", ax=ax, color="#486a9a")
        ax.set_title("Factor F1 for supported labels")
        ax.set_xlabel("F1")
        ax.set_ylabel("")
        ax.set_xlim(0, 1)
        plt.tight_layout()
        plt.show()
        """
    ),
    md("## 8. News-level error table"),
    code(
        r"""
        def factors_from_vector(vector: np.ndarray) -> list[str]:
            return [factor_keys[idx] for idx in np.where(vector == 1)[0]]

        error_rows = []
        for row_idx, dataset_row_id in enumerate(ids):
            true_set = set(factors_from_vector(Y[row_idx]))
            pred_set = set(factors_from_vector(recommended_pred[row_idx]))
            tp = sorted(true_set & pred_set)
            fp = sorted(pred_set - true_set)
            fn = sorted(true_set - pred_set)
            union = true_set | pred_set
            jaccard = 1.0 if not union else len(true_set & pred_set) / len(union)
            sample_f1 = 1.0 if not (true_set or pred_set) else 2 * len(tp) / max(1, len(true_set) + len(pred_set))
            error_rows.append(
                {
                    "dataset_row_id": int(dataset_row_id),
                    "title": df.iloc[row_idx]["Заголовок новости"] if "Заголовок новости" in df.columns else "",
                    "true_factors": ";".join(sorted(true_set)),
                    "pred_factors": ";".join(sorted(pred_set)),
                    "tp_factors": ";".join(tp),
                    "fp_factors": ";".join(fp),
                    "fn_factors": ";".join(fn),
                    "true_count": len(true_set),
                    "pred_count": len(pred_set),
                    "sample_f1": sample_f1,
                    "sample_jaccard": jaccard,
                    "needs_review": bool(fp or fn),
                }
            )

        news_errors = pd.DataFrame(error_rows).sort_values(["sample_f1", "true_count"], ascending=[True, False])
        display(news_errors.head(30))
        """
    ),
    md("## 9. Save outputs and final reading"),
    code(
        r"""
        output_paths = {
            "summary": OUTPUT_DIR / "manual_dataset_multilabel_metrics_summary.json",
            "dashboard_metrics": OUTPUT_DIR / "manual_dataset_multilabel_dashboard_metrics.csv",
            "threshold_sweep": OUTPUT_DIR / "manual_dataset_multilabel_threshold_sweep.csv",
            "topk_results": OUTPUT_DIR / "manual_dataset_multilabel_topk_results.csv",
            "factor_report": OUTPUT_DIR / "manual_dataset_multilabel_factor_report.csv",
            "news_errors": OUTPUT_DIR / "manual_dataset_multilabel_news_errors.csv",
        }

        metrics.to_csv(output_paths["dashboard_metrics"], index=False, encoding="utf-8-sig")
        threshold_metrics.to_csv(output_paths["threshold_sweep"], index=False, encoding="utf-8-sig")
        topk_results.to_csv(output_paths["topk_results"], index=False, encoding="utf-8-sig")
        factor_report.to_csv(output_paths["factor_report"], index=False, encoding="utf-8-sig")
        news_errors.to_csv(output_paths["news_errors"], index=False, encoding="utf-8-sig")

        default_row = default_metrics.iloc[0].to_dict()
        recommended_row = best_row.to_dict()
        summary = {
            "dataset_path": str(DATASET_PATH),
            "llm_predictions_path": str(LLM_PREDICTIONS_PATH),
            "dataset_encoding": dataset_encoding,
            "dataset_stats": dataset_stats,
            "prediction_integrity": prediction_integrity,
            "recommended_metric_dashboard": {
                "primary_metric": "factor_micro_f1",
                "secondary_metrics": [
                    "sample_f1_empty_correct",
                    "any_relevant_f1",
                    "factor_macro_f1_supported",
                    "exact_match",
                ],
                "why": "Multilabel task with sparse labels and zero-support factors; strict all-class macro-F1 is diagnostic, not headline.",
            },
            "default_is_relevant_metrics": {
                key: float(default_row[key])
                for key in default_row
                if key != "name" and isinstance(default_row[key], (int, float, np.integer, np.floating))
            },
            "recommended_threshold_variant": recommended_row["name"],
            "recommended_threshold_metrics": {
                key: float(recommended_row[key])
                for key in recommended_row
                if key != "name" and isinstance(recommended_row[key], (int, float, np.integer, np.floating))
            },
            "topk_results": topk_results.to_dict(orient="records"),
            "output_paths": {key: str(value) for key, value in output_paths.items()},
        }
        output_paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        display(
            Markdown(
                f'''
                ## Итоговое табло

                Основная задача здесь multilabel: одна новость может иметь 0, 1, 2 или 3 фактора.

                В датасете: **{dataset_stats['positive_factor_pairs']}** positive factor pairs,
                **{dataset_stats['multi_label_news']}** multi-label news,
                **{dataset_stats['supported_factor_count']}** supported factors out of **{dataset_stats['factor_count']}**.

                Рекомендованный threshold variant: `{recommended_row['name']}`.

                `factor_micro_f1`: **{recommended_row['factor_micro_f1']:.3f}**.

                `sample_f1_empty_correct`: **{recommended_row['sample_f1_empty_correct']:.3f}**.

                `any_relevant_f1`: **{recommended_row['any_relevant_f1']:.3f}**.

                `factor_macro_f1_supported`: **{recommended_row['factor_macro_f1_supported']:.3f}**.

                `exact_match`: **{recommended_row['exact_match']:.3f}**.

                Reading: the relevance gate is much stronger than factor-set extraction.
                The model often detects that the news is relevant, but still misses or swaps specific factors.
                '''
            )
        )

        print("Saved artifacts:")
        for key, value in output_paths.items():
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
