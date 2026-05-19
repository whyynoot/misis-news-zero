from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "v3_clean_broad_weak_llm_evaluation.ipynb"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(source).strip())


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb.cells = [
        md(
            """
            # V3 clean broad weak dataset: LLM multilabel evaluation

            This notebook re-evaluates the cached LLM classifiers on the new v3 broad/weak multilabel dataset.

            Main metric: `micro_f1` over the 36 binary factor labels.

            Supporting metrics:
            - `macro_f1_supported`: macro-F1 only over factors with positive gold support.
            - `sample_f1_empty_correct`: average per-news multilabel F1, with empty-empty counted as 1.
            - `any_relevant_f1`: relevance gate quality: does the news have at least one relevant factor.
            - per-factor precision/recall/F1.

            Important scope note: the new CSV contains 1000 rows, but the old cached full LLM predictions cover
            only 739 of those rows. The richer search experiments cover 237 rows. All comparisons below use
            explicit overlap sets so we do not silently count missing predictions as model failures.
            """
        ),
        code(
            """
            from pathlib import Path
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt

            ROOT = Path.cwd()
            OUT = ROOT / "analysis_outputs"
            DATASET_PATH = OUT / "v3_clean_broad_weak_dataset_1000.csv"

            from analyzer.factors import FACTOR_CONFIG

            FACTOR_KEYS = [item["key"] for item in FACTOR_CONFIG]
            FACTOR_NAMES = {item["key"]: item["name"] for item in FACTOR_CONFIG}

            pd.set_option("display.max_columns", 80)
            pd.set_option("display.max_colwidth", 120)
            """
        ),
        md("## Dataset and label sanity checks"),
        code(
            """
            df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
            df["dataset_row_id"] = df["dataset_row_id"].astype(int)
            df = df.set_index("dataset_row_id", drop=False)

            missing_factor_cols = [key for key in FACTOR_KEYS if f"factor__{key}" not in df.columns]
            duplicate_ids = int(df.index.duplicated().sum())

            Y_all = pd.DataFrame(
                {
                    key: pd.to_numeric(df[f"factor__{key}"], errors="coerce").fillna(0).astype(int).clip(0, 1)
                    for key in FACTOR_KEYS
                },
                index=df.index,
            )

            strength_rows = []
            for key in FACTOR_KEYS:
                support_mask = Y_all[key].eq(1)
                vc = df.loc[support_mask, f"strength__{key}"].fillna("NA").astype(str).str.lower().value_counts()
                strength_rows.append(
                    {
                        "factor_key": key,
                        "support": int(support_mask.sum()),
                        "direct": int(vc.get("direct", 0)),
                        "context": int(vc.get("context", 0)),
                        "weak": int(vc.get("weak", 0)),
                    }
                )
            support = pd.DataFrame(strength_rows).sort_values("support", ascending=False)

            overview = pd.DataFrame(
                [
                    {"metric": "rows", "value": len(df)},
                    {"metric": "unique_dataset_row_id", "value": df.index.nunique()},
                    {"metric": "duplicate_dataset_row_id", "value": duplicate_ids},
                    {"metric": "factor_columns_missing", "value": len(missing_factor_cols)},
                    {"metric": "factor_count", "value": len(FACTOR_KEYS)},
                    {"metric": "positive_factor_pairs", "value": int(Y_all.values.sum())},
                    {"metric": "mean_labels_per_news", "value": round(float(Y_all.sum(axis=1).mean()), 3)},
                    {"metric": "empty_news", "value": int((Y_all.sum(axis=1) == 0).sum())},
                    {"metric": "direct_labels", "value": int(support["direct"].sum())},
                    {"metric": "context_labels", "value": int(support["context"].sum())},
                    {"metric": "weak_labels", "value": int(support["weak"].sum())},
                    {"metric": "zero_support_factors", "value": int((support["support"] == 0).sum())},
                ]
            )
            overview
            """
        ),
        code(
            """
            display(support.head(20))
            display(support.tail(12))
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(11, 5))
            top = support.head(20).iloc[::-1]
            ax.barh(top["factor_key"], top["support"], color="#4c78a8")
            ax.set_title("Top factor supports in v3 gold")
            ax.set_xlabel("gold positive pairs")
            ax.grid(axis="x", alpha=0.25)
            plt.tight_layout()
            plt.show()
            """
        ),
        md("## Prediction coverage"),
        code(
            """
            source_paths = {
                "full_production_v1": OUT / "prompt_search_full_production_v1_gemma4-e2b_5df03950348d_b6a2d2f27b85.csv",
                "full_balanced_recall_v2": OUT / "prompt_search_full_balanced_recall_v2_gemma4-e2b_5df03950348d_e98b1a3c5abc.csv",
                "search_production_v1": OUT / "prompt_search_search_production_v1_gemma4-e2b_5df03950348d_b6a2d2f27b85.csv",
                "search_balanced_recall_v2": OUT / "prompt_search_search_balanced_recall_v2_gemma4-e2b_5df03950348d_e98b1a3c5abc.csv",
                "search_primary_first_v2": OUT / "prompt_search_search_primary_first_v2_gemma4-e2b_5df03950348d_5f153b0981f2.csv",
                "search_hard_negative_v2": OUT / "prompt_search_search_hard_negative_v2_gemma4-e2b_5df03950348d_1a35bcd28344.csv",
                "search_few_shot_major_v3": OUT / "prompt_search_search_few_shot_major_v3_gemma4-e2b_5df03950348d_0c3d39008fbb.csv",
                "recall_max_think_false": OUT / "thinking_recall_experiment_recall_max_v1_think_false.csv",
                "recall_max_think_true": OUT / "thinking_recall_experiment_recall_max_v1_think_true.csv",
                "grouped_event_think_true": OUT / "grouped_event_v1_search_think_true.csv",
                "latent_candidate_v2_bad_schema": OUT / "thinking_recall_experiment_latent_candidate_v2_think_true.csv",
                "latent_candidate_v3_think_true": OUT / "thinking_recall_experiment_latent_candidate_v3_think_true.csv",
            }

            coverage_rows = []
            for name, path in source_paths.items():
                if not path.exists():
                    coverage_rows.append({"source": name, "exists": False, "prediction_ids": 0, "overlap_v3": 0})
                    continue
                ids = pd.read_csv(path, usecols=["dataset_row_id"])["dataset_row_id"].astype(int).unique()
                coverage_rows.append(
                    {
                        "source": name,
                        "exists": True,
                        "prediction_ids": len(ids),
                        "overlap_v3": len(set(ids).intersection(df.index.astype(int))),
                        "min_id": int(np.min(ids)),
                        "max_id": int(np.max(ids)),
                    }
                )
            coverage = pd.DataFrame(coverage_rows)
            coverage
            """
        ),
        md(
            """
            ## Main scoreboard

            `full_overlap` rows are evaluated on the 739 v3 rows that already have full cached predictions.
            `search_overlap` rows are evaluated on the common 237-row overlap for prompt-search, thinking,
            grouped, and latent-candidate experiments.
            """
        ),
        code(
            """
            single_source = pd.read_csv(OUT / "v3_clean_broad_weak_single_source_threshold_metrics.csv")
            single_source["overlap_group"] = np.where(single_source["source"].str.startswith("full_"), "full_overlap", "search_overlap")

            best_single = (
                single_source.sort_values("micro_f1", ascending=False)
                .groupby("overlap_group", as_index=False)
                .head(8)
                [[
                    "overlap_group",
                    "config",
                    "n_news",
                    "gold_positive_pairs",
                    "pred_positive_pairs",
                    "micro_precision",
                    "micro_recall",
                    "micro_f1",
                    "macro_f1_supported",
                    "sample_f1_empty_correct",
                    "any_relevant_f1",
                    "false_relevant_news",
                    "missed_all_relevant_news",
                ]]
            )
            best_single.round(4)
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_df = best_single.head(12).iloc[::-1]
            ax.barh(plot_df["config"], plot_df["micro_f1"], color="#59a14f")
            ax.set_title("Best single-source thresholds on v3 labels")
            ax.set_xlabel("micro-F1")
            ax.set_xlim(0, max(0.55, float(plot_df["micro_f1"].max()) + 0.05))
            ax.grid(axis="x", alpha=0.25)
            plt.tight_layout()
            plt.show()
            """
        ),
        md("## Ensemble and label-scope checks"),
        code(
            """
            ens = pd.read_csv(OUT / "v3_clean_broad_weak_ensemble_metrics.csv")

            best_ens_all = ens[ens["label_scope"].eq("all")].sort_values("micro_f1", ascending=False).head(15)
            best_ens_direct_context = ens[ens["label_scope"].eq("direct_context")].sort_values("micro_f1", ascending=False).head(8)
            best_ens_direct = ens[ens["label_scope"].eq("direct")].sort_values("micro_f1", ascending=False).head(8)

            display(
                best_ens_all[
                    [
                        "config",
                        "n_news",
                        "gold_positive_pairs",
                        "pred_positive_pairs",
                        "micro_precision",
                        "micro_recall",
                        "micro_f1",
                        "macro_f1_supported",
                        "sample_f1_empty_correct",
                        "any_relevant_f1",
                        "missed_all_relevant_news",
                    ]
                ].round(4)
            )
            display(best_ens_direct_context[["config", "gold_positive_pairs", "micro_precision", "micro_recall", "micro_f1", "any_relevant_f1"]].round(4))
            display(best_ens_direct[["config", "gold_positive_pairs", "micro_precision", "micro_recall", "micro_f1", "any_relevant_f1"]].round(4))
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(best_ens_all["micro_recall"], best_ens_all["micro_precision"], s=70, color="#f28e2b")
            for _, row in best_ens_all.head(6).iterrows():
                ax.annotate(str(row["config"])[:28], (row["micro_recall"], row["micro_precision"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
            ax.set_title("Ensemble precision/recall tradeoff")
            ax.set_xlabel("micro-recall")
            ax.set_ylabel("micro-precision")
            ax.grid(alpha=0.25)
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Sentiment on true-positive factor pairs

            Sentiment is evaluated only where both gold and model agree that a factor is relevant.
            This isolates sentiment quality from factor-detection misses and false positives.

            Expansion-added labels are not included here because they do not have an independent LLM sentiment score.
            """
        ),
        code(
            """
            sentiment = pd.read_csv(OUT / "v3_clean_broad_weak_sentiment_metrics.csv")
            sentiment.round(4)
            """
        ),
        md(
            """
            ## V3-aware expansion experiment

            The broad v3 taxonomy often marks contextual co-labels. Example: one economic or industrial item
            can carry `consumer_price_index`, `real_income_index`, `per_capita_income`, `industrial_production_index`,
            and `enterprises_count` at once.

            A plain threshold cannot recover labels that no prompt emitted. This experiment learns simple
            candidate-expansion rules on half of the 237-row search overlap and reports the result on the other half.
            The goal is not to claim final production quality, but to test whether the error shape is fixable by
            v3-aware co-label expansion.
            """
        ),
        code(
            """
            expansion = pd.read_csv(OUT / "v3_clean_broad_weak_expansion_holdout_metrics.csv")
            prefix = pd.read_csv(OUT / "v3_clean_broad_weak_expansion_prefix_metrics.csv")
            rules = pd.read_csv(OUT / "v3_clean_broad_weak_greedy_expansion_rules.csv")

            display(expansion.round(4))
            display(rules.head(10).round(4))
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(8, 4))
            for split, sub in prefix.groupby("split"):
                ax.plot(sub["n_rules"], sub["micro_f1"], marker="o", label=split)
            ax.set_title("Expansion prefix size vs micro-F1")
            ax.set_xlabel("number of learned expansion rules")
            ax.set_ylabel("micro-F1")
            ax.grid(alpha=0.25)
            ax.legend()
            plt.tight_layout()
            plt.show()
            """
        ),
        md("## Per-factor diagnostics"),
        code(
            """
            practical = pd.read_csv(OUT / "v3_clean_broad_weak_best_practical_per_factor_report.csv")
            expanded = pd.read_csv(OUT / "v3_clean_broad_weak_expanded_per_factor_report.csv")

            compare = practical.merge(
                expanded[["factor_key", "precision", "recall", "f1", "predicted"]],
                on="factor_key",
                suffixes=("_base_ensemble", "_expanded"),
            )
            compare["f1_delta_expanded_minus_base"] = compare["f1_expanded"] - compare["f1_base_ensemble"]
            display(compare.sort_values("support", ascending=False).head(24).round(4))
            display(compare[compare["support"].gt(0)].sort_values("f1_base_ensemble").head(15).round(4))
            """
        ),
        code(
            """
            top_compare = compare[compare["support"].gt(0)].sort_values("support", ascending=False).head(18).iloc[::-1]
            y = np.arange(len(top_compare))
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(y - 0.18, top_compare["f1_base_ensemble"], height=0.35, label="base ensemble", color="#4c78a8")
            ax.barh(y + 0.18, top_compare["f1_expanded"], height=0.35, label="expanded", color="#e15759")
            ax.set_yticks(y)
            ax.set_yticklabels(top_compare["factor_key"])
            ax.set_xlabel("per-factor F1")
            ax.set_title("Per-factor F1: base ensemble vs expanded")
            ax.grid(axis="x", alpha=0.25)
            ax.legend()
            plt.tight_layout()
            plt.show()
            """
        ),
        md("## Problem examples"),
        code(
            """
            problems = pd.read_csv(OUT / "v3_clean_broad_weak_problem_examples_best_practical.csv")
            problems[["dataset_row_id", "title", "gold", "pred", "fp", "fn", "n_fp", "n_fn"]].head(20)
            """
        ),
        md(
            """
            ## Conclusions

            1. The v3 dataset is valid structurally: 36 factor columns are present, IDs are unique, UTF-8/BOM reads cleanly.
            2. The new gold is much broader than v2: 2765 positive factor pairs, 2.77 labels/news on average, and only 33 empty news.
            3. The LLM relevance gate is strong. On the 237-row search overlap, best broad prompts reach `any_relevant_f1` around 0.97.
            4. The main weakness is factor-set completeness, not "is this news relevant at all". The model often emits one obvious factor
               and misses contextual v3 co-labels.
            5. Best cached single source on v3-all labels is `latent_candidate_v3` at micro-F1 about 0.426.
            6. Best prompt union reaches micro-F1 about 0.475.
            7. A v3-aware expansion layer improves holdout micro-F1 from about 0.483 to about 0.556 with the first 5 learned rules.
               The full greedy 35-rule expansion reaches about 0.615 on all 237 rows, but its holdout curve shows overfitting risk.

            Practical next step: do not solve this only by lowering thresholds. Use a two-stage pipeline:
            high-recall candidate generation, v3-aware co-label expansion, then a verifier/reranker that removes weak false positives.
            """
        ),
    ]
    return nb


def main() -> None:
    nb = build()
    NOTEBOOK_PATH.parent.mkdir(exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    client = NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
