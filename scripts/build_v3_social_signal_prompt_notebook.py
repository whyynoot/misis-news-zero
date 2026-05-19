from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "v3_social_signal_prompt_experiment.ipynb"


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
            # V3 Social-Signal Prompt Experiment

            Goal: test whether a prompt aligned with the VKR methodology improves multilabel social-risk annotation.

            Methodological framing:
            - news item = unit of observation;
            - factor = unit of interpretation;
            - news is a weak social signal, not a direct measurement of social reality;
            - LLM output remains structured annotation for downstream aggregation and expert review;
            - output schema stays compatible: `relevance`, `sentiment`, `pressure`, `confidence`, `evidence`, `reason`.

            We do not predict `direct/context/weak` as separate target labels in the main task. They are used as
            diagnostic gold strata only: which kind of gold signal does the model catch.
            """
        ),
        code(
            """
            from pathlib import Path
            import json
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from sklearn.metrics import precision_recall_fscore_support

            ROOT = Path.cwd()
            OUT = ROOT / "analysis_outputs"
            DATASET = OUT / "v3_clean_broad_weak_dataset_1000.csv"

            from analyzer.factors import FACTOR_CONFIG
            FACTOR_KEYS = [item["key"] for item in FACTOR_CONFIG]
            pd.set_option("display.max_colwidth", 140)
            """
        ),
        md("## Why batch size mattered"),
        md(
            """
            The first social-signal pilot used `batch_size=6` and `max_tokens=8192`.

            `max_tokens` is only an upper generation cap. It does not force the model to write 8192 tokens.
            The stronger problem was batching: with several news items in one request, the local LLM had to keep
            several IDs, texts, factor decisions, evidence strings and JSON objects in memory at once. That increased
            schema errors and made the model conservative.

            The final runs use `batch_size=1`. This is slower with `thinking=true`, but methodologically cleaner:
            each news item gets a full factor checklist.
            """
        ),
        md("## Prompt Policy"),
        code(
            """
            prompt_policy = [
                "Разметь одну новость как источник слабых социальных сигналов.",
                "Верни все factor_id, с которыми есть содержательная связь; не выбирай один главный класс.",
                "relevance=0.9: прямой/главный сигнал; 0.6: явная связь; 0.3: слабая, но реальная аналитическая связь.",
                "Если релевантная новость дала только 0-1 фактор, перепроверь co-labels.",
                "Проверь группы: безопасность/смертность, медицина, цены/доходы, производство/компании, жилье/ЖКХ, экология, семья/дети, миграция/демография.",
            ]
            count_prior_v4 = [
                "В v3-разметке релевантная новость обычно содержит около 3 релевантных факторов.",
                "Для обычной релевантной новости целевой диапазон 2-4 фактора.",
                "Для комплексной новости про атаку, аварию, бюджет, санкции, промышленность, цены, медицину или ЖКХ допустимо 4-8 факторов.",
                "Лучше вернуть дополнительную потенциальную связь с relevance=0.3, чем пропустить фактор, который потом может быть отфильтрован downstream-верификатором.",
            ]
            co_label_examples = [
                {
                    "news": "удар по энергообъекту, погибший и раненые",
                    "expected": "crime_count; mortality_rate; life_expectancy; industrial_production_index",
                },
                {
                    "news": "авария на водоканале, дома остались без воды",
                    "expected": "housing_area_per_capita; wastewater_discharge",
                },
                {
                    "news": "бюджет ФОМС и Социального фонда",
                    "expected": "hospitals; per_capita_income; living_wage; children_benefits",
                },
                {
                    "news": "санкции против энергетической компании",
                    "expected": "industrial_production_index; enterprises_count; consumer_price_index only if population prices/tariffs are affected",
                },
                {
                    "news": "переговоры РФ-США без перемещения людей",
                    "expected": "no international_inflow/outflow only for diplomacy",
                },
            ]
            display(pd.DataFrame({"prompt_policy": prompt_policy}))
            display(pd.DataFrame({"explicit_count_prior_tested_in_v4": count_prior_v4}))
            display(pd.DataFrame(co_label_examples))
            """
        ),
        md("## Run Coverage"),
        code(
            """
            run_files = {
                "social_signal_v3_false": OUT / "v3_social_signal_experiment_social_signal_v3_think_false_v3_b1_search237.csv",
                "social_signal_v3_true": OUT / "v3_social_signal_experiment_social_signal_v3_think_true_v3_b1_search237_think.csv",
                "social_signal_v3_false_raw": OUT / "v3_social_signal_experiment_social_signal_v3_think_false_v3_b1_search237.raw.jsonl",
                "social_signal_v3_true_raw": OUT / "v3_social_signal_experiment_social_signal_v3_think_true_v3_b1_search237_think.raw.jsonl",
            }
            rows = []
            for name, path in run_files.items():
                if not path.exists():
                    rows.append({"run": name, "exists": False})
                    continue
                if path.suffix == ".csv":
                    pred = pd.read_csv(path)
                    rows.append(
                        {
                            "run": name,
                            "exists": True,
                            "rows": len(pred),
                            "news_ids": pred["dataset_row_id"].nunique(),
                            "positive_rows": int((pd.to_numeric(pred["relevance"], errors="coerce").fillna(0) > 0).sum()),
                            "mean_pred_labels_per_news": round(float((pd.to_numeric(pred["relevance"], errors="coerce").fillna(0) > 0).sum() / pred["dataset_row_id"].nunique()), 3),
                        }
                    )
                else:
                    statuses = []
                    for line in path.read_text(encoding="utf-8").splitlines():
                        statuses.append(json.loads(line).get("status"))
                    rows.append(
                        {
                            "run": name,
                            "exists": True,
                            "raw_calls": len(statuses),
                            "ok_calls": statuses.count("ok"),
                            "error_calls": statuses.count("error"),
                        }
                    )
            pd.DataFrame(rows)
            """
        ),
        md("## Main Metrics"),
        code(
            """
            metrics = pd.read_csv(OUT / "v3_social_signal_final_comparison_metrics.csv")
            main = metrics[metrics["scope"].eq("all")].sort_values("micro_f1", ascending=False)
            main[
                [
                    "config",
                    "gold_pairs",
                    "pred_pairs",
                    "mean_pred_labels",
                    "precision",
                    "recall",
                    "micro_f1",
                    "macro_f1_supported",
                    "sample_f1",
                    "any_f1",
                    "false_relevant",
                    "missed_all",
                ]
            ].round(4)
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(10, 4.8))
            plot_df = main.sort_values("micro_f1").copy()
            ax.barh(plot_df["config"], plot_df["micro_f1"], color="#4c78a8")
            ax.set_xlabel("micro-F1 over 36 binary factor labels")
            ax.set_title("V3 social-signal prompt comparison")
            ax.grid(axis="x", alpha=0.25)
            plt.tight_layout()
            plt.show()
            """
        ),
        md("## Explicit `~3 Factors Per Relevant News` Prior"),
        code(
            """
            v4 = pd.read_csv(OUT / "v3_social_signal_v4_comparison_metrics.csv")
            v4[
                [
                    "config",
                    "gold_pairs",
                    "pred_pairs",
                    "mean_pred_labels",
                    "precision",
                    "recall",
                    "micro_f1",
                    "any_f1",
                    "false_relevant",
                    "missed_all",
                ]
            ].round(4)
            """
        ),
        md(
            """
            The explicit count prior was tested as `social_signal_v4`.

            It did not improve this local model. The model became slower, had more JSON/schema failures, and returned
            fewer labels on average. So the lesson is not "never use a count prior"; the lesson is that for this
            Gemma run, count pressure inside the prompt is weaker than architectural expansion/reranking.
            """
        ),
        md("## Direct / Context / Weak Diagnostics"),
        code(
            """
            strength_view = metrics[metrics["scope"].isin(["direct", "context", "weak", "direct_context"])].copy()
            display(
                strength_view.sort_values(["scope", "micro_f1"], ascending=[True, False])[
                    ["scope", "config", "gold_pairs", "pred_pairs", "precision", "recall", "micro_f1", "any_f1", "missed_all"]
                ].round(4)
            )
            """
        ),
        code(
            """
            focus = strength_view[strength_view["config"].isin(["best_old_union", "social_signal_v3_think@0.05", "old_union+social_signal_think"])].copy()
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for config, sub in focus.groupby("config"):
                sub = sub.set_index("scope").loc[["direct", "context", "weak"]]
                ax.plot(sub.index, sub["micro_f1"], marker="o", label=config)
            ax.set_ylabel("micro-F1")
            ax.set_title("Metric by gold signal strength")
            ax.grid(alpha=0.25)
            ax.legend()
            plt.tight_layout()
            plt.show()
            """
        ),
        md("## Per-Factor Behavior"),
        code(
            """
            per_factor = pd.read_csv(OUT / "v3_social_signal_final_per_factor_report.csv")
            display(
                per_factor[per_factor["config"].eq("social_signal_v3_think@0.05")]
                .sort_values("support", ascending=False)
                .head(22)
                .round(4)
            )
            display(
                per_factor[per_factor["config"].eq("old_union+social_signal_think")]
                .sort_values("support", ascending=False)
                .head(22)
                .round(4)
            )
            """
        ),
        code(
            """
            compare = (
                per_factor[per_factor["config"].isin(["social_signal_v3_think@0.05", "old_union+social_signal_think"])]
                .pivot(index="factor_key", columns="config", values="f1")
                .join(per_factor.groupby("factor_key")["support"].max())
                .reset_index()
            )
            compare["delta"] = compare["old_union+social_signal_think"] - compare["social_signal_v3_think@0.05"]
            compare.sort_values("delta", ascending=False).head(15).round(4)
            """
        ),
        md("## Sentiment Metrics"),
        code(
            """
            sentiment = pd.read_csv(OUT / "v3_social_signal_final_sentiment_metrics.csv")
            sentiment.round(4)
            """
        ),
        md("## V5 Model Sweep"),
        md(
            """
            After the JSON repair parser and the shorter `social_signal_v5` prompt, the same evaluation set was
            rerun with three local models:

            - `gemma4:e2b`: original baseline model;
            - `gemma4:e4b`: same family, larger model;
            - `qwen3.5:9b`: larger alternative model.

            This isolates prompt/model behavior from dataset and metric changes. All runs use `batch_size=1`,
            `max_tokens=4096`, and no thinking mode.
            """
        ),
        code(
            """
            model_sweep = pd.read_csv(OUT / "v3_social_signal_model_sweep_comparison_metrics.csv")
            selected_configs = [
                "social_v5_e2b@default",
                "social_v5_gemma4_e4b@default",
                "social_v5_gemma4_e4b@0.60",
                "social_v5_qwen35_9b@default",
                "social_v5_qwen35_9b@0.40",
                "social_v5_qwen35_9b@0.60",
                "best_old_union",
                "old_union+social_v3_think@0.05",
                "gemma4_e4b@0.60 OR qwen35_9b@0.40",
                "grid_or_q0.60_g0.60",
            ]
            (
                model_sweep[model_sweep["config"].isin(selected_configs)]
                .sort_values("micro_f1", ascending=False)
                [
                    [
                        "config",
                        "pred_pairs",
                        "mean_pred_labels",
                        "precision",
                        "recall",
                        "micro_f1",
                        "macro_f1_supported",
                        "sample_f1",
                        "any_f1",
                        "false_relevant",
                        "missed_all",
                    ]
                ]
                .round(4)
            )
            """
        ),
        code(
            """
            plot_configs = [
                "social_v5_e2b@default",
                "social_v5_gemma4_e4b@default",
                "social_v5_gemma4_e4b@0.60",
                "social_v5_qwen35_9b@default",
                "social_v5_qwen35_9b@0.40",
                "gemma4_e4b@0.60 OR qwen35_9b@0.40",
            ]
            plot_df = model_sweep[model_sweep["config"].isin(plot_configs)].sort_values("micro_f1")
            fig, ax = plt.subplots(figsize=(10, 4.2))
            ax.barh(plot_df["config"], plot_df["micro_f1"], color="#2f6f7e")
            ax.set_xlabel("micro-F1 over 36 binary factor labels")
            ax.set_title("V5 prompt: model sweep")
            ax.grid(axis="x", alpha=0.25)
            plt.tight_layout()
            plt.show()
            """
        ),
        md("## JSON Stability"),
        code(
            """
            raw_status = pd.read_csv(OUT / "v3_social_signal_model_sweep_raw_status.csv")
            raw_status
            """
        ),
        md(
            """
            The larger models changed the main failure mode. With `gemma4:e2b`, the short prompt was technically
            better than previous long prompts, but still had a few JSON failures and returned too few factors.

            With `gemma4:e4b` and `qwen3.5:9b`, JSON failures disappeared on this 237-news run. More importantly,
            the models followed the "around 3 factors" prior much better.
            """
        ),
        md("## Threshold / Voting Search"),
        code(
            """
            grid = pd.read_csv(OUT / "v3_social_signal_model_sweep_grid_metrics.csv")
            grid.head(12)[
                [
                    "config",
                    "mode",
                    "q_threshold",
                    "g_threshold",
                    "pred_pairs",
                    "mean_pred_labels",
                    "precision",
                    "recall",
                    "micro_f1",
                    "sample_f1",
                    "any_f1",
                    "missed_all",
                ]
            ].round(4)
            """
        ),
        md(
            """
            The best simple voting setup is not the old ensemble. It is a two-model OR:
            `qwen3.5:9b >= 0.60 OR gemma4:e4b >= 0.60`.

            This reaches about `0.616` micro-F1 on the 237-news overlap. It is still not a final production
            calibrator, because the thresholds were selected on this same overlap, but it is a real improvement
            over the earlier prompt/model setup.
            """
        ),
        md("## Model-Sweep Signal Diagnostics"),
        code(
            """
            strength_sweep = pd.read_csv(OUT / "v3_social_signal_model_sweep_strength_metrics.csv")
            strength_sweep[
                strength_sweep["config"].isin(
                    [
                        "social_v5_qwen35_9b@default",
                        "social_v5_qwen35_9b@0.40",
                        "social_v5_gemma4_e4b@0.60",
                        "qwen35_0.60_OR_gemma4_0.60",
                    ]
                )
            ].sort_values(["scope", "micro_f1"], ascending=[True, False]).round(4)
            """
        ),
        code(
            """
            sentiment_sweep = pd.read_csv(OUT / "v3_social_signal_model_sweep_sentiment_metrics.csv")
            sentiment_sweep.round(4)
            """
        ),
        code(
            """
            pf_sweep = pd.read_csv(OUT / "v3_social_signal_model_sweep_per_factor.csv")
            (
                pf_sweep[(pf_sweep["config"].eq("qwen_default")) & (pf_sweep["support"] >= 10)]
                .sort_values("f1")
                .head(14)
                .round(4)
            )
            """
        ),
        md(
            """
            ## Interpretation

            The earlier low F1 was not just "bad data" and not just "broken JSON". Both mattered, but the larger
            model sweep shows the main bottleneck was model/prompt following:

            - `gemma4:e2b + v5` still averaged only about 1.1 predicted factors per news item and reached about
              `0.376` micro-F1.
            - `gemma4:e4b + v5` followed the count prior much better, averaged about 2.9 factors at the default
              gate, and reached about `0.570` micro-F1 at a stricter threshold.
            - `qwen3.5:9b + v5` was the best single-model run: about `0.586-0.589` micro-F1 depending on threshold,
              with default recall around `0.65`.
            - simple two-model voting improved the main score to about `0.616` micro-F1.

            The task is now behaving like a real multilabel weak-signal extraction problem. The remaining gap is
            concentrated in weak/context links and several ambiguous factor definitions. Direct signals are much
            easier; weak links are still noisy and require either expert relabeling, a verifier/reranker, or a
            calibrated second-stage expansion.

            Current best practical pipeline:
            short social-signal prompt + batch size 1 + larger LLM + thresholded two-model voting + downstream
            verifier for weak/context links.
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
