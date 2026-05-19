from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "v4_social_signal_prompt_evaluation.ipynb"


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
            # V4 social-signal prompt evaluation

            This notebook compares the new v4 prompt runs against the v3 social-signal experiments on the same v3-clean evaluation rows.

            Main metric: multilabel micro-F1 over the 36 factor labels.

            Supporting metrics:
            - `micro_precision`, `micro_recall`: pair-level factor quality.
            - `any_relevant_f1`: whether the model detects that a news item has any relevant factor.
            - `sample_f1_empty_correct`: per-news multilabel F1, with empty-empty counted as correct.
            - per-factor precision/recall/F1 for the selected v4 run.
            """
        ),
        code(
            """
            from pathlib import Path
            import re
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from sklearn.metrics import precision_recall_fscore_support

            ROOT = Path.cwd()
            OUT = ROOT / "analysis_outputs"
            DATASET_PATH = OUT / "v3_clean_broad_weak_dataset_1000.csv"
            PROMPT_PATH = ROOT / "FINAL_v4_prompt_social_signal_high_recall_ru.md"

            from analyzer.factors import FACTOR_CONFIG

            FACTOR_KEYS = [item["key"] for item in FACTOR_CONFIG]
            FACTOR_NAMES = {item["key"]: item["name"] for item in FACTOR_CONFIG}

            pd.set_option("display.max_columns", 100)
            pd.set_option("display.max_colwidth", 140)
            """
        ),
        md("## Prompt and dataset sanity"),
        code(
            """
            df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
            df["dataset_row_id"] = pd.to_numeric(df["dataset_row_id"], errors="raise").astype(int)
            df = df.set_index("dataset_row_id", drop=False)

            Y_all = pd.DataFrame(
                {
                    key: pd.to_numeric(df[f"factor__{key}"], errors="coerce").fillna(0).astype(int).clip(0, 1)
                    for key in FACTOR_KEYS
                },
                index=df.index,
            )

            prompt_text = PROMPT_PATH.read_text(encoding="utf-8")
            prompt_overview = pd.DataFrame(
                [
                    {"metric": "prompt_chars", "value": len(prompt_text)},
                    {"metric": "dataset_rows", "value": len(df)},
                    {"metric": "positive_factor_pairs", "value": int(Y_all.values.sum())},
                    {"metric": "mean_labels_per_news", "value": round(float(Y_all.sum(axis=1).mean()), 3)},
                    {"metric": "empty_news", "value": int((Y_all.sum(axis=1) == 0).sum())},
                    {"metric": "factor_count", "value": len(FACTOR_KEYS)},
                ]
            )
            prompt_overview
            """
        ),
        md("## Run discovery"),
        code(
            """
            def read_metrics(pattern: str, family: str) -> pd.DataFrame:
                rows = []
                for path in sorted(OUT.glob(pattern)):
                    frame = pd.read_csv(path)
                    frame["family"] = family
                    frame["metrics_file"] = path.name
                    frame["run_id"] = re.sub(r"^(v3|v4)_social_signal_experiment_metrics_", "", path.stem)
                    rows.append(frame)
                return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

            metrics = pd.concat(
                [
                    read_metrics("v3_social_signal_experiment_metrics_*.csv", "v3"),
                    read_metrics("v4_social_signal_experiment_metrics_*.csv", "v4"),
                ],
                ignore_index=True,
            )
            metrics["is_search237"] = metrics["metrics_file"].str.contains("search237", regex=False)
            metrics["is_pilot"] = metrics["metrics_file"].str.contains("pilot", case=False, regex=False)
            metrics["sort_scope"] = np.select(
                [metrics["is_search237"], metrics["is_pilot"]],
                ["search237", "pilot"],
                default="other",
            )
            coverage = (
                metrics.groupby(["family", "metrics_file"], as_index=False)
                .agg(n_news=("n_news", "max"), best_micro_f1=("micro_f1", "max"), best_any_f1=("any_relevant_f1", "max"))
                .sort_values(["family", "n_news", "best_micro_f1"], ascending=[True, False, False])
            )
            coverage
            """
        ),
        md("## Scoreboard"),
        code(
            """
            scoreboard = (
                metrics.sort_values(["n_news", "micro_f1", "micro_recall"], ascending=[False, False, False])
                .groupby(["family", "metrics_file"], as_index=False)
                .head(3)
                [[
                    "family",
                    "metrics_file",
                    "variant",
                    "threshold",
                    "n_news",
                    "gold_positive_pairs",
                    "pred_positive_pairs",
                    "pred_mean_labels",
                    "micro_precision",
                    "micro_recall",
                    "micro_f1",
                    "macro_f1_supported",
                    "sample_f1_empty_correct",
                    "any_relevant_f1",
                    "false_relevant_news",
                    "missed_all_relevant_news",
                ]]
                .sort_values(["n_news", "micro_f1"], ascending=[False, False])
            )
            scoreboard.round(4)
            """
        ),
        code(
            """
            plot_df = scoreboard.copy()
            plot_df["label"] = plot_df["family"] + " " + plot_df["metrics_file"].str.replace("_metrics_", "_", regex=False).str[:48] + " @" + plot_df["threshold"].astype(str)
            plot_df = plot_df.sort_values("micro_f1", ascending=True).tail(14)

            fig, ax = plt.subplots(figsize=(11, 6))
            colors = np.where(plot_df["family"].eq("v4"), "#e15759", "#4c78a8")
            ax.barh(plot_df["label"], plot_df["micro_f1"], color=colors)
            ax.set_title("Best thresholds by run")
            ax.set_xlabel("micro-F1")
            ax.grid(axis="x", alpha=0.25)
            plt.tight_layout()
            plt.show()
            """
        ),
        md("## Select best v4 prediction file"),
        code(
            """
            v4_metrics = metrics[metrics["family"].eq("v4")].copy()
            if v4_metrics.empty:
                raise RuntimeError("No v4 metrics found")

            prefer = v4_metrics[v4_metrics["is_search237"]].copy()
            if prefer.empty:
                prefer = v4_metrics.copy()

            best_v4 = prefer.sort_values(["micro_f1", "micro_recall"], ascending=False).iloc[0].to_dict()
            suffix = best_v4["metrics_file"].replace("v4_social_signal_experiment_metrics_", "").replace(".csv", "")
            pred_path = OUT / f"v4_social_signal_experiment_{suffix}.csv"
            if not pred_path.exists():
                pred_path = sorted(OUT.glob(f"v4_social_signal_experiment_{suffix}.partial.csv"))[-1]
            threshold = None if best_v4["threshold"] == "default" else float(best_v4["threshold"])

            best_v4_summary = pd.DataFrame([{
                "metrics_file": best_v4["metrics_file"],
                "prediction_file": pred_path.name,
                "threshold": best_v4["threshold"],
                "n_news": best_v4["n_news"],
                "micro_precision": best_v4["micro_precision"],
                "micro_recall": best_v4["micro_recall"],
                "micro_f1": best_v4["micro_f1"],
                "any_relevant_f1": best_v4["any_relevant_f1"],
            }])
            best_v4_summary.round(4)
            """
        ),
        md("## Per-factor diagnostics for selected v4 run"),
        code(
            """
            pred = pd.read_csv(pred_path)
            eval_ids = sorted(pred["dataset_row_id"].astype(int).unique().tolist())
            y_true = Y_all.reindex(index=eval_ids, columns=FACTOR_KEYS).fillna(0).astype(int)
            scores = (
                pred.pivot(index="dataset_row_id", columns="factor_key", values="relevance")
                .reindex(index=eval_ids, columns=FACTOR_KEYS)
                .fillna(0.0)
            )
            if threshold is None:
                y_pred = (
                    pred.pivot(index="dataset_row_id", columns="factor_key", values="is_relevant")
                    .reindex(index=eval_ids, columns=FACTOR_KEYS)
                    .fillna(False)
                    .astype(int)
                )
            else:
                y_pred = (scores >= threshold).astype(int)

            per_factor_rows = []
            for key in FACTOR_KEYS:
                p, r, f1, _ = precision_recall_fscore_support(y_true[key], y_pred[key], average="binary", zero_division=0)
                per_factor_rows.append(
                    {
                        "factor_key": key,
                        "factor_name": FACTOR_NAMES[key],
                        "support": int(y_true[key].sum()),
                        "predicted": int(y_pred[key].sum()),
                        "precision": float(p),
                        "recall": float(r),
                        "f1": float(f1),
                    }
                )
            per_factor = pd.DataFrame(per_factor_rows).sort_values(["support", "f1"], ascending=[False, False])
            display(per_factor.head(24).round(4))
            display(per_factor[per_factor["support"].gt(0)].sort_values("f1").head(16).round(4))
            """
        ),
        code(
            """
            top = per_factor[per_factor["support"].gt(0)].sort_values("support", ascending=False).head(18).iloc[::-1]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top["factor_key"], top["f1"], color="#59a14f")
            ax.set_title("Selected v4 per-factor F1")
            ax.set_xlabel("F1")
            ax.grid(axis="x", alpha=0.25)
            plt.tight_layout()
            plt.show()
            """
        ),
        md("## Strength breakdown"),
        code(
            """
            strength_rows = []
            for key in FACTOR_KEYS:
                strength_col = f"strength__{key}"
                if strength_col not in df.columns:
                    continue
                for strength in ["direct", "context", "weak"]:
                    mask = y_true[key].eq(1) & df.reindex(eval_ids)[strength_col].fillna("").astype(str).str.lower().eq(strength)
                    support = int(mask.sum())
                    if support == 0:
                        continue
                    hits = int((y_pred.loc[mask.index[mask], key] == 1).sum())
                    strength_rows.append({"strength": strength, "factor_key": key, "support": support, "hits": hits, "recall": hits / support})
            strength_report = pd.DataFrame(strength_rows)
            if not strength_report.empty:
                display(strength_report.groupby("strength", as_index=False).agg(support=("support", "sum"), hits=("hits", "sum")).assign(recall=lambda x: x["hits"] / x["support"]).round(4))
                display(strength_report.sort_values(["strength", "support"], ascending=[True, False]).head(30).round(4))
            else:
                print("No strength columns available")
            """
        ),
        md("## Problem examples"),
        code(
            """
            true_sets = {idx: set(y_true.columns[y_true.loc[idx].eq(1)]) for idx in y_true.index}
            pred_sets = {idx: set(y_pred.columns[y_pred.loc[idx].eq(1)]) for idx in y_pred.index}
            problem_rows = []
            for idx in eval_ids:
                gold = true_sets[idx]
                pred_set = pred_sets[idx]
                fp = sorted(pred_set - gold)
                fn = sorted(gold - pred_set)
                if fp or fn:
                    problem_rows.append(
                        {
                            "dataset_row_id": idx,
                            "title": df.loc[idx, "title"],
                            "gold": ", ".join(sorted(gold)),
                            "pred": ", ".join(sorted(pred_set)),
                            "fp": ", ".join(fp),
                            "fn": ", ".join(fn),
                            "n_fp": len(fp),
                            "n_fn": len(fn),
                        }
                    )
            problems = pd.DataFrame(problem_rows).sort_values(["n_fn", "n_fp"], ascending=False)
            problems.head(30)
            """
        ),
        md("## Raw v4 false-positive snippets"),
        code(
            """
            fp_rows = []
            pred_positive = pred[pred["dataset_row_id"].isin(eval_ids)].copy()
            if threshold is None:
                pred_positive = pred_positive[pred_positive["is_relevant"].astype(bool)]
            else:
                pred_positive = pred_positive[pred_positive["relevance"].astype(float).ge(threshold)]
            for _, row in pred_positive.iterrows():
                idx = int(row["dataset_row_id"])
                key = row["factor_key"]
                if key not in true_sets[idx]:
                    fp_rows.append(
                        {
                            "dataset_row_id": idx,
                            "factor_key": key,
                            "title": df.loc[idx, "title"],
                            "relevance": row["relevance"],
                            "evidence": row.get("evidence", ""),
                            "reason": row.get("reason", ""),
                        }
                    )
            fp_detail = pd.DataFrame(fp_rows).sort_values(["factor_key", "relevance"], ascending=[True, False])
            fp_detail.head(40)
            """
        ),
        md(
            """
            ## Working notes

            Use this notebook as the v4 prompt dashboard:
            1. Compare `v4` versus `v3` on the same `search237` rows when the full run is present.
            2. Check whether higher context and longer article text helped; pilot runs are intentionally kept in the scoreboard.
            3. Use `Problem examples` and `Raw v4 false-positive snippets` to revise the prompt or add deterministic post-filters.
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
