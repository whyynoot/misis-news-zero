from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "event_level_grouped_llm_experiment.ipynb"


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
            # Event-level grouped LLM experiment

            Цель: проверить гипотезу, что факторный каталог должен быть не статистическим
            (`mortality_rate = коэффициент смертности`), а событийным
            (`mortality_rate = гибель людей, погибшие, летальные исходы`), и что 36 факторов
            лучше классифицировать группами.

            В этом ноутбуке уже используются сохранённые результаты LLM-прогона:

            - `balanced_recall_v2`: лучший одиночный baseline из прошлого ноутбука;
            - `recall_max_v1_think_true`: high-recall prompt с thinking;
            - `grouped_event_v1_search_think_true`: новый grouped event-level prompt.

            Основная метрика остаётся `factor_micro_f1` по 36 бинарным факторным меткам.
            Вспомогательные метрики объясняют, за счёт чего меняется качество.
            """
        ),
        code(
            """
            from pathlib import Path
            import numpy as np
            import pandas as pd
            from sklearn.metrics import precision_recall_fscore_support, f1_score

            ROOT = Path.cwd()
            OUTPUT_DIR = ROOT / "analysis_outputs"
            DATASET_PATH = OUTPUT_DIR / "interfax_news_multilabel_factor_dataset_1000_wide.csv"

            from analyzer.factors import FACTOR_CONFIG
            from analyzer.event_factor_taxonomy import EVENT_FACTOR_GROUPS, EVENT_FACTOR_TAXONOMY

            FACTOR_KEYS = [item["key"] for item in FACTOR_CONFIG]
            FACTOR_NAMES = {item["key"]: item["name"] for item in FACTOR_CONFIG}

            paths = {
                "baseline_balanced": OUTPUT_DIR / "prompt_search_search_balanced_recall_v2_gemma4-e2b_5df03950348d_e98b1a3c5abc.csv",
                "thinking_recall": OUTPUT_DIR / "thinking_recall_experiment_recall_max_v1_think_true.csv",
                "grouped_event": OUTPUT_DIR / "grouped_event_v1_search_think_true.csv",
            }
            for name, path in paths.items():
                print(name, path.exists(), path)
            """
        ),
        md(
            """
            ## Taxonomy sanity check

            Проверяем, что event-level каталог покрывает все 36 факторов ровно один раз
            в grouped pipeline.
            """
        ),
        code(
            """
            covered = [key for group in EVENT_FACTOR_GROUPS.values() for key in group["factor_keys"]]
            taxonomy_check = {
                "factor_count": len(FACTOR_KEYS),
                "taxonomy_entries": len(EVENT_FACTOR_TAXONOMY),
                "grouped_entries": len(covered),
                "missing_from_groups": sorted(set(FACTOR_KEYS) - set(covered)),
                "duplicate_in_groups": sorted({key for key in covered if covered.count(key) > 1}),
                "missing_from_event_taxonomy": sorted(set(FACTOR_KEYS) - set(EVENT_FACTOR_TAXONOMY)),
            }
            taxonomy_check
            """
        ),
        code(
            """
            group_rows = []
            for group_id, payload in EVENT_FACTOR_GROUPS.items():
                for key in payload["factor_keys"]:
                    event = EVENT_FACTOR_TAXONOMY[key]
                    group_rows.append({
                        "group": group_id,
                        "factor_key": key,
                        "dashboard_name": FACTOR_NAMES[key],
                        "event_name": event["event_name"],
                        "event_definition": event["include"],
                    })
            pd.DataFrame(group_rows)
            """
        ),
        md(
            """
            ## Metric functions

            `sample_f1_empty_correct` считается так же, как в предыдущем multilabel notebook:
            если в новости нет gold-факторов и модель тоже ничего не вернула, это 1.0.
            """
        ),
        code(
            """
            df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
            df["dataset_row_id"] = pd.to_numeric(df[df.columns[0]], errors="raise").astype(int)

            y_true_df = pd.DataFrame(
                {
                    key: pd.to_numeric(df[f"factor__{key}"], errors="coerce").fillna(0).astype(int).clip(0, 1).to_numpy()
                    for key in FACTOR_KEYS
                },
                index=df["dataset_row_id"],
            )

            def load_scores(path: Path) -> pd.DataFrame:
                pred = pd.read_csv(path)
                ids = sorted(pred["dataset_row_id"].unique().astype(int).tolist())
                scores = pd.DataFrame(0.0, index=ids, columns=FACTOR_KEYS)
                for key, sub in pred.groupby("factor_key"):
                    if key in scores.columns:
                        values = sub.groupby("dataset_row_id")["relevance"].max()
                        scores.loc[values.index.astype(int), key] = values.astype(float).values
                return scores

            scores = {name: load_scores(path) for name, path in paths.items()}
            ids = sorted(set.intersection(*(set(frame.index) for frame in scores.values())))
            Y = y_true_df.loc[ids, FACTOR_KEYS].to_numpy(int)

            def sample_f1_empty_correct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
                values = []
                for true_row, pred_row in zip(y_true, y_pred):
                    true_sum = int(true_row.sum())
                    pred_sum = int(pred_row.sum())
                    tp = int(((true_row == 1) & (pred_row == 1)).sum())
                    if true_sum == 0 and pred_sum == 0:
                        values.append(1.0)
                    elif true_sum == 0 or pred_sum == 0:
                        values.append(0.0)
                    else:
                        values.append(2 * tp / (true_sum + pred_sum))
                return float(np.mean(values))

            def evaluate_matrix(y_pred: np.ndarray, name: str) -> dict:
                micro = precision_recall_fscore_support(Y, y_pred, average="micro", zero_division=0)
                supported = Y.sum(axis=0) > 0
                any_metric = precision_recall_fscore_support(
                    Y.sum(axis=1) > 0,
                    y_pred.sum(axis=1) > 0,
                    average="binary",
                    zero_division=0,
                )
                return {
                    "name": name,
                    "pred_positive_pairs": int(y_pred.sum()),
                    "mean_labels_per_news": float(y_pred.sum(axis=1).mean()),
                    "factor_micro_precision": float(micro[0]),
                    "factor_micro_recall": float(micro[1]),
                    "factor_micro_f1": float(micro[2]),
                    "factor_macro_f1_supported": float(f1_score(Y[:, supported], y_pred[:, supported], average="macro", zero_division=0)),
                    "sample_f1_empty_correct": sample_f1_empty_correct(Y, y_pred),
                    "any_relevant_precision": float(any_metric[0]),
                    "any_relevant_recall": float(any_metric[1]),
                    "any_relevant_f1": float(any_metric[2]),
                    "false_relevant_news": int(((Y.sum(axis=1) == 0) & (y_pred.sum(axis=1) > 0)).sum()),
                    "missed_all_relevant_news": int(((Y.sum(axis=1) > 0) & (y_pred.sum(axis=1) == 0)).sum()),
                }

            print("search ids:", len(ids))
            print("gold positive factor pairs:", int(Y.sum()))
            """
        ),
        md(
            """
            ## Main scoreboard

            Сравниваем одиночные модели, старый ensemble и новый event-level grouped ensemble.
            """
        ),
        code(
            """
            B = scores["baseline_balanced"].loc[ids, FACTOR_KEYS].to_numpy(float)
            T = scores["thinking_recall"].loc[ids, FACTOR_KEYS].to_numpy(float)
            G = scores["grouped_event"].loc[ids, FACTOR_KEYS].to_numpy(float)

            variants = {
                "balanced_recall_v2@0.30": (B >= 0.30).astype(int),
                "thinking_recall@0.30": (T >= 0.30).astype(int),
                "grouped_event@0.35": (G >= 0.35).astype(int),
                "grouped_event@0.05_high_recall": (G >= 0.05).astype(int),
                "old_ensemble_baseline+thinking@0.35": np.logical_or(B >= 0.35, T >= 0.35).astype(int),
                "new_ensemble_baseline+thinking+grouped@0.35": np.logical_or.reduce([B >= 0.35, T >= 0.35, G >= 0.35]).astype(int),
                "new_ensemble_high_recall@0.10/0.35/0.35": np.logical_or.reduce([B >= 0.10, T >= 0.35, G >= 0.35]).astype(int),
            }

            scoreboard = pd.DataFrame([evaluate_matrix(pred, name) for name, pred in variants.items()])
            scoreboard.sort_values(["factor_micro_f1", "sample_f1_empty_correct"], ascending=False).round(4)
            """
        ),
        md(
            """
            ## Threshold and tuning result

            `grouped_event` сам по себе оказался слишком слабым по micro-F1. Его ценность не в одиночном режиме,
            а в том, что он добавляет часть редких/спорных факторов к ensemble.
            """
        ),
        code(
            """
            grouped_metrics = pd.read_csv(OUTPUT_DIR / "grouped_event_v1_search_think_true_metrics.csv")
            grouped_metrics[
                [
                    "mode",
                    "threshold",
                    "factor_micro_f1",
                    "factor_micro_precision",
                    "factor_micro_recall",
                    "factor_macro_f1_supported",
                    "sample_f1_empty_correct",
                    "any_relevant_f1",
                    "mean_pred_labels_per_news",
                    "false_relevant_news",
                    "missed_all_relevant_news",
                ]
            ].head(12).round(4)
            """
        ),
        code(
            """
            hybrid_metrics = pd.read_csv(OUTPUT_DIR / "grouped_event_hybrid_search_metrics_empty_correct.csv")
            hybrid_metrics.head(12).round(4)
            """
        ),
        code(
            """
            holdout_metrics = pd.read_csv(OUTPUT_DIR / "grouped_event_hybrid_factor_tuned_holdout_metrics.csv")
            holdout_metrics.round(4)
            """
        ),
        md(
            """
            ## Per-factor diagnostics

            Смотрим, где новый grouped prompt добавил пользу, а где испортил качество.
            """
        ),
        code(
            """
            def per_factor_metrics(y_pred: np.ndarray, label: str) -> pd.DataFrame:
                rows = []
                for i, key in enumerate(FACTOR_KEYS):
                    pr = precision_recall_fscore_support(Y[:, i], y_pred[:, i], average="binary", zero_division=0)
                    rows.append({
                        "factor_key": key,
                        "factor_name": FACTOR_NAMES[key],
                        "support": int(Y[:, i].sum()),
                        "predicted_positive": int(y_pred[:, i].sum()),
                        "precision": float(pr[0]),
                        "recall": float(pr[1]),
                        "f1": float(pr[2]),
                        "variant": label,
                    })
                return pd.DataFrame(rows)

            old_ensemble = variants["old_ensemble_baseline+thinking@0.35"]
            new_ensemble = variants["new_ensemble_baseline+thinking+grouped@0.35"]
            grouped_only = variants["grouped_event@0.35"]

            per_old = per_factor_metrics(old_ensemble, "old_ensemble")
            per_new = per_factor_metrics(new_ensemble, "new_ensemble")
            per_grouped = per_factor_metrics(grouped_only, "grouped_only")

            factor_compare = (
                per_new[["factor_key", "factor_name", "support", "predicted_positive", "precision", "recall", "f1"]]
                .merge(
                    per_old[["factor_key", "predicted_positive", "precision", "recall", "f1"]],
                    on="factor_key",
                    suffixes=("_new", "_old"),
                )
            )
            factor_compare["f1_delta_vs_old"] = factor_compare["f1_new"] - factor_compare["f1_old"]
            factor_compare["recall_delta_vs_old"] = factor_compare["recall_new"] - factor_compare["recall_old"]
            factor_compare[factor_compare["support"] > 0].sort_values("f1_delta_vs_old", ascending=False).round(4)
            """
        ),
        code(
            """
            grouped_report = pd.read_csv(OUTPUT_DIR / "grouped_event_v1_search_think_true_factor_report.csv")
            grouped_report[grouped_report["support"] > 0].sort_values("f1", ascending=False).round(4)
            """
        ),
        md(
            """
            ## Conclusion

            1. Event-level taxonomy как идея правильная: `mortality_rate`, `hospitals`, часть housing/children факторов
            стали лучше объяснимы, а `factor_macro_f1_supported` в новом ensemble вырос.

            2. Grouped-only pipeline не надо выкатывать как замену baseline: лучший grouped-only micro-F1 на search
            около 0.339, это хуже текущего balanced baseline.

            3. Лучший практический кандидат из уже прогнанных вариантов: union ensemble
            `balanced_recall_v2 + recall_max_thinking + grouped_event`, threshold около 0.35.
            Он даёт лучший общий баланс на search: выше recall и macro-supported, но precision проседает.

            4. Ожидание F1 0.7-0.8 одним prompt-тюнингом здесь не подтверждается. Следующий правильный шаг:
            двухстадийная схема candidate generator -> verifier/reranker по парам news-factor, желательно с
            supervised calibration на этом gold dataset.
            """
        ),
    ]
    return nb


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(exist_ok=True)
    nb = build()
    nbf.write(nb, NOTEBOOK_PATH)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
