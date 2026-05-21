# Old gold

Открывать первым: [01 old gold workflow.ipynb](notebooks/01%20old%20gold%20workflow.ipynb).

Внутри notebook идет как человеческая работа: загрузка компактных данных, просмотр old gold sample, prompt-план, статусы запусков, код агрегации threshold и только потом результаты.

## Итог
- Лучший полный old gold micro-F1: 0.6599.
- Лучший полный recall: 0.7698.
- Полных прогонов `n_news >= 200`: 36.
- Пилоты отделены от полных запусков.

## Top full runs
| model           | thinking | prompt_version               | threshold | old_gold_n_news | old_gold_precision | old_gold_recall | old_gold_micro_f1 |
| --------------- | -------- | ---------------------------- | --------- | --------------- | ------------------ | --------------- | ----------------- |
| qwen3.6:35b-iq3 | false    | social_signal_v5             | 0.35      | 237.0           | 0.5774             | 0.7698          | 0.6599            |
| gemma4:26b      | false    | social_signal_v9_f1_balanced | 0.75      | 706.0           | 0.6179             | 0.6903          | 0.6521            |
| gemma4:26b      | false    | v4_prompt                    | 0.5       | 237.0           | 0.5939             | 0.6905          | 0.6385            |
| gemma4:26b      | false    | direct_core_targeted_v7      | 0.85      | 237.0           | 0.6192             | 0.6578          | 0.6379            |
| gemma4:e4b      | false    | social_signal_v5             | 0.75      | 237.0           | 0.6125             | 0.6587          | 0.6348            |
| gemma4:e4b      | true     | direct_core_targeted_v7      | 0.9       | 237.0           | 0.7283             | 0.56            | 0.6332            |
| gemma4:e4b      | true     | direct_core_recall_v2        | 0.85      | 237.0           | 0.6311             | 0.6311          | 0.6311            |
| gemma4:e4b      | true     | direct_core_gold_fewshot_v5  | 0.9       | 237.0           | 0.6721             | 0.5467          | 0.6029            |
