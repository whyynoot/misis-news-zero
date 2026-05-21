# V4 wide

Открывать первым: [01 v4 wide workflow.ipynb](notebooks/01%20v4%20wide%20workflow.ipynb).

Внутри notebook: просмотр v4 wide sample, prompt-план, статус inference, код threshold/strength агрегации, pilot60, full winner и threshold trade-off.

## Итог
- Лучший v4-strength micro-F1: 0.6151.
- Лучший wide micro-F1: 0.5906.
- Основной полный winner: `gemma4:26b + social_signal_v9_f1_balanced + no thinking`, 965 новостей.

## Top v4 strength runs
| model      | thinking | prompt_version               | threshold | v4_strength_n_news | v4_strength_all_precision | v4_strength_all_recall | v4_strength_all_micro_f1 | old_gold_micro_f1 |
| ---------- | -------- | ---------------------------- | --------- | ------------------ | ------------------------- | ---------------------- | ------------------------ | ----------------- |
| gemma4:26b | false    | social_signal_v9_f1_balanced | 0.05      | 965.0              | 0.7843                    | 0.506                  | 0.6151                   | 0.5805            |
| gemma4:e4b | true     | direct_core_gold_fewshot_v5  | 0.05      | 237.0              | 0.7259                    | 0.4978                 | 0.5906                   | 0.5058            |
| gemma4:26b | false    | direct_core_targeted_v7      | 0.65      | 237.0              | 0.8119                    | 0.4626                 | 0.5893                   | 0.5808            |
| gemma4:e4b | true     | direct_core_clean_fewshot_v6 | 0.05      | 237.0              | 0.7906                    | 0.4435                 | 0.5682                   | 0.5437            |
| gemma4:e4b | true     | direct_core_balanced_v8      | 0.05      | 237.0              | 0.7978                    | 0.4347                 | 0.5627                   | 0.5403            |
| gemma4:26b | false    | social_signal_v9_f1_balanced | 0.05      | 60.0               | 0.7                       | 0.4667                 | 0.56                     | 0.5818            |
| gemma4:e4b | true     | direct_core_targeted_v7      | 0.05      | 237.0              | 0.8148                    | 0.42                   | 0.5543                   | 0.5903            |
| qwen3.5:9b | false    | social_signal_v9_f1_balanced | 0.05      | 60.0               | 0.625                     | 0.4848                 | 0.5461                   | 0.5041            |
