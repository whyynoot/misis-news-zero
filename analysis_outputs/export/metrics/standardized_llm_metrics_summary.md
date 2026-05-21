# Standardized LLM Metrics

Rows in all-threshold table: 1240
Runs in best-by-run table: 62
Rows with wide metrics: 1040
Rows with v4 strength metrics: 240
Rows with old gold metrics: 1220

## Top Wide Dataset, Together
| model           | thinking | context_window_tokens | max_output_tokens | prompt_version               | threshold | wide_n_news | wide_all_precision | wide_all_recall | wide_all_micro_f1 |
| --------------- | -------- | --------------------- | ----------------- | ---------------------------- | --------- | ----------- | ------------------ | --------------- | ----------------- |
| gemma4:e4b      | true     | 16384                 | 16384             | direct_core_gold_fewshot_v5  | 0.05      | 237         | 0.7259             | 0.4978          | 0.5906            |
| gemma4:26b      | false    | 32768                 | 16384             | direct_core_targeted_v7      | 0.65      | 237         | 0.8119             | 0.4626          | 0.5893            |
| qwen3.5:9b      | false    |                       |                   | social_signal_v5             | 0.35      | 237         | 0.7254             | 0.4954          | 0.5888            |
| gemma4:26b      | false    | 32768                 | 8192              | v4_prompt                    | 0.05      | 237         | 0.8147             | 0.4558          | 0.5846            |
| qwen3.6:35b-iq3 | false    | 16384                 |                   | social_signal_v5             | 0.45      | 237         | 0.8507             | 0.4345          | 0.5752            |
| gemma4:e4b      | false    |                       |                   | social_signal_v5             | 0.55      | 237         | 0.675              | 0.4939          | 0.5704            |
| gemma4:e4b      | true     | 16384                 | 16384             | direct_core_clean_fewshot_v6 | 0.05      | 237         | 0.7906             | 0.4435          | 0.5682            |
| qwen3.6:35b-iq3 | false    |                       |                   | social_signal_v5             | 0.35      | 30          | 0.8                | 0.4375          | 0.5657            |
| gemma4:e4b      | true     | 16384                 | 16384             | direct_core_balanced_v8      | 0.05      | 237         | 0.7978             | 0.4347          | 0.5627            |
| qwen3.6:35b-iq3 | false    | 16384                 |                   | social_signal_v5             | 0.35      | 30          | 0.8182             | 0.4219          | 0.5567            |
| qwen3.6:35b-iq3 | false    |                       |                   | v4_prompt                    | 0.40      | 4           | 0.8333             | 0.4167          | 0.5556            |
| gemma4:e4b      | true     | 16384                 | 16384             | direct_core_targeted_v7      | 0.05      | 237         | 0.8148             | 0.42            | 0.5543            |

## Top Wide Dataset, direct
| model           | thinking | prompt_version                 | threshold | wide_n_news | wide_direct_precision | wide_direct_recall | wide_direct_micro_f1 |
| --------------- | -------- | ------------------------------ | --------- | ----------- | --------------------- | ------------------ | -------------------- |
| gemma4:26b      | false    | direct_core_targeted_v7        | 0.80      | 237         | 0.5675                | 0.6471             | 0.6047               |
| gemma4:e4b      | true     | direct_core_targeted_v7        | 0.85      | 237         | 0.5752                | 0.5882             | 0.5817               |
| gemma4:e4b      | true     | direct_core_recall_v2          | 0.85      | 237         | 0.5733                | 0.5837             | 0.5785               |
| gemma4:e4b      | true     | direct_core_gold_fewshot_v5    | 0.90      | 237         | 0.623                 | 0.5158             | 0.5644               |
| qwen3.5:9b      | false    | social_signal_v5               | 0.65      | 237         | 0.583                 | 0.535              | 0.5579               |
| gemma4:e4b      | true     | direct_core_filtered_v3        | 0.80      | 237         | 0.5487                | 0.5611             | 0.5548               |
| qwen3.6:35b-iq3 | false    | social_signal_v5               | 0.45      | 237         | 0.4776                | 0.6584             | 0.5536               |
| gemma4:e4b      | true     | direct_core_balanced_v8        | 0.80      | 237         | 0.4719                | 0.6471             | 0.5458               |
| gemma4:e4b      | true     | direct_core_clean_fewshot_v6   | 0.80      | 237         | 0.498                 | 0.5747             | 0.5336               |
| gemma4:26b      | false    | v4_prompt                      | 0.45      | 237         | 0.462                 | 0.6255             | 0.5315               |
| gemma4:e4b      | true     | direct_core_gold_calibrated_v4 | 0.80      | 237         | 0.515                 | 0.543              | 0.5286               |
| qwen3.6:35b-iq3 | false    | social_signal_v5               | 0.35      | 30          | 0.4242                | 0.7                | 0.5283               |

## Top Wide Dataset, context
| model           | thinking | prompt_version               | threshold | wide_n_news | wide_context_precision | wide_context_recall | wide_context_micro_f1 |
| --------------- | -------- | ---------------------------- | --------- | ----------- | ---------------------- | ------------------- | --------------------- |
| qwen3.6:35b-iq3 | false    | v4_prompt                    | 0.40      | 4           | 0.5                    | 0.75                | 0.6                   |
| gemma4:e4b      | true     | direct_core_gold_fewshot_v5  | 0.75      | 237         | 0.2614                 | 0.4122              | 0.3199                |
| gemma4:26b      | false    | v4_prompt                    | 0.45      | 237         | 0.2827                 | 0.3591              | 0.3163                |
| gemma4:e4b      | true     | direct_core_clean_fewshot_v6 | 0.05      | 237         | 0.2565                 | 0.3513              | 0.2965                |
| qwen3.5:9b      | false    | social_signal_v5             | 0.35      | 237         | 0.2299                 | 0.3977              | 0.2914                |
| gemma4:e4b      | false    | social_signal_v5             | 0.55      | 237         | 0.2188                 | 0.4054              | 0.2842                |
| gemma4:e4b      | true     | direct_core_balanced_v8      | 0.05      | 237         | 0.248                  | 0.3297              | 0.2831                |
| qwen3.6:35b-iq3 | false    | social_signal_v5             | 0.45      | 237         | 0.2507                 | 0.3243              | 0.2828                |
| gemma4:e4b      | true     | gold_mimic_broad_v1          | 0.05      | 237         | 0.2241                 | 0.3728              | 0.2799                |
| qwen3.6:35b-iq3 | false    | social_signal_v5             | 0.05      | 30          | 0.1687                 | 0.5833              | 0.2617                |
| gemma4:e2b      | false    | v4_prompt                    | 0.60      | 18          | 0.2105                 | 0.3333              | 0.2581                |
| gemma4:e2b      | true     | social_signal_v3             | 0.05      | 237         | 0.2446                 | 0.2625              | 0.2533                |

## Top Wide Dataset, weak
| model           | thinking | prompt_version               | threshold | wide_n_news | wide_weak_precision | wide_weak_recall | wide_weak_micro_f1 |
| --------------- | -------- | ---------------------------- | --------- | ----------- | ------------------- | ---------------- | ------------------ |
| qwen3.6:35b-iq3 | false    | v4_prompt                    | 0.70      | 4           | 1.0                 | 0.25             | 0.4                |
| qwen3.6:35b-iq3 | false    | social_signal_v5             | 0.35      | 30          | 0.2121              | 0.35             | 0.2642             |
| qwen3.6:35b-iq3 | false    | social_signal_v5             | 0.35      | 30          | 0.2                 | 0.35             | 0.2545             |
| gemma4:26b      | false    | direct_core_targeted_v7      | 0.65      | 237         | 0.1598              | 0.3425           | 0.2179             |
| gemma4:26b      | false    | social_signal_v3             | 0.35      | 18          | 0.2                 | 0.2308           | 0.2143             |
| gemma4:e4b      | true     | direct_core_clean_fewshot_v6 | 0.65      | 237         | 0.1474              | 0.3094           | 0.1996             |
| gemma4:e4b      | true     | direct_core_balanced_v8      | 0.05      | 237         | 0.1482              | 0.3039           | 0.1993             |
| gemma4:e4b      | true     | direct_core_targeted_v7      | 0.05      | 237         | 0.1481              | 0.2873           | 0.1955             |
| gemma4:e4b      | false    | social_signal_v5             | 0.55      | 237         | 0.1271              | 0.3961           | 0.1924             |
| gemma4:e4b      | true     | gold_mimic_broad_v1          | 0.75      | 237         | 0.1345              | 0.3315           | 0.1914             |
| gemma4:e2b      | false    | social_signal_v3             | 0.35      | 18          | 0.25                | 0.1538           | 0.1905             |
| gemma4:e2b      | false    | v4_prompt                    | 0.60      | 18          | 0.1579              | 0.2308           | 0.1875             |

## Top Old Gold Only
| model           | thinking | prompt_version          | threshold | old_gold_n_news | old_gold_precision | old_gold_recall | old_gold_micro_f1 |
| --------------- | -------- | ----------------------- | --------- | --------------- | ------------------ | --------------- | ----------------- |
| qwen3.6:35b-iq3 | false    | social_signal_v5        | 0.35      | 30              | 0.6364             | 0.9545          | 0.7636            |
| qwen3.6:35b-iq3 | false    | social_signal_v5        | 0.35      | 30              | 0.6                | 0.9545          | 0.7368            |
| qwen3.5:9b      | true     | direct_core_targeted_v7 | 0.05      | 1               | 0.5                | 1.0             | 0.6667            |
| gemma4:26b      | false    | social_signal_v3        | 0.35      | 18              | 0.6                | 0.75            | 0.6667            |
| qwen3.6:35b-iq3 | false    | social_signal_v5        | 0.35      | 237             | 0.5774             | 0.7698          | 0.6599            |
| gemma4:26b      | false    | v4_prompt               | 0.50      | 237             | 0.5939             | 0.6905          | 0.6385            |
| gemma4:26b      | false    | direct_core_targeted_v7 | 0.85      | 237             | 0.6192             | 0.6578          | 0.6379            |
| gemma4:e4b      | false    | social_signal_v5        | 0.75      | 237             | 0.6125             | 0.6587          | 0.6348            |
| gemma4:e4b      | true     | direct_core_targeted_v7 | 0.90      | 237             | 0.7283             | 0.56            | 0.6332            |
| gemma4:e4b      | true     | direct_core_recall_v2   | 0.85      | 237             | 0.6311             | 0.6311          | 0.6311            |
| gemma4:26b      | true     | direct_core_targeted_v7 | 0.70      | 8               | 0.5                | 0.8333          | 0.625             |
| gemma4:e2b      | false    | v4_prompt               | 0.75      | 18              | 0.5714             | 0.6667          | 0.6154            |

## Latest Old Gold Check
| model      | thinking | context_window_tokens | max_output_tokens | batch_size | prompt_version          | threshold | old_gold_n_news | old_gold_precision | old_gold_recall | old_gold_micro_f1 | wide_all_micro_f1 | v4_strength_direct_micro_f1 |
| ---------- | -------- | --------------------- | ----------------- | ---------- | ----------------------- | --------- | --------------- | ------------------ | --------------- | ----------------- | ----------------- | --------------------------- |
| gemma4:26b | false    | 32768                 | 16384             | 2          | direct_core_targeted_v7 | 0.85      | 237             | 0.6192             | 0.6578          | 0.6379            | 0.4717            | 0.6                         |

## Thinking Old Gold Pilots
| model      | thinking | context_window_tokens | max_output_tokens | batch_size | prompt_version          | threshold | old_gold_n_news | old_gold_precision | old_gold_recall | old_gold_micro_f1 | old_gold_error_rows |
| ---------- | -------- | --------------------- | ----------------- | ---------- | ----------------------- | --------- | --------------- | ------------------ | --------------- | ----------------- | ------------------- |
| gemma4:26b | true     | 32768                 | 8192              | 1          | direct_core_targeted_v7 | 0.70      | 8               | 0.5                | 0.8333          | 0.625             | 0.0                 |
| gemma4:26b | true     | 32768                 | 8192              | 1          | direct_core_targeted_v7 | 0.70      | 30              | 0.4722             | 0.7727          | 0.5862            | 0.0                 |
| gemma4:26b | true     | 32768                 | 8192              | 1          | social_signal_v5        | 0.45      | 10              | 0.3636             | 0.5714          | 0.4444            | 36.0                |
| qwen3.5:9b | true     | 32768                 | 8192              | 1          | direct_core_targeted_v7 | 0.05      | 3               | 0.3333             | 0.3333          | 0.3333            | 72.0                |
| gemma4:26b | true     | 32768                 | 8192              | 1          | social_signal_v5        | 0.35      | 3               | 0.2                | 0.3333          | 0.25              | 0.0                 |

## Thinking Trace Diagnostics
| run_label                                      | response_rows | ok_rows | error_rows | length_rows | empty_content_rows | avg_thinking_chars | max_thinking_chars | avg_duration_s | wait_count | final_check_count | markdown_count |
| ---------------------------------------------- | ------------- | ------- | ---------- | ----------- | ------------------ | ------------------ | ------------------ | -------------- | ---------- | ----------------- | -------------- |
| gemma4:26b b1 ctx32768 tok4096 guard           | 6             | 4       | 2          | 2           | 2                  | 7204.5             | 13928              | 73.6           | 102        | 14                | 94             |
| gemma4:26b b1 ctx32768 tok8192                 | 3             | 2       | 1          | 1           | 1                  | 12926.3            | 26573              | 123.6          | 10         | 360               | 6              |
| old_gold gemma4:26b b1 ctx32768 tok8192 guard2 | 10            | 9       | 1          | 1           | 1                  | 9521.4             | 26843              | 92.9           | 63         | 384               | 5              |
| old_gold gemma4:26b b1 ctx32768 tok8192 guard2 | 3             | 3       | 0          | 0           | 0                  | 5537.3             | 6070               | 52.2           | 5          | 7                 | 2              |
| old_gold gemma4:26b b1 ctx32768 tok8192 guard2 | 30            | 30      | 0          | 0           | 0                  | 279.4              | 5868               | 11.4           | 3          | 2                 | 2              |
| old_gold gemma4:26b b1 ctx32768 tok8192 guard2 | 8             | 8       | 0          | 0           | 0                  | 0.0                | 0                  | 11.0           | 0          | 0                 | 0              |
| old_gold qwen3.5:9b b1 ctx32768 tok8192 guard2 | 3             | 1       | 2          | 1           | 2                  | 15128.7            | 29208              | 209.2          | 243        | 2                 | 3              |
| qwen3.5:9b b1 ctx32768 tok4096 guard           | 1             | 0       | 1          | 1           | 1                  | 14949.0            | 14949              | 199.0          | 83         | 0                 | 0              |
| qwen3.5:9b b1 ctx32768 tok8192                 | 1             | 1       | 0          | 0           | 0                  | 19813.0            | 19813              | 275.7          | 55         | 4                 | 5              |
| qwen3.5:9b b2 ctx65536 tok32768                | 3             | 0       | 3          | 0           | 3                  | 0.0                | 0                  | 0.0            | 0          | 0                 | 0              |

## Column Standard
- `wide_all_*` is the together score on the broad weak/direct/context dataset.
- `wide_direct_*`, `wide_context_*`, and `wide_weak_*` are separated strength scores.
- `v4_strength_*` is the same strength split on the normalized v4 benchmark subset.
- `old_gold_*` is evaluated only on the old multilabel gold dataset.
- `context_window_tokens` and `max_output_tokens` are parsed from `ctx*` and `tok*` filename tokens.