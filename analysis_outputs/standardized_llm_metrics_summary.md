# Standardized LLM Metrics

Rows in all-threshold table: 1580
Runs in best-by-run table: 79
V4 together rows in this summary: 43
Old-gold rows in this summary: 84

## Reading This Summary
Each table keeps pilots, partials, checkpoints, large runs, and full runs together. `n_news` and `status` show how much of the dataset was evaluated. Each row is one run/result at its best threshold for that benchmark and scope.

Tables are sorted from worst run to best run by `micro_f1` ascending. This keeps all results inside the same semantic table while making the strongest runs appear at the bottom.

The V4 benchmark is split by meaningful strength scope: `Together` includes all labels, then `Direct`, `Context`, and `Weak` show the separated slices. Checkpoint rows are available only for `Together`, because checkpoint metrics were computed as all-label micro metrics.

Current V4 together leader: `gemma4:26b + social_signal_v9_f1_balanced`, non-thinking, `ctx32768/tok16384/b2`, full 965 rows, `micro-F1=0.6151`. Best qwen3.6/iQ3 V4 direction so far is `social_signal_v9_f1_balanced` around `F1=0.58` on the 237/242-row checks; `high_recall_minimal_v3` is weaker on V4 together but remains visible in all relevant tables.

## V4 Together Results
| run | model | prompt | status | thinking | ctx | tok | b | n_news | threshold | precision | recall | micro_f1 | gold_pairs | pred_pairs | errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tt_qwen35_9b_direct_core_targeted_v7_think_guard_ctx32768_tok4096_b1_pilot3.partial | qwen3.5:9b | direct_core_targeted_v7 | partial | true | 32768 | 4096 | 1 | 1 | default | 0.0000 | 0.0000 | 0.0000 | 3 | 0 |  |
| tt_v4pilot60_qwen35_9b_high_recall_minimal_v3_think_guard2_ctx32768_tok8192_b1.partial | qwen3.5:9b | high_recall_minimal_v3 | partial | true | 32768 | 8192 | 1 | 1 | default | 0.0000 | 0.0000 | 0.0000 | 2 | 0 |  |
| tt_qwen35_9b_direct_core_targeted_v7_think_nojson_ctx65536_tok32768_b2_pilot30.partial | qwen3.5:9b | direct_core_targeted_v7 | partial | true | 65536 | 32768 | 2 | 2 | default | 0.0000 | 0.0000 | 0.0000 | 9 | 0 |  |
| tt_v4pilot60_qwen35_9b_social_signal_v9_f1_balanced_think_guard2_ctx32768_tok8192_b1.partial | qwen3.5:9b | social_signal_v9_f1_balanced | partial | true | 32768 | 8192 | 1 | 3 | 0.90 | 1.0000 | 0.1250 | 0.2222 | 8 | 1 |  |
| tt_oldgold_qwen35_9b_direct_core_targeted_v7_think_guard2_ctx32768_tok8192_b1_pilot3 | qwen3.5:9b | direct_core_targeted_v7 | pilot | true | 32768 | 8192 | 1 | 3 | default | 0.6667 | 0.1818 | 0.2857 | 11 | 3 | 2 |
| tt_gemma4_e2b_direct_core_recall_v2_think_ctx16384_tok16384_n237 | gemma4:e2b | direct_core_recall_v2 | large | true | 16384 | 16384 |  | 237 | default | 0.7337 | 0.1821 | 0.2918 | 681 | 169 | 10 |
| tt_gemma4_e4b_direct_brief_thinking_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_brief_thinking | large | true | 16384 | 16384 |  | 237 | default | 0.6280 | 0.2702 | 0.3778 | 681 | 293 | 5 |
| tt_gemma4_e4b_direct_business_migration_fixed_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_business_migration_fixed | large | true | 16384 | 16384 |  | 237 | default | 0.7299 | 0.2937 | 0.4188 | 681 | 274 | 2 |
| tt_gemma4_26b_direct_core_targeted_v7_think_nojson_ctx32768_tok8192_b1_pilot6.partial | gemma4:26b | direct_core_targeted_v7 | partial | true | 32768 | 8192 | 1 | 3 | default | 0.7500 | 0.3000 | 0.4286 | 10 | 4 |  |
| tt_v4pilot60_gemma4_26b_social_signal_v9_f1_balanced_think_guard2_ctx32768_tok8192_b1 | gemma4:26b | social_signal_v9_f1_balanced | pilot | true | 32768 | 8192 | 1 | 60 | default | 0.7067 | 0.3212 | 0.4417 | 165 | 75 | 12 |
| v4_prompt_tf_v4full965_qwen36_35b_iq3_high_recall_minimal_v3_nothink_ctx32768_tok16384_b2_checkpoint60 | qwen3.6:35b-iq3 | high_recall_minimal_v3 | checkpoint | false | 32768 | 16384 | 2 | 60 | default | 0.5289 | 0.4183 | 0.4672 | 153 | 121 | 0 |
| v4_prompt_tf_v4full965_qwen36_35b_iq3_high_recall_minimal_v3_nothink_ctx32768_tok16384_b2_checkpoint237 | qwen3.6:35b-iq3 | high_recall_minimal_v3 | checkpoint | false | 32768 | 16384 | 2 | 237 | default | 0.5728 | 0.4273 | 0.4895 | 681 | 508 | 0 |
| tt_gemma4_e4b_direct_core_filtered_v3_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_filtered_v3 | large | true | 16384 | 16384 |  | 237 | 0.65 | 0.8136 | 0.3524 | 0.4918 | 681 | 295 | 0 |
| tt_gemma4_e4b_direct_core_recall_v2_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_recall_v2 | large | true | 16384 | 16384 |  | 237 | default | 0.7993 | 0.3568 | 0.4934 | 681 | 304 | 2 |
| tf_v4pilot60_qwen35_9b_high_recall_minimal_v3_nothink_ctx32768_tok16384_b2 | qwen3.5:9b | high_recall_minimal_v3 | pilot | false | 32768 | 16384 | 2 | 60 | default | 0.5130 | 0.4788 | 0.4953 | 165 | 154 | 5 |
| tf_v4full965_qwen36_35b_iq3_high_recall_minimal_v3_nothink_ctx32768_tok16384_b2.partial | qwen3.6:35b-iq3 | high_recall_minimal_v3 | partial | false | 32768 | 16384 | 2 | 394 | default | 0.5990 | 0.4287 | 0.4998 | 1171 | 838 |  |
| tt_gemma4_e4b_direct_core_gold_calibrated_v4_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_gold_calibrated_v4 | large | true | 16384 | 16384 |  | 237 | default | 0.7306 | 0.3862 | 0.5053 | 681 | 360 | 0 |
| tf_v4full965_gemma4_26b_high_recall_minimal_v2_nothink_ctx32768_tok16384_b2.partial | gemma4:26b | high_recall_minimal_v2 | partial | false | 32768 | 16384 | 2 | 78 | default | 0.6721 | 0.4100 | 0.5093 | 200 | 122 |  |
| tf_v4pilot60_gemma4_26b_high_recall_minimal_v3_nothink_ctx32768_tok16384_b2 | gemma4:26b | high_recall_minimal_v3 | pilot | false | 32768 | 16384 | 2 | 60 | default | 0.6339 | 0.4303 | 0.5126 | 165 | 112 | 0 |
| tt_gemma4_e4b_gold_mimic_broad_v1_think_ctx16384_tok16384_n237 | gemma4:e4b | gold_mimic_broad_v1 | large | true | 16384 | 16384 |  | 237 | default | 0.6487 | 0.4420 | 0.5258 | 681 | 464 | 3 |
| tt_v4pilot60_gemma4_26b_high_recall_minimal_v3_think_guard2_ctx32768_tok8192_b1 | gemma4:26b | high_recall_minimal_v3 | pilot | true | 32768 | 8192 | 1 | 60 | default | 0.6729 | 0.4364 | 0.5294 | 165 | 107 | 2 |
| v4_prompt_tf_v4full965_qwen36_35b_iq3_social_signal_v5_nothink_ctx32768_tok16384_b2_checkpoint60 | qwen3.6:35b-iq3 | social_signal_v5 | checkpoint | false | 32768 | 16384 | 2 | 60 | default | 0.5556 | 0.5229 | 0.5387 | 153 | 144 | 0 |
| tt_oldgold_gemma4_26b_high_recall_minimal_v2_think_guard2_ctx32768_tok8192_b1_pilot30 | gemma4:26b | high_recall_minimal_v2 | pilot | true | 32768 | 8192 | 1 | 19 | default | 0.6897 | 0.4444 | 0.5405 | 45 | 29 | 2 |
| tf_v4pilot60_qwen35_9b_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2 | qwen3.5:9b | social_signal_v9_f1_balanced | pilot | false | 32768 | 16384 | 2 | 60 | default | 0.6250 | 0.4848 | 0.5461 | 165 | 128 | 1 |
| tt_gemma4_e4b_direct_core_targeted_v7_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_targeted_v7 | large | true | 16384 | 16384 |  | 237 | default | 0.8148 | 0.4200 | 0.5543 | 681 | 351 | 3 |
| tf_v4full965_qwen36_35b_iq3_social_signal_v5_nothink_ctx32768_tok16384_b2.partial | qwen3.6:35b-iq3 | social_signal_v5 | partial | false | 32768 | 16384 | 2 | 240 | default | 0.5942 | 0.5289 | 0.5596 | 692 | 616 |  |
| v4_prompt_tf_v4full965_qwen36_35b_iq3_social_signal_v5_nothink_ctx32768_tok16384_b2_checkpoint237 | qwen3.6:35b-iq3 | social_signal_v5 | checkpoint | false | 32768 | 16384 | 2 | 237 | default | 0.5928 | 0.5301 | 0.5597 | 681 | 609 | 0 |
| tf_v4pilot60_gemma4_26b_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2 | gemma4:26b | social_signal_v9_f1_balanced | pilot | false | 32768 | 16384 | 2 | 60 | default | 0.7000 | 0.4667 | 0.5600 | 165 | 110 | 0 |
| tt_gemma4_e4b_direct_core_balanced_v8_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_balanced_v8 | large | true | 16384 | 16384 |  | 237 | default | 0.7978 | 0.4347 | 0.5627 | 681 | 371 | 2 |
| tt_gemma4_e4b_direct_core_clean_fewshot_v6_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_clean_fewshot_v6 | large | true | 16384 | 16384 |  | 237 | default | 0.7906 | 0.4435 | 0.5682 | 681 | 382 | 2 |
| tf_v4full965_qwen36_35b_iq3_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2.partial | qwen3.6:35b-iq3 | social_signal_v9_f1_balanced | partial | false | 32768 | 16384 | 2 | 242 | 0.70 | 0.7651 | 0.4693 | 0.5818 | 701 | 430 |  |
| v4_prompt_tf_v4full965_qwen36_35b_iq3_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2_checkpoint237 | qwen3.6:35b-iq3 | social_signal_v9_f1_balanced | checkpoint | false | 32768 | 16384 | 2 | 237 | 0.65 | 0.6648 | 0.5213 | 0.5844 | 681 | 534 | 0 |
| tf_gemma4_26b_direct_core_targeted_v7_nothink_ctx32768_tok16384_b2_n237 | gemma4:26b | direct_core_targeted_v7 | large | false | 32768 | 16384 | 2 | 237 | 0.65 | 0.8119 | 0.4626 | 0.5893 | 681 | 388 | 0 |
| tt_gemma4_e4b_direct_core_gold_fewshot_v5_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_gold_fewshot_v5 | large | true | 16384 | 16384 |  | 237 | default | 0.7259 | 0.4978 | 0.5906 | 681 | 467 | 1 |
| tt_gemma4_26b_direct_core_targeted_v7_think_guard_ctx32768_tok4096_b1_pilot6 | gemma4:26b | direct_core_targeted_v7 | pilot | true | 32768 | 4096 | 1 | 6 | default | 1.0000 | 0.4211 | 0.5926 | 19 | 8 | 2 |
| v4_prompt_tf_v4full965_qwen36_35b_iq3_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2_checkpoint60 | qwen3.6:35b-iq3 | social_signal_v9_f1_balanced | checkpoint | false | 32768 | 16384 | 2 | 60 | 0.70 | 0.7700 | 0.5033 | 0.6087 | 153 | 100 | 0 |
| tt_oldgold_gemma4_26b_high_recall_minimal_v1_think_guard2_ctx32768_tok8192_b1_pilot30 | gemma4:26b | high_recall_minimal_v1 | pilot | true | 32768 | 8192 | 1 | 19 | default | 0.6500 | 0.5778 | 0.6118 | 45 | 40 | 1 |
| tf_v4full965_gemma4_26b_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2 | gemma4:26b | social_signal_v9_f1_balanced | full | false | 32768 | 16384 | 2 | 965 | default | 0.7843 | 0.5060 | 0.6151 | 2759 | 1780 | 0 |
| tt_oldgold_gemma4_26b_direct_core_targeted_v7_think_guard2_ctx32768_tok8192_b1_pilot30 | gemma4:26b | direct_core_targeted_v7 | pilot | true | 32768 | 8192 | 1 | 19 | default | 0.8000 | 0.5333 | 0.6400 | 45 | 30 | 0 |
| tt_oldgold_gemma4_26b_direct_core_targeted_v7_think_guard2_ctx32768_tok8192_b1_pilot8 | gemma4:26b | direct_core_targeted_v7 | pilot | true | 32768 | 8192 | 1 | 7 | default | 0.7692 | 0.6250 | 0.6897 | 16 | 13 | 0 |
| tf_v4full965_qwen36_35b_iq4_social_signal_v9_f1_balanced_nothink_ctx16384_tok8192_b4.partial | qwen3.6:35b-iq4 | social_signal_v9_f1_balanced | partial | false | 16384 | 8192 | 4 | 4 | default | 1.0000 | 0.6364 | 0.7778 | 11 | 7 |  |
| tf_v4full965_qwen36_35b_iq4_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2.partial | qwen3.6:35b-iq4 | social_signal_v9_f1_balanced | partial | false | 32768 | 16384 | 2 | 8 | default | 0.8261 | 0.7600 | 0.7917 | 25 | 23 |  |
| tt_qwen35_9b_direct_core_targeted_v7_think_nojson_ctx32768_tok8192_b1_pilot6.partial | qwen3.5:9b | direct_core_targeted_v7 | partial | true | 32768 | 8192 | 1 | 1 | default | 1.0000 | 0.6667 | 0.8000 | 3 | 2 |  |

## V4 Direct Results
| run | model | prompt | status | thinking | ctx | tok | b | n_news | threshold | precision | recall | micro_f1 | gold_pairs | pred_pairs | errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tt_qwen35_9b_direct_core_targeted_v7_think_guard_ctx32768_tok4096_b1_pilot3.partial | qwen3.5:9b | direct_core_targeted_v7 | partial | true | 32768 | 4096 | 1 | 1 | default | 0.0000 | 0.0000 | 0.0000 | 1 | 0 |  |
| tt_v4pilot60_qwen35_9b_high_recall_minimal_v3_think_guard2_ctx32768_tok8192_b1.partial | qwen3.5:9b | high_recall_minimal_v3 | partial | true | 32768 | 8192 | 1 | 1 | default | 0.0000 | 0.0000 | 0.0000 | 1 | 0 |  |
| tt_qwen35_9b_direct_core_targeted_v7_think_nojson_ctx65536_tok32768_b2_pilot30.partial | qwen3.5:9b | direct_core_targeted_v7 | partial | true | 65536 | 32768 | 2 | 2 | default | 0.0000 | 0.0000 | 0.0000 | 1 | 0 |  |
| tt_oldgold_qwen35_9b_direct_core_targeted_v7_think_guard2_ctx32768_tok8192_b1_pilot3 | qwen3.5:9b | direct_core_targeted_v7 | pilot | true | 32768 | 8192 | 1 | 3 | default | 0.3333 | 0.2500 | 0.2857 | 4 | 3 | 2 |
| tf_v4pilot60_gemma4_26b_high_recall_minimal_v3_nothink_ctx32768_tok16384_b2 | gemma4:26b | high_recall_minimal_v3 | pilot | false | 32768 | 16384 | 2 | 60 | default | 0.3214 | 0.4557 | 0.3770 | 79 | 112 | 0 |
| tt_gemma4_e2b_direct_core_recall_v2_think_ctx16384_tok16384_n237 | gemma4:e2b | direct_core_recall_v2 | large | true | 16384 | 16384 |  | 237 | default | 0.4379 | 0.3348 | 0.3795 | 221 | 169 | 10 |
| tt_gemma4_e4b_direct_brief_thinking_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_brief_thinking | large | true | 16384 | 16384 |  | 237 | 0.85 | 0.4078 | 0.3801 | 0.3934 | 221 | 206 | 5 |
| tt_v4pilot60_qwen35_9b_social_signal_v9_f1_balanced_think_guard2_ctx32768_tok8192_b1.partial | qwen3.5:9b | social_signal_v9_f1_balanced | partial | true | 32768 | 8192 | 1 | 3 | 0.90 | 1.0000 | 0.2500 | 0.4000 | 4 | 1 |  |
| tt_gemma4_26b_direct_core_targeted_v7_think_nojson_ctx32768_tok8192_b1_pilot6.partial | gemma4:26b | direct_core_targeted_v7 | partial | true | 32768 | 8192 | 1 | 3 | 0.70 | 0.3333 | 0.5000 | 0.4000 | 2 | 3 |  |
| tf_v4pilot60_qwen35_9b_high_recall_minimal_v3_nothink_ctx32768_tok16384_b2 | qwen3.5:9b | high_recall_minimal_v3 | pilot | false | 32768 | 16384 | 2 | 60 | 0.35 | 0.5814 | 0.3165 | 0.4098 | 79 | 43 | 5 |
| tt_v4pilot60_gemma4_26b_high_recall_minimal_v3_think_guard2_ctx32768_tok8192_b1 | gemma4:26b | high_recall_minimal_v3 | pilot | true | 32768 | 8192 | 1 | 60 | 0.35 | 0.5510 | 0.3418 | 0.4219 | 79 | 49 | 2 |
| tt_oldgold_gemma4_26b_high_recall_minimal_v2_think_guard2_ctx32768_tok8192_b1_pilot30 | gemma4:26b | high_recall_minimal_v2 | pilot | true | 32768 | 8192 | 1 | 19 | 0.35 | 0.5000 | 0.4000 | 0.4444 | 15 | 12 | 2 |
| tf_v4full965_qwen36_35b_iq3_high_recall_minimal_v3_nothink_ctx32768_tok16384_b2.partial | qwen3.6:35b-iq3 | high_recall_minimal_v3 | partial | false | 32768 | 16384 | 2 | 394 | 0.35 | 0.4660 | 0.4684 | 0.4672 | 395 | 397 |  |
| tf_v4pilot60_qwen35_9b_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2 | qwen3.5:9b | social_signal_v9_f1_balanced | pilot | false | 32768 | 16384 | 2 | 60 | 0.90 | 0.7000 | 0.3544 | 0.4706 | 79 | 40 | 1 |
| tt_gemma4_e4b_gold_mimic_broad_v1_think_ctx16384_tok16384_n237 | gemma4:e4b | gold_mimic_broad_v1 | large | true | 16384 | 16384 |  | 237 | 0.90 | 0.4975 | 0.4480 | 0.4714 | 221 | 199 | 3 |
| tt_v4pilot60_gemma4_26b_social_signal_v9_f1_balanced_think_guard2_ctx32768_tok8192_b1 | gemma4:26b | social_signal_v9_f1_balanced | pilot | true | 32768 | 8192 | 1 | 60 | 0.70 | 0.6078 | 0.3924 | 0.4769 | 79 | 51 | 12 |
| tt_gemma4_e4b_direct_business_migration_fixed_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_business_migration_fixed | large | true | 16384 | 16384 |  | 237 | 0.75 | 0.4777 | 0.4842 | 0.4809 | 221 | 224 | 2 |
| tf_v4full965_qwen36_35b_iq3_social_signal_v5_nothink_ctx32768_tok16384_b2.partial | qwen3.6:35b-iq3 | social_signal_v5 | partial | false | 32768 | 16384 | 2 | 240 | 0.35 | 0.4286 | 0.5973 | 0.4991 | 226 | 315 |  |
| tt_gemma4_26b_direct_core_targeted_v7_think_guard_ctx32768_tok4096_b1_pilot6 | gemma4:26b | direct_core_targeted_v7 | pilot | true | 32768 | 4096 | 1 | 6 | 0.90 | 0.6667 | 0.4000 | 0.5000 | 5 | 3 | 2 |
| tf_v4full965_gemma4_26b_high_recall_minimal_v2_nothink_ctx32768_tok16384_b2.partial | gemma4:26b | high_recall_minimal_v2 | partial | false | 32768 | 16384 | 2 | 78 | 0.35 | 0.5161 | 0.4923 | 0.5039 | 65 | 62 |  |
| tt_gemma4_e4b_direct_core_gold_calibrated_v4_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_gold_calibrated_v4 | large | true | 16384 | 16384 |  | 237 | 0.80 | 0.5150 | 0.5430 | 0.5286 | 221 | 233 | 0 |
| tt_oldgold_gemma4_26b_high_recall_minimal_v1_think_guard2_ctx32768_tok8192_b1_pilot30 | gemma4:26b | high_recall_minimal_v1 | pilot | true | 32768 | 8192 | 1 | 19 | 0.35 | 0.5333 | 0.5333 | 0.5333 | 15 | 15 | 1 |
| tt_gemma4_e4b_direct_core_clean_fewshot_v6_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_clean_fewshot_v6 | large | true | 16384 | 16384 |  | 237 | 0.80 | 0.4980 | 0.5747 | 0.5336 | 221 | 255 | 2 |
| tf_v4pilot60_gemma4_26b_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2 | gemma4:26b | social_signal_v9_f1_balanced | pilot | false | 32768 | 16384 | 2 | 60 | 0.70 | 0.5375 | 0.5443 | 0.5409 | 79 | 80 | 0 |
| tt_gemma4_e4b_direct_core_balanced_v8_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_balanced_v8 | large | true | 16384 | 16384 |  | 237 | 0.80 | 0.4719 | 0.6471 | 0.5458 | 221 | 303 | 2 |
| tt_gemma4_e4b_direct_core_filtered_v3_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_filtered_v3 | large | true | 16384 | 16384 |  | 237 | 0.80 | 0.5487 | 0.5611 | 0.5548 | 221 | 226 | 0 |
| tt_gemma4_e4b_direct_core_gold_fewshot_v5_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_gold_fewshot_v5 | large | true | 16384 | 16384 |  | 237 | 0.90 | 0.6230 | 0.5158 | 0.5644 | 221 | 183 | 1 |
| tf_v4full965_qwen36_35b_iq3_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2.partial | qwen3.6:35b-iq3 | social_signal_v9_f1_balanced | partial | false | 32768 | 16384 | 2 | 242 | 0.80 | 0.5747 | 0.5570 | 0.5657 | 228 | 221 |  |
| tt_gemma4_e4b_direct_core_recall_v2_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_recall_v2 | large | true | 16384 | 16384 |  | 237 | 0.85 | 0.5733 | 0.5837 | 0.5785 | 221 | 225 | 2 |
| tt_gemma4_e4b_direct_core_targeted_v7_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_targeted_v7 | large | true | 16384 | 16384 |  | 237 | 0.85 | 0.5752 | 0.5882 | 0.5817 | 221 | 226 | 3 |
| tf_gemma4_26b_direct_core_targeted_v7_nothink_ctx32768_tok16384_b2_n237 | gemma4:26b | direct_core_targeted_v7 | large | false | 32768 | 16384 | 2 | 237 | 0.80 | 0.5675 | 0.6471 | 0.6047 | 221 | 252 | 0 |
| tf_v4full965_gemma4_26b_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2 | gemma4:26b | social_signal_v9_f1_balanced | full | false | 32768 | 16384 | 2 | 965 | 0.70 | 0.5802 | 0.6461 | 0.6114 | 1153 | 1284 | 0 |
| tt_oldgold_gemma4_26b_direct_core_targeted_v7_think_guard2_ctx32768_tok8192_b1_pilot8 | gemma4:26b | direct_core_targeted_v7 | pilot | true | 32768 | 8192 | 1 | 7 | 0.75 | 0.5714 | 0.8000 | 0.6667 | 5 | 7 | 0 |
| tt_qwen35_9b_direct_core_targeted_v7_think_nojson_ctx32768_tok8192_b1_pilot6.partial | qwen3.5:9b | direct_core_targeted_v7 | partial | true | 32768 | 8192 | 1 | 1 | default | 0.5000 | 1.0000 | 0.6667 | 1 | 2 |  |
| tt_oldgold_gemma4_26b_direct_core_targeted_v7_think_guard2_ctx32768_tok8192_b1_pilot30 | gemma4:26b | direct_core_targeted_v7 | pilot | true | 32768 | 8192 | 1 | 19 | 0.80 | 0.6471 | 0.7333 | 0.6875 | 15 | 17 | 0 |
| tf_v4full965_qwen36_35b_iq4_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2.partial | qwen3.6:35b-iq4 | social_signal_v9_f1_balanced | partial | false | 32768 | 16384 | 2 | 8 | 0.80 | 0.6364 | 0.7778 | 0.7000 | 9 | 11 |  |
| tf_v4full965_qwen36_35b_iq4_social_signal_v9_f1_balanced_nothink_ctx16384_tok8192_b4.partial | qwen3.6:35b-iq4 | social_signal_v9_f1_balanced | partial | false | 16384 | 8192 | 4 | 4 | 0.85 | 1.0000 | 0.6667 | 0.8000 | 3 | 2 |  |

## V4 Context Results
| run | model | prompt | status | thinking | ctx | tok | b | n_news | threshold | precision | recall | micro_f1 | gold_pairs | pred_pairs | errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tt_qwen35_9b_direct_core_targeted_v7_think_guard_ctx32768_tok4096_b1_pilot3.partial | qwen3.5:9b | direct_core_targeted_v7 | partial | true | 32768 | 4096 | 1 | 1 | default | 0.0000 | 0.0000 | 0.0000 | 1 | 0 |  |
| tt_v4pilot60_qwen35_9b_high_recall_minimal_v3_think_guard2_ctx32768_tok8192_b1.partial | qwen3.5:9b | high_recall_minimal_v3 | partial | true | 32768 | 8192 | 1 | 1 | default | 0.0000 | 0.0000 | 0.0000 | 1 | 0 |  |
| tt_qwen35_9b_direct_core_targeted_v7_think_nojson_ctx65536_tok32768_b2_pilot30.partial | qwen3.5:9b | direct_core_targeted_v7 | partial | true | 65536 | 32768 | 2 | 2 | default | 0.0000 | 0.0000 | 0.0000 | 1 | 0 |  |
| tt_v4pilot60_qwen35_9b_social_signal_v9_f1_balanced_think_guard2_ctx32768_tok8192_b1.partial | qwen3.5:9b | social_signal_v9_f1_balanced | partial | true | 32768 | 8192 | 1 | 3 | default | 0.0000 | 0.0000 | 0.0000 | 2 | 2 |  |
| tt_gemma4_e2b_direct_core_recall_v2_think_ctx16384_tok16384_n237 | gemma4:e2b | direct_core_recall_v2 | large | true | 16384 | 16384 |  | 237 | 0.65 | 0.1728 | 0.1004 | 0.1270 | 279 | 162 | 10 |
| tt_v4pilot60_gemma4_26b_social_signal_v9_f1_balanced_think_guard2_ctx32768_tok8192_b1 | gemma4:26b | social_signal_v9_f1_balanced | pilot | true | 32768 | 8192 | 1 | 60 | 0.65 | 0.1667 | 0.2264 | 0.1920 | 53 | 72 | 12 |
| tt_gemma4_e4b_direct_brief_thinking_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_brief_thinking | large | true | 16384 | 16384 |  | 237 | default | 0.1877 | 0.1971 | 0.1923 | 279 | 293 | 5 |
| tt_gemma4_e4b_direct_business_migration_fixed_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_business_migration_fixed | large | true | 16384 | 16384 |  | 237 | default | 0.2153 | 0.2115 | 0.2134 | 279 | 274 | 2 |
| tf_v4pilot60_qwen35_9b_high_recall_minimal_v3_nothink_ctx32768_tok16384_b2 | qwen3.5:9b | high_recall_minimal_v3 | pilot | false | 32768 | 16384 | 2 | 60 | default | 0.1494 | 0.4340 | 0.2222 | 53 | 154 | 5 |
| tt_gemma4_e4b_direct_core_recall_v2_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_recall_v2 | large | true | 16384 | 16384 |  | 237 | default | 0.2204 | 0.2401 | 0.2298 | 279 | 304 | 2 |
| tf_v4pilot60_gemma4_26b_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2 | gemma4:26b | social_signal_v9_f1_balanced | pilot | false | 32768 | 16384 | 2 | 60 | 0.65 | 0.1782 | 0.3396 | 0.2338 | 53 | 101 | 0 |
| tt_gemma4_e4b_direct_core_filtered_v3_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_filtered_v3 | large | true | 16384 | 16384 |  | 237 | 0.65 | 0.2305 | 0.2437 | 0.2369 | 279 | 295 | 0 |
| tf_v4full965_gemma4_26b_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2 | gemma4:26b | social_signal_v9_f1_balanced | full | false | 32768 | 16384 | 2 | 965 | default | 0.1860 | 0.3271 | 0.2371 | 1012 | 1780 | 0 |
| tt_gemma4_e4b_direct_core_targeted_v7_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_targeted_v7 | large | true | 16384 | 16384 |  | 237 | default | 0.2222 | 0.2796 | 0.2476 | 279 | 351 | 3 |
| tf_gemma4_26b_direct_core_targeted_v7_nothink_ctx32768_tok16384_b2_n237 | gemma4:26b | direct_core_targeted_v7 | large | false | 32768 | 16384 | 2 | 237 | 0.65 | 0.2139 | 0.2975 | 0.2489 | 279 | 388 | 0 |
| tt_gemma4_e4b_direct_core_gold_calibrated_v4_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_gold_calibrated_v4 | large | true | 16384 | 16384 |  | 237 | 0.70 | 0.2257 | 0.2832 | 0.2512 | 279 | 350 | 0 |
| tf_v4full965_qwen36_35b_iq3_high_recall_minimal_v3_nothink_ctx32768_tok16384_b2.partial | qwen3.6:35b-iq3 | high_recall_minimal_v3 | partial | false | 32768 | 16384 | 2 | 394 | default | 0.2017 | 0.3596 | 0.2584 | 470 | 838 |  |
| tf_v4pilot60_qwen35_9b_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2 | qwen3.5:9b | social_signal_v9_f1_balanced | pilot | false | 32768 | 16384 | 2 | 60 | 0.65 | 0.1840 | 0.4340 | 0.2584 | 53 | 125 | 1 |
| tf_v4full965_gemma4_26b_high_recall_minimal_v2_nothink_ctx32768_tok16384_b2.partial | gemma4:26b | high_recall_minimal_v2 | partial | false | 32768 | 16384 | 2 | 78 | default | 0.2131 | 0.3611 | 0.2680 | 72 | 122 |  |
| tt_oldgold_gemma4_26b_high_recall_minimal_v2_think_guard2_ctx32768_tok8192_b1_pilot30 | gemma4:26b | high_recall_minimal_v2 | pilot | true | 32768 | 8192 | 1 | 19 | default | 0.2069 | 0.4286 | 0.2791 | 14 | 29 | 2 |
| tt_gemma4_e4b_gold_mimic_broad_v1_think_ctx16384_tok16384_n237 | gemma4:e4b | gold_mimic_broad_v1 | large | true | 16384 | 16384 |  | 237 | default | 0.2241 | 0.3728 | 0.2799 | 279 | 464 | 3 |
| tf_v4full965_qwen36_35b_iq3_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2.partial | qwen3.6:35b-iq3 | social_signal_v9_f1_balanced | partial | false | 32768 | 16384 | 2 | 242 | default | 0.2111 | 0.4155 | 0.2800 | 284 | 559 |  |
| tt_gemma4_e4b_direct_core_balanced_v8_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_balanced_v8 | large | true | 16384 | 16384 |  | 237 | default | 0.2480 | 0.3297 | 0.2831 | 279 | 371 | 2 |
| tt_gemma4_26b_direct_core_targeted_v7_think_guard_ctx32768_tok4096_b1_pilot6 | gemma4:26b | direct_core_targeted_v7 | pilot | true | 32768 | 4096 | 1 | 6 | 0.70 | 0.2857 | 0.2857 | 0.2857 | 7 | 7 | 2 |
| tf_v4full965_qwen36_35b_iq3_social_signal_v5_nothink_ctx32768_tok16384_b2.partial | qwen3.6:35b-iq3 | social_signal_v5 | partial | false | 32768 | 16384 | 2 | 240 | default | 0.2110 | 0.4577 | 0.2889 | 284 | 616 |  |
| tt_oldgold_gemma4_26b_high_recall_minimal_v1_think_guard2_ctx32768_tok8192_b1_pilot30 | gemma4:26b | high_recall_minimal_v1 | pilot | true | 32768 | 8192 | 1 | 19 | default | 0.2000 | 0.5714 | 0.2963 | 14 | 40 | 1 |
| tt_gemma4_e4b_direct_core_clean_fewshot_v6_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_clean_fewshot_v6 | large | true | 16384 | 16384 |  | 237 | default | 0.2565 | 0.3513 | 0.2965 | 279 | 382 | 2 |
| tt_v4pilot60_gemma4_26b_high_recall_minimal_v3_think_guard2_ctx32768_tok8192_b1 | gemma4:26b | high_recall_minimal_v3 | pilot | true | 32768 | 8192 | 1 | 60 | default | 0.2243 | 0.4528 | 0.3000 | 53 | 107 | 2 |
| tf_v4pilot60_gemma4_26b_high_recall_minimal_v3_nothink_ctx32768_tok16384_b2 | gemma4:26b | high_recall_minimal_v3 | pilot | false | 32768 | 16384 | 2 | 60 | default | 0.2321 | 0.4906 | 0.3152 | 53 | 112 | 0 |
| tt_gemma4_e4b_direct_core_gold_fewshot_v5_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_gold_fewshot_v5 | large | true | 16384 | 16384 |  | 237 | 0.75 | 0.2614 | 0.4122 | 0.3199 | 279 | 440 | 1 |
| tt_oldgold_gemma4_26b_direct_core_targeted_v7_think_guard2_ctx32768_tok8192_b1_pilot30 | gemma4:26b | direct_core_targeted_v7 | pilot | true | 32768 | 8192 | 1 | 19 | 0.65 | 0.2500 | 0.5000 | 0.3333 | 14 | 28 | 0 |
| tf_v4full965_qwen36_35b_iq4_social_signal_v9_f1_balanced_nothink_ctx16384_tok8192_b4.partial | qwen3.6:35b-iq4 | social_signal_v9_f1_balanced | partial | false | 16384 | 8192 | 4 | 4 | 0.70 | 0.2000 | 1.0000 | 0.3333 | 1 | 5 |  |
| tt_oldgold_qwen35_9b_direct_core_targeted_v7_think_guard2_ctx32768_tok8192_b1_pilot3 | qwen3.5:9b | direct_core_targeted_v7 | pilot | true | 32768 | 8192 | 1 | 3 | 0.90 | 1.0000 | 0.2500 | 0.4000 | 4 | 1 | 2 |
| tt_gemma4_26b_direct_core_targeted_v7_think_nojson_ctx32768_tok8192_b1_pilot6.partial | gemma4:26b | direct_core_targeted_v7 | partial | true | 32768 | 8192 | 1 | 3 | default | 0.2500 | 1.0000 | 0.4000 | 1 | 4 |  |
| tf_v4full965_qwen36_35b_iq4_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2.partial | qwen3.6:35b-iq4 | social_signal_v9_f1_balanced | partial | false | 32768 | 16384 | 2 | 8 | 0.70 | 0.2857 | 0.8571 | 0.4286 | 7 | 21 |  |
| tt_oldgold_gemma4_26b_direct_core_targeted_v7_think_guard2_ctx32768_tok8192_b1_pilot8 | gemma4:26b | direct_core_targeted_v7 | pilot | true | 32768 | 8192 | 1 | 7 | 0.65 | 0.3636 | 0.6667 | 0.4706 | 6 | 11 | 0 |
| tt_qwen35_9b_direct_core_targeted_v7_think_nojson_ctx32768_tok8192_b1_pilot6.partial | qwen3.5:9b | direct_core_targeted_v7 | partial | true | 32768 | 8192 | 1 | 1 | default | 0.5000 | 1.0000 | 0.6667 | 1 | 2 |  |

## V4 Weak Results
| run | model | prompt | status | thinking | ctx | tok | b | n_news | threshold | precision | recall | micro_f1 | gold_pairs | pred_pairs | errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tt_qwen35_9b_direct_core_targeted_v7_think_nojson_ctx32768_tok8192_b1_pilot6.partial | qwen3.5:9b | direct_core_targeted_v7 | partial | true | 32768 | 8192 | 1 | 1 | default | 0.0000 | 0.0000 | 0.0000 | 1 | 2 |  |
| tt_qwen35_9b_direct_core_targeted_v7_think_guard_ctx32768_tok4096_b1_pilot3.partial | qwen3.5:9b | direct_core_targeted_v7 | partial | true | 32768 | 4096 | 1 | 1 | default | 0.0000 | 0.0000 | 0.0000 | 1 | 0 |  |
| tt_v4pilot60_qwen35_9b_high_recall_minimal_v3_think_guard2_ctx32768_tok8192_b1.partial | qwen3.5:9b | high_recall_minimal_v3 | partial | true | 32768 | 8192 | 1 | 1 | default | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |  |
| tt_qwen35_9b_direct_core_targeted_v7_think_nojson_ctx65536_tok32768_b2_pilot30.partial | qwen3.5:9b | direct_core_targeted_v7 | partial | true | 65536 | 32768 | 2 | 2 | default | 0.0000 | 0.0000 | 0.0000 | 7 | 0 |  |
| tt_oldgold_qwen35_9b_direct_core_targeted_v7_think_guard2_ctx32768_tok8192_b1_pilot3 | qwen3.5:9b | direct_core_targeted_v7 | pilot | true | 32768 | 8192 | 1 | 3 | default | 0.0000 | 0.0000 | 0.0000 | 3 | 3 | 2 |
| tt_v4pilot60_qwen35_9b_social_signal_v9_f1_balanced_think_guard2_ctx32768_tok8192_b1.partial | qwen3.5:9b | social_signal_v9_f1_balanced | partial | true | 32768 | 8192 | 1 | 3 | default | 0.0000 | 0.0000 | 0.0000 | 2 | 2 |  |
| tf_v4pilot60_gemma4_26b_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2 | gemma4:26b | social_signal_v9_f1_balanced | pilot | false | 32768 | 16384 | 2 | 60 | default | 0.0727 | 0.2424 | 0.1119 | 33 | 110 | 0 |
| tt_gemma4_e4b_direct_business_migration_fixed_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_business_migration_fixed | large | true | 16384 | 16384 |  | 237 | 0.85 | 0.1127 | 0.1326 | 0.1218 | 181 | 213 | 2 |
| tf_v4pilot60_gemma4_26b_high_recall_minimal_v3_nothink_ctx32768_tok16384_b2 | gemma4:26b | high_recall_minimal_v3 | pilot | false | 32768 | 16384 | 2 | 60 | default | 0.0804 | 0.2727 | 0.1241 | 33 | 112 | 0 |
| tt_gemma4_e2b_direct_core_recall_v2_think_ctx16384_tok16384_n237 | gemma4:e2b | direct_core_recall_v2 | large | true | 16384 | 16384 |  | 237 | default | 0.1302 | 0.1215 | 0.1257 | 181 | 169 | 10 |
| tt_v4pilot60_gemma4_26b_social_signal_v9_f1_balanced_think_guard2_ctx32768_tok8192_b1 | gemma4:26b | social_signal_v9_f1_balanced | pilot | true | 32768 | 8192 | 1 | 60 | default | 0.0933 | 0.2121 | 0.1296 | 33 | 75 | 12 |
| tt_gemma4_e4b_direct_core_recall_v2_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_recall_v2 | large | true | 16384 | 16384 |  | 237 | 0.65 | 0.1107 | 0.1823 | 0.1378 | 181 | 298 | 2 |
| tt_gemma4_e4b_direct_brief_thinking_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_brief_thinking | large | true | 16384 | 16384 |  | 237 | default | 0.1126 | 0.1823 | 0.1392 | 181 | 293 | 5 |
| tt_v4pilot60_gemma4_26b_high_recall_minimal_v3_think_guard2_ctx32768_tok8192_b1 | gemma4:26b | high_recall_minimal_v3 | pilot | true | 32768 | 8192 | 1 | 60 | default | 0.0935 | 0.3030 | 0.1429 | 33 | 107 | 2 |
| tt_gemma4_e4b_direct_core_filtered_v3_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_filtered_v3 | large | true | 16384 | 16384 |  | 237 | 0.65 | 0.1186 | 0.1934 | 0.1471 | 181 | 295 | 0 |
| tf_v4pilot60_qwen35_9b_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2 | qwen3.5:9b | social_signal_v9_f1_balanced | pilot | false | 32768 | 16384 | 2 | 60 | 0.70 | 0.0982 | 0.3333 | 0.1517 | 33 | 112 | 1 |
| tf_v4full965_qwen36_35b_iq3_high_recall_minimal_v3_nothink_ctx32768_tok16384_b2.partial | qwen3.6:35b-iq3 | high_recall_minimal_v3 | partial | false | 32768 | 16384 | 2 | 394 | default | 0.1098 | 0.3007 | 0.1608 | 306 | 838 |  |
| tt_gemma4_e4b_direct_core_gold_calibrated_v4_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_gold_calibrated_v4 | large | true | 16384 | 16384 |  | 237 | default | 0.1278 | 0.2541 | 0.1701 | 181 | 360 | 0 |
| tf_v4full965_gemma4_26b_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2 | gemma4:26b | social_signal_v9_f1_balanced | full | false | 32768 | 16384 | 2 | 965 | default | 0.1140 | 0.3418 | 0.1710 | 594 | 1780 | 0 |
| tf_v4pilot60_qwen35_9b_high_recall_minimal_v3_nothink_ctx32768_tok16384_b2 | qwen3.5:9b | high_recall_minimal_v3 | pilot | false | 32768 | 16384 | 2 | 60 | default | 0.1039 | 0.4848 | 0.1711 | 33 | 154 | 5 |
| tt_gemma4_e4b_direct_core_gold_fewshot_v5_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_gold_fewshot_v5 | large | true | 16384 | 16384 |  | 237 | default | 0.1199 | 0.3094 | 0.1728 | 181 | 467 | 1 |
| tf_v4full965_gemma4_26b_high_recall_minimal_v2_nothink_ctx32768_tok16384_b2.partial | gemma4:26b | high_recall_minimal_v2 | partial | false | 32768 | 16384 | 2 | 78 | default | 0.1393 | 0.2698 | 0.1838 | 63 | 122 |  |
| tt_gemma4_e4b_gold_mimic_broad_v1_think_ctx16384_tok16384_n237 | gemma4:e4b | gold_mimic_broad_v1 | large | true | 16384 | 16384 |  | 237 | 0.75 | 0.1345 | 0.3315 | 0.1914 | 181 | 446 | 3 |
| tt_gemma4_e4b_direct_core_targeted_v7_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_targeted_v7 | large | true | 16384 | 16384 |  | 237 | default | 0.1481 | 0.2873 | 0.1955 | 181 | 351 | 3 |
| tf_v4full965_qwen36_35b_iq3_social_signal_v5_nothink_ctx32768_tok16384_b2.partial | qwen3.6:35b-iq3 | social_signal_v5 | partial | false | 32768 | 16384 | 2 | 240 | default | 0.1266 | 0.4286 | 0.1955 | 182 | 616 |  |
| tt_gemma4_e4b_direct_core_balanced_v8_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_balanced_v8 | large | true | 16384 | 16384 |  | 237 | default | 0.1482 | 0.3039 | 0.1993 | 181 | 371 | 2 |
| tt_gemma4_e4b_direct_core_clean_fewshot_v6_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_clean_fewshot_v6 | large | true | 16384 | 16384 |  | 237 | 0.65 | 0.1474 | 0.3094 | 0.1996 | 181 | 380 | 2 |
| tf_v4full965_qwen36_35b_iq3_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2.partial | qwen3.6:35b-iq3 | social_signal_v9_f1_balanced | partial | false | 32768 | 16384 | 2 | 242 | 0.70 | 0.1465 | 0.3333 | 0.2036 | 189 | 430 |  |
| tf_gemma4_26b_direct_core_targeted_v7_nothink_ctx32768_tok16384_b2_n237 | gemma4:26b | direct_core_targeted_v7 | large | false | 32768 | 16384 | 2 | 237 | 0.65 | 0.1598 | 0.3425 | 0.2179 | 181 | 388 | 0 |
| tt_oldgold_gemma4_26b_high_recall_minimal_v2_think_guard2_ctx32768_tok8192_b1_pilot30 | gemma4:26b | high_recall_minimal_v2 | pilot | true | 32768 | 8192 | 1 | 19 | default | 0.1724 | 0.3125 | 0.2222 | 16 | 29 | 2 |
| tt_oldgold_gemma4_26b_direct_core_targeted_v7_think_guard2_ctx32768_tok8192_b1_pilot8 | gemma4:26b | direct_core_targeted_v7 | pilot | true | 32768 | 8192 | 1 | 7 | default | 0.1538 | 0.4000 | 0.2222 | 5 | 13 | 0 |
| tt_gemma4_26b_direct_core_targeted_v7_think_nojson_ctx32768_tok8192_b1_pilot6.partial | gemma4:26b | direct_core_targeted_v7 | partial | true | 32768 | 8192 | 1 | 3 | 0.95 | 1.0000 | 0.1429 | 0.2500 | 7 | 1 |  |
| tt_oldgold_gemma4_26b_direct_core_targeted_v7_think_guard2_ctx32768_tok8192_b1_pilot30 | gemma4:26b | direct_core_targeted_v7 | pilot | true | 32768 | 8192 | 1 | 19 | default | 0.2000 | 0.3750 | 0.2609 | 16 | 30 | 0 |
| tt_oldgold_gemma4_26b_high_recall_minimal_v1_think_guard2_ctx32768_tok8192_b1_pilot30 | gemma4:26b | high_recall_minimal_v1 | pilot | true | 32768 | 8192 | 1 | 19 | default | 0.2000 | 0.5000 | 0.2857 | 16 | 40 | 1 |
| tf_v4full965_qwen36_35b_iq4_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2.partial | qwen3.6:35b-iq4 | social_signal_v9_f1_balanced | partial | false | 32768 | 16384 | 2 | 8 | default | 0.2174 | 0.5556 | 0.3125 | 9 | 23 |  |
| tf_v4full965_qwen36_35b_iq4_social_signal_v9_f1_balanced_nothink_ctx16384_tok8192_b4.partial | qwen3.6:35b-iq4 | social_signal_v9_f1_balanced | partial | false | 16384 | 8192 | 4 | 4 | default | 0.4286 | 0.4286 | 0.4286 | 7 | 7 |  |
| tt_gemma4_26b_direct_core_targeted_v7_think_guard_ctx32768_tok4096_b1_pilot6 | gemma4:26b | direct_core_targeted_v7 | pilot | true | 32768 | 4096 | 1 | 6 | default | 0.5000 | 0.5714 | 0.5333 | 7 | 8 | 2 |

## Old Gold All Results
| run | model | prompt | status | thinking | ctx | tok | b | n_news | threshold | precision | recall | micro_f1 | gold_pairs | pred_pairs | errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tt_gemma4_26b_ctx32k_tok8k_b1_search237_think.partial | gemma4:26b | unknown | partial | true | 32768 | 8192 | 1 | 1 | default | 0.0000 | 0.0000 | 0.0000 | 1 | 0 | 36 |
| tt_qwen35_9b_direct_core_targeted_v7_think_guard_ctx32768_tok4096_b1_pilot3.partial | qwen3.5:9b | direct_core_targeted_v7 | partial | true | 32768 | 4096 | 1 | 1 | default | 0.0000 | 0.0000 | 0.0000 | 1 | 0 | 36 |
| tt_v4pilot60_qwen35_9b_high_recall_minimal_v3_think_guard2_ctx32768_tok8192_b1.partial | qwen3.5:9b | high_recall_minimal_v3 | partial | true | 32768 | 8192 | 1 | 1 | default | 0.0000 | 0.0000 | 0.0000 | 1 | 0 | 36 |
| tt_qwen35_9b_direct_core_targeted_v7_think_nojson_ctx65536_tok32768_b2_pilot30.partial | qwen3.5:9b | direct_core_targeted_v7 | partial | true | 65536 | 32768 | 2 | 2 | default | 0.0000 | 0.0000 | 0.0000 | 1 | 0 | 72 |
| tt_gemma4_26b_direct_core_targeted_v7_think_nojson_ctx32768_tok8192_b1_pilot6.partial | gemma4:26b | direct_core_targeted_v7 | partial | true | 32768 | 8192 | 1 | 3 | default | 0.0000 | 0.0000 | 0.0000 | 1 | 4 | 36 |
| tf_gemma4_26b_b1_search237.partial | gemma4:26b | unknown | partial | false |  |  | 1 | 15 | default | 0.0000 | 0.0000 | 0.0000 | 8 | 0 | 540 |
| thinking_recall_experiment_latent_candidate_v2_think_true | unknown | unknown | large | true |  |  |  | 300 | 0.35 | 0.4062 | 0.0423 | 0.0767 | 307 | 32 | 0 |
| social_signal_v2_think_false | unknown | unknown | pilot | false |  |  |  | 18 | default | 0.2222 | 0.1667 | 0.1905 | 12 | 9 | 396 |
| social_signal_v2_tf_b1_pilot18 | unknown | unknown | pilot | false |  |  | 1 | 18 | default | 0.2222 | 0.1667 | 0.1905 | 12 | 9 | 396 |
| tf_qwen36_35b_iq3_compact_b1_pilot30.partial | qwen3.6:35b-iq3 | unknown | partial | false |  |  | 1 | 4 | 0.40 | 0.1667 | 0.2500 | 0.2000 | 4 | 6 | 0 |
| social_signal_v5_tf_v5_b1_pilot18 | unknown | social_signal_v5 | pilot | false |  |  | 1 | 18 | default | 0.1667 | 0.2500 | 0.2000 | 12 | 18 | 0 |
| social_signal_v1_think_false | unknown | unknown | pilot | false |  |  |  | 18 | default | 0.2308 | 0.2500 | 0.2400 | 12 | 13 | 252 |
| social_signal_v1_tf_b1_pilot18 | unknown | unknown | pilot | false |  |  | 1 | 18 | default | 0.2308 | 0.2500 | 0.2400 | 12 | 13 | 252 |
| social_signal_v5_tt_oldgold_gemma4_26b_ctx32768_tok8192_b1_pilot3_social_signal_v5_think_nojson_guard2 | gemma4:26b | social_signal_v5 | pilot | true | 32768 | 8192 | 1 | 3 | 0.35 | 0.2000 | 0.3333 | 0.2500 | 3 | 5 | 0 |
| search_production_v1_gemma4-e2b_5df03950348d_b6a2d2f27b85 | gemma4:e2b | production_v1 | large |  |  |  | 6 | 300 | 0.05 | 0.4037 | 0.2117 | 0.2778 | 307 | 161 | 0 |
| search_hard_negative_v2_gemma4-e2b_5df03950348d_1a35bcd28344 | gemma4:e2b | unknown | large |  |  |  |  | 300 | 0.25 | 0.7105 | 0.1759 | 0.2820 | 307 | 76 | 0 |
| llm_predictions_gemma4-e2b_social-risk-classification-v1_5df03950348d_033f6fd8763b | gemma4:e2b | social-risk-classification-v1 | full |  |  |  |  | 1000 | 0.05 | 0.3878 | 0.2230 | 0.2831 | 915 | 526 | 0 |
| full_production_v1_gemma4-e2b_5df03950348d_b6a2d2f27b85 | gemma4:e2b | production_v1 | full |  |  |  | 6 | 1000 | 0.05 | 0.3878 | 0.2230 | 0.2831 | 915 | 526 | 0 |
| tt_oldgold_qwen35_9b_direct_core_targeted_v7_think_guard2_ctx32768_tok8192_b1_pilot3 | qwen3.5:9b | direct_core_targeted_v7 | pilot | true | 32768 | 8192 | 1 | 3 | default | 0.3333 | 0.3333 | 0.3333 | 3 | 3 | 72 |
| tt_gemma4_26b_direct_core_targeted_v7_think_guard_ctx32768_tok4096_b1_pilot6 | gemma4:26b | direct_core_targeted_v7 | pilot | true | 32768 | 4096 | 1 | 6 | 0.90 | 0.3333 | 0.3333 | 0.3333 | 3 | 3 | 72 |
| grouped_event_v1_search_think_true | unknown | unknown | large | true |  |  |  | 300 | 0.35 | 0.4848 | 0.2606 | 0.3390 | 307 | 165 | 0 |
| tf_compact_b1_ctx16k_text6k_pilot18 | unknown | unknown | pilot | false | 16384 |  | 1 | 18 | 0.50 | 0.3125 | 0.4167 | 0.3571 | 12 | 16 | 0 |
| tf_compact_b1_ctx16k_search237.partial | unknown | unknown | partial | false | 16384 |  | 1 | 21 | 0.40 | 0.3000 | 0.4615 | 0.3636 | 13 | 20 | 0 |
| social_signal_v3_tf_v3_b1_pilot18 | unknown | social_signal_v3 | pilot | false |  |  | 1 | 18 | 0.35 | 0.5000 | 0.3333 | 0.4000 | 12 | 8 | 72 |
| tf_compact_b1_ctx16k_pilot18 | unknown | unknown | pilot | false | 16384 |  | 1 | 18 | 0.50 | 0.3529 | 0.5000 | 0.4138 | 12 | 17 | 0 |
| search_primary_first_v2_gemma4-e2b_5df03950348d_5f153b0981f2 | gemma4:e2b | unknown | large |  |  |  |  | 300 | 0.05 | 0.4913 | 0.3681 | 0.4209 | 307 | 230 | 0 |
| social_signal_v5_tt_oldgold_gemma4_26b_ctx32768_tok8192_b1_pilot30_social_signal_v5_think_nojson_guard2.partial | gemma4:26b | social_signal_v5 | partial | true | 32768 | 8192 | 1 | 10 | 0.45 | 0.3636 | 0.5714 | 0.4444 | 7 | 11 | 36 |
| search_few_shot_major_v3_gemma4-e2b_5df03950348d_0c3d39008fbb | gemma4:e2b | unknown | large |  |  |  |  | 300 | 0.05 | 0.4833 | 0.4235 | 0.4514 | 307 | 269 | 0 |
| thinking_recall_experiment_recall_max_v1_think_false | unknown | unknown | large | false |  |  |  | 300 | 0.05 | 0.5039 | 0.4202 | 0.4583 | 307 | 256 | 0 |
| social_signal_v4_tf_v4_b1_search237 | unknown | unknown | large | false |  |  | 1 | 237 | 0.35 | 0.5312 | 0.4048 | 0.4595 | 252 | 192 | 576 |
| tf_compact_b1_search237 | unknown | unknown | large | false |  |  | 1 | 237 | 0.55 | 0.4596 | 0.4960 | 0.4771 | 252 | 272 | 0 |
| thinking_recall_experiment_recall_max_v1_think_true | unknown | unknown | large | true |  |  |  | 300 | 0.05 | 0.5296 | 0.4365 | 0.4786 | 307 | 253 | 0 |
| thinking_recall_experiment_latent_candidate_v3_think_true | unknown | unknown | large | true |  |  |  | 300 | 0.45 | 0.4131 | 0.5733 | 0.4802 | 307 | 426 | 0 |
| tt_gemma4_e4b_direct_business_migration_fixed_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_business_migration_fixed | large | true | 16384 | 16384 |  | 237 | 0.75 | 0.4866 | 0.4844 | 0.4855 | 225 | 224 | 72 |
| search_balanced_recall_v2_gemma4-e2b_5df03950348d_e98b1a3c5abc | gemma4:e2b | balanced_recall_v2 | large |  |  |  |  | 300 | 0.05 | 0.5593 | 0.4300 | 0.4862 | 307 | 236 | 0 |
| social_signal_v3_tt_v3_b1_search237_think | unknown | social_signal_v3 | large | true |  |  | 1 | 237 | 0.35 | 0.6644 | 0.3849 | 0.4874 | 252 | 146 | 0 |
| best_balanced_recall_v2 | unknown | balanced_recall_v2 | full |  |  |  |  | 1000 | 0.15 | 0.5300 | 0.4536 | 0.4888 | 915 | 783 | 0 |
| full_balanced_recall_v2_gemma4-e2b_5df03950348d_e98b1a3c5abc | gemma4:e2b | balanced_recall_v2 | full |  |  |  |  | 1000 | 0.15 | 0.5300 | 0.4536 | 0.4888 | 915 | 783 | 0 |
| social_signal_v5_tf_v5_b1_search237 | unknown | social_signal_v5 | large | false |  |  | 1 | 237 | 0.35 | 0.5433 | 0.4484 | 0.4913 | 252 | 208 | 144 |
| tt_gemma4_e4b_direct_brief_thinking_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_brief_thinking | large | true | 16384 | 16384 |  | 237 | 0.85 | 0.5146 | 0.4711 | 0.4919 | 225 | 206 | 180 |
| social_signal_v3_tf_v3_b1_search237 | unknown | social_signal_v3 | large | false |  |  | 1 | 237 | 0.35 | 0.6831 | 0.3849 | 0.4924 | 252 | 142 | 288 |
| tt_gemma4_e2b_direct_core_recall_v2_think_ctx16384_tok16384_n237 | gemma4:e2b | direct_core_recall_v2 | large | true | 16384 | 16384 |  | 237 | default | 0.5740 | 0.4311 | 0.4924 | 225 | 169 | 360 |
| tt_v4pilot60_qwen35_9b_social_signal_v9_f1_balanced_think_guard2_ctx32768_tok8192_b1.partial | qwen3.5:9b | social_signal_v9_f1_balanced | partial | true | 32768 | 8192 | 1 | 3 | 0.90 | 1.0000 | 0.3333 | 0.5000 | 3 | 1 | 72 |
| tt_oldgold_gemma4_26b_high_recall_minimal_v1_think_guard2_ctx32768_tok8192_b1_pilot30 | gemma4:26b | high_recall_minimal_v1 | pilot | true | 32768 | 8192 | 1 | 30 | 0.35 | 0.4800 | 0.5455 | 0.5106 | 22 | 25 | 36 |
| tt_gemma4_e4b_gold_mimic_broad_v1_think_ctx16384_tok16384_n237 | gemma4:e4b | gold_mimic_broad_v1 | large | true | 16384 | 16384 |  | 237 | 0.90 | 0.5930 | 0.5244 | 0.5566 | 225 | 199 | 108 |
| tt_v4pilot60_gemma4_26b_social_signal_v9_f1_balanced_think_guard2_ctx32768_tok8192_b1 | gemma4:26b | social_signal_v9_f1_balanced | pilot | true | 32768 | 8192 | 1 | 41 | 0.75 | 0.6333 | 0.5135 | 0.5672 | 37 | 30 | 216 |
| tt_gemma4_e4b_direct_core_balanced_v8_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_balanced_v8 | large | true | 16384 | 16384 |  | 237 | 0.80 | 0.5017 | 0.6756 | 0.5758 | 225 | 303 | 72 |
| social_signal_v5_tf_qwen35_9b_v5_b1_search237 | qwen3.5:9b | social_signal_v5 | large | false |  |  | 1 | 237 | 0.65 | 0.6143 | 0.5437 | 0.5768 | 252 | 223 | 0 |
| tt_gemma4_e4b_direct_core_clean_fewshot_v6_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_clean_fewshot_v6 | large | true | 16384 | 16384 |  | 237 | 0.80 | 0.5451 | 0.6178 | 0.5792 | 225 | 255 | 72 |
| tt_oldgold_gemma4_26b_direct_core_targeted_v7_think_guard2_ctx32768_tok8192_b1_pilot30 | gemma4:26b | direct_core_targeted_v7 | pilot | true | 32768 | 8192 | 1 | 30 | 0.70 | 0.4722 | 0.7727 | 0.5862 | 22 | 36 | 0 |
| tt_gemma4_e4b_direct_core_gold_calibrated_v4_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_gold_calibrated_v4 | large | true | 16384 | 16384 |  | 237 | 0.80 | 0.5794 | 0.6000 | 0.5895 | 225 | 233 | 0 |
| v4_prompt_tf_v4full965_qwen36_35b_iq3_social_signal_v5_nothink_ctx32768_tok16384_b2_checkpoint237 | qwen3.6:35b-iq3 | social_signal_v5 | checkpoint | false | 32768 | 16384 | 2 | 237 | 0.35 | 0.5113 | 0.7067 | 0.5933 | 225 | 311 | 0 |
| tf_v4pilot60_qwen35_9b_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2 | qwen3.5:9b | social_signal_v9_f1_balanced | pilot | false | 32768 | 16384 | 2 | 41 | 0.90 | 0.7037 | 0.5135 | 0.5938 | 37 | 27 | 0 |
| tf_v4full965_qwen36_35b_iq3_social_signal_v5_nothink_ctx32768_tok16384_b2.partial | qwen3.6:35b-iq3 | social_signal_v5 | partial | false | 32768 | 16384 | 2 | 240 | 0.35 | 0.5111 | 0.7093 | 0.5941 | 227 | 315 | 0 |
| tt_gemma4_e4b_direct_core_filtered_v3_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_filtered_v3 | large | true | 16384 | 16384 |  | 237 | 0.85 | 0.6839 | 0.5289 | 0.5965 | 225 | 174 | 0 |
| v4_prompt_tf_v4full965_qwen36_35b_iq3_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2_checkpoint237 | qwen3.6:35b-iq3 | social_signal_v9_f1_balanced | checkpoint | false | 32768 | 16384 | 2 | 237 | 0.80 | 0.6121 | 0.5822 | 0.5968 | 225 | 214 | 0 |
| tf_v4full965_qwen36_35b_iq3_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2.partial | qwen3.6:35b-iq3 | social_signal_v9_f1_balanced | partial | false | 32768 | 16384 | 2 | 242 | 0.80 | 0.6063 | 0.5877 | 0.5969 | 228 | 221 | 0 |
| tt_gemma4_e4b_direct_core_gold_fewshot_v5_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_gold_fewshot_v5 | large | true | 16384 | 16384 |  | 237 | 0.90 | 0.6721 | 0.5467 | 0.6029 | 225 | 183 | 36 |
| v4_prompt_tf_v4full965_qwen36_35b_iq3_high_recall_minimal_v3_nothink_ctx32768_tok16384_b2_checkpoint237 | qwen3.6:35b-iq3 | high_recall_minimal_v3 | checkpoint | false | 32768 | 16384 | 2 | 237 | 0.35 | 0.5854 | 0.6400 | 0.6115 | 225 | 246 | 0 |
| tf_v4full965_qwen36_35b_iq3_high_recall_minimal_v3_nothink_ctx32768_tok16384_b2.partial | qwen3.6:35b-iq3 | high_recall_minimal_v3 | partial | false | 32768 | 16384 | 2 | 394 | 0.35 | 0.5919 | 0.6369 | 0.6136 | 369 | 397 | 0 |
| tf_b1_pilot18 | unknown | unknown | pilot | false |  |  | 1 | 18 | 0.75 | 0.5714 | 0.6667 | 0.6154 | 12 | 14 | 0 |
| v4_prompt_tf_v4full965_qwen36_35b_iq3_social_signal_v5_nothink_ctx32768_tok16384_b2_checkpoint60 | qwen3.6:35b-iq3 | social_signal_v5 | checkpoint | false | 32768 | 16384 | 2 | 60 | 0.35 | 0.5143 | 0.7826 | 0.6207 | 46 | 70 | 0 |
| tt_oldgold_gemma4_26b_direct_core_targeted_v7_think_guard2_ctx32768_tok8192_b1_pilot8 | gemma4:26b | direct_core_targeted_v7 | pilot | true | 32768 | 8192 | 1 | 8 | 0.70 | 0.5000 | 0.8333 | 0.6250 | 6 | 10 | 0 |
| tt_gemma4_e4b_direct_core_recall_v2_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_recall_v2 | large | true | 16384 | 16384 |  | 237 | 0.85 | 0.6311 | 0.6311 | 0.6311 | 225 | 225 | 72 |
| tt_gemma4_e4b_direct_core_targeted_v7_think_ctx16384_tok16384_n237 | gemma4:e4b | direct_core_targeted_v7 | large | true | 16384 | 16384 |  | 237 | 0.90 | 0.7283 | 0.5600 | 0.6332 | 225 | 173 | 108 |
| social_signal_v5_tf_gemma4_e4b_v5_b1_search237 | gemma4:e4b | social_signal_v5 | large | false |  |  | 1 | 237 | 0.75 | 0.6125 | 0.6587 | 0.6348 | 252 | 271 | 0 |
| tf_gemma4_26b_direct_core_targeted_v7_nothink_ctx32768_tok16384_b2_n237 | gemma4:26b | direct_core_targeted_v7 | large | false | 32768 | 16384 | 2 | 237 | 0.85 | 0.6192 | 0.6578 | 0.6379 | 225 | 239 | 0 |
| tf_gemma4_26b_ctx32k_tok8k_b1_search237 | gemma4:26b | unknown | large | false | 32768 | 8192 | 1 | 237 | 0.50 | 0.5939 | 0.6905 | 0.6385 | 252 | 293 | 0 |
| tf_v4full965_gemma4_26b_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2 | gemma4:26b | social_signal_v9_f1_balanced | large | false | 32768 | 16384 | 2 | 706 | 0.75 | 0.6179 | 0.6903 | 0.6521 | 691 | 772 | 0 |
| social_signal_v5_tf_qwen36_35b_iq3_ctx16k_v5_b1_search237_full | qwen3.6:35b-iq3 | social_signal_v5 | large | false | 16384 |  | 1 | 237 | 0.35 | 0.5774 | 0.7698 | 0.6599 | 252 | 336 | 0 |
| v4_prompt_tf_v4full965_qwen36_35b_iq3_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2_checkpoint60 | qwen3.6:35b-iq3 | social_signal_v9_f1_balanced | checkpoint | false | 32768 | 16384 | 2 | 60 | 0.80 | 0.6111 | 0.7174 | 0.6600 | 46 | 54 | 0 |
| v4_prompt_tf_v4full965_qwen36_35b_iq3_high_recall_minimal_v3_nothink_ctx32768_tok16384_b2_checkpoint60 | qwen3.6:35b-iq3 | high_recall_minimal_v3 | checkpoint | false | 32768 | 16384 | 2 | 60 | 0.35 | 0.5965 | 0.7391 | 0.6602 | 46 | 57 | 0 |
| tf_v4full965_qwen36_35b_iq4_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2.partial | qwen3.6:35b-iq4 | social_signal_v9_f1_balanced | partial | false | 32768 | 16384 | 2 | 8 | 0.90 | 0.7500 | 0.6000 | 0.6667 | 5 | 4 | 0 |
| tt_oldgold_gemma4_26b_high_recall_minimal_v2_think_guard2_ctx32768_tok8192_b1_pilot30 | gemma4:26b | high_recall_minimal_v2 | pilot | true | 32768 | 8192 | 1 | 30 | 0.35 | 0.6522 | 0.6818 | 0.6667 | 22 | 23 | 72 |
| social_signal_v3_tf_gemma4_26b_ctx32k_tok8k_b1_search237.partial | gemma4:26b | social_signal_v3 | partial | false | 32768 | 8192 | 1 | 18 | 0.35 | 0.6000 | 0.7500 | 0.6667 | 12 | 15 | 0 |
| tt_qwen35_9b_direct_core_targeted_v7_think_nojson_ctx32768_tok8192_b1_pilot6.partial | qwen3.5:9b | direct_core_targeted_v7 | partial | true | 32768 | 8192 | 1 | 1 | default | 0.5000 | 1.0000 | 0.6667 | 1 | 2 | 0 |
| tf_v4full965_gemma4_26b_high_recall_minimal_v2_nothink_ctx32768_tok16384_b2.partial | gemma4:26b | high_recall_minimal_v2 | partial | false | 32768 | 16384 | 2 | 78 | 0.35 | 0.6935 | 0.7167 | 0.7049 | 60 | 62 | 0 |
| tf_v4pilot60_gemma4_26b_social_signal_v9_f1_balanced_nothink_ctx32768_tok16384_b2 | gemma4:26b | social_signal_v9_f1_balanced | pilot | false | 32768 | 16384 | 2 | 41 | 0.85 | 0.8214 | 0.6216 | 0.7077 | 37 | 28 | 0 |
| tt_v4pilot60_gemma4_26b_high_recall_minimal_v3_think_guard2_ctx32768_tok8192_b1 | gemma4:26b | high_recall_minimal_v3 | pilot | true | 32768 | 8192 | 1 | 41 | 0.35 | 0.7222 | 0.7027 | 0.7123 | 37 | 36 | 72 |
| tf_v4pilot60_gemma4_26b_high_recall_minimal_v3_nothink_ctx32768_tok16384_b2 | gemma4:26b | high_recall_minimal_v3 | pilot | false | 32768 | 16384 | 2 | 41 | 0.35 | 0.7647 | 0.7027 | 0.7324 | 37 | 34 | 0 |
| social_signal_v5_tf_qwen36_35b_iq3_v5_b1_pilot30 | qwen3.6:35b-iq3 | social_signal_v5 | pilot | false |  |  | 1 | 30 | 0.35 | 0.6000 | 0.9545 | 0.7368 | 22 | 35 | 0 |
| tf_v4pilot60_qwen35_9b_high_recall_minimal_v3_nothink_ctx32768_tok16384_b2 | qwen3.5:9b | high_recall_minimal_v3 | pilot | false | 32768 | 16384 | 2 | 41 | 0.35 | 0.8125 | 0.7027 | 0.7536 | 37 | 32 | 0 |
| social_signal_v5_tf_qwen36_35b_iq3_ctx16k_v5_b1_pilot30 | qwen3.6:35b-iq3 | social_signal_v5 | pilot | false | 16384 |  | 1 | 30 | 0.35 | 0.6364 | 0.9545 | 0.7636 | 22 | 33 | 0 |
| tf_v4full965_qwen36_35b_iq4_social_signal_v9_f1_balanced_nothink_ctx16384_tok8192_b4.partial | qwen3.6:35b-iq4 | social_signal_v9_f1_balanced | partial | false | 16384 | 8192 | 4 | 4 | 0.90 | 1.0000 | 1.0000 | 1.0000 | 1 | 1 | 0 |

## Column Standard
- `status`: `full` is about 900+ rows, `large` is 200+ rows, `pilot` is smaller, `partial` is a stopped/in-progress full run, and `checkpoint` is an intermediate measurement from a run checkpoint.
- `threshold` is the best threshold for that row on the corresponding benchmark/scope.
- `ctx`, `tok`, and `b` are parsed from filename tokens when available.
- Tables are sorted from worst to best by `micro_f1`; use `n_news` and `status` to judge whether a result is a pilot, partial, large run, or full run.
- Full per-threshold rows remain in `analysis_outputs/standardized_llm_metrics_all_thresholds.csv`; this markdown keeps every run/result once at its best threshold per meaningful benchmark/scope.
