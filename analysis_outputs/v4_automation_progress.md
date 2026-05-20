# v4 direct prompt search progress

## 2026-05-20 04:03 MSK

- Stopped `qwen3.5:9b + direct_strict + think=true + ctx=16384 + max_tokens=16384` after diagnostics showed it was not a useful bounded run: 14 raw records, 5 ok, 9 `LLM returned an empty response` errors.
- Verified the apparent mojibake was only PowerShell rendering; the normalized dataset and raw JSON files are valid UTF-8 when read with Python.
- Added `prompts/v4_direct_search/direct_brief_thinking.md`, a shorter direct-only prompt that tells the model to keep thinking brief and emit JSON immediately.
- Updated `scripts/evaluate_v4_strength_metrics.py` so `direct_brief_thinking` outputs are included in v4 strength summaries.
- Started background run `qwen35_9b_direct_brief_thinking_think_ctx16384_tok16384_n237`; wrapper will run `scripts/evaluate_v4_strength_metrics.py` after the experiment finishes.

## 2026-05-20 04:31 MSK

- Stopped `qwen3.5:9b + direct_brief_thinking + think=true + ctx=16384 + max_tokens=16384` after it repeated the same failure mode: 13 raw records, 4 ok, 9 empty-response errors, about 25 minutes elapsed.
- Moved aborted qwen `direct_strict` and `direct_brief_thinking` partial/raw files to `.aborted` filenames so `scripts/evaluate_v4_strength_metrics.py` will not count them as candidate metrics.
- Started `gemma4:e4b + direct_brief_thinking + think=true + ctx=16384 + max_tokens=16384` on the same 237 eval rows. Initial health check: first 4 rows completed in 15.7s with 0 errors.

## 2026-05-20 05:02 MSK

- Completed `gemma4:e4b + direct_brief_thinking + think=true + ctx=16384 + max_tokens=16384` on 237 rows: 232 ok, 5 errors.
- Best direct-only result: threshold 0.85, precision 0.408, recall 0.380, micro-F1 0.393. Best direct_context micro-F1 0.382; all micro-F1 0.378; context 0.192; weak 0.139.
- Error analysis at direct threshold 0.85 shows two major prompt failures: `enterprises_count` recall 0.098 despite 51 direct gold examples, and false migration labels from foreign trade/sanctions/business context (`international_inflow/outflow`).
- Added `prompts/v4_direct_search/direct_business_migration_fixed.md`: stronger enterprise rule, strict physical-people-only migration rule, title-is-evidence rule.
- Updated `scripts/evaluate_v4_strength_metrics.py` to include `direct_business_migration_fixed` outputs.
- Started `gemma4:e4b + direct_business_migration_fixed + think=true + ctx=16384 + max_tokens=16384` on 237 rows. Initial health check: 11/237 rows, 11 ok, 0 errors.

## 2026-05-20 05:31 MSK

- Completed `gemma4:e4b + direct_business_migration_fixed + think=true + ctx=16384 + max_tokens=16384` on 237 rows: 235 ok, 2 errors.
- Best direct-only result improved from 0.393 to 0.481 micro-F1 at threshold 0.75: precision 0.478, recall 0.484, mean predicted labels 0.945/news.
- Scope summary for this prompt: direct_context best 0.447, all best 0.419, context best 0.213. Weak is not improved because this is still a direct-only prompt.
- The prompt fix worked for `enterprises_count`: direct recall rose to 0.804 at threshold 0.75, but precision is 0.451. The new largest direct misses are now `crime_count` (recall 0.275), `industrial_production_index` (0.477), and `consumer_price_index` (0.444).
- Started `batiai/qwen3.6-35b:iq3 + direct_business_migration_fixed + think=true + ctx=16384 + max_tokens=16384` on the same 237 rows to test whether a larger model can use the corrected taxonomy better.

## 2026-05-20 06:00 MSK

- `batiai/qwen3.6-35b:iq3 + direct_business_migration_fixed + think=true` is still running and is technically healthy, but slow.
- Current progress: 26/237 rows, 26 ok, 0 errors. No metrics yet because the run is incomplete.
- The model is using ctx=16384/max_tokens=16384 and is taking roughly 45-90 seconds per news item, so this is a multi-hour run. Keeping it alive because it is the first stable 35B thinking candidate on the corrected taxonomy.

## 2026-05-20 06:31 MSK

- Stopped the `batiai/qwen3.6-35b:iq3 + direct_business_migration_fixed + think=true` full run after 52/237 rows because it was too slow and the interim result did not justify blocking further search.
- On the same first 52 ids, 35B partial direct-F1 was 0.487 versus 0.489 for completed `gemma4:e4b + direct_business_migration_fixed`; 35B was also worse on `direct_context` and `all`.
- Moved the 35B partial/raw artifacts to `.aborted` so the evaluator will not treat them as completed metrics.
- Added `prompts/v4_direct_search/direct_core_recall_v2.md`, preserving the successful enterprise/migration fixes and adding stronger direct rules for `crime_count`, `industrial_production_index`, and `consumer_price_index`.
- Updated `scripts/evaluate_v4_strength_metrics.py` to include `direct_core_recall_v2` outputs.
- Started `gemma4:e4b + direct_core_recall_v2 + think=true + ctx=16384 + max_tokens=16384`. Initial health check: 6/237 rows, 6 ok, 0 errors.

## 2026-05-20 07:02 MSK

- Completed `gemma4:e4b + direct_core_recall_v2 + think=true`: 235 ok, 2 errors.
- New best direct-only metric: threshold 0.85, precision 0.573, recall 0.584, micro-F1 0.578. This is a real improvement over 0.481.
- Other scopes for the same run: direct_context best 0.522, all best 0.493, context best 0.230. Weak remains low because the prompt is still direct/core focused.
- Factor-level at threshold 0.85: `crime_count` F1 0.743, `industrial_production_index` 0.609, `enterprises_count` 0.595, `consumer_price_index` 0.542, `mortality_rate` 0.609.
- Manual inspection shows a gold-label mismatch: several `direct` gold labels are broad v3-style analytical signals, not literal direct signals. Examples: `consumer_price_index` on a monument/freedom/entrepreneurship news, `mortality_rate` on stock-market and historical/genocide/legal-memory news, and `crime_count` on market-session delay and broad legal/security context.
- Added `prompts/v4_direct_search/gold_mimic_broad_v1.md` to explicitly mimic the current gold logic rather than strict semantic directness.
- Updated `scripts/evaluate_v4_strength_metrics.py` to include `gold_mimic_broad_v1` outputs.
- Started `gemma4:e4b + gold_mimic_broad_v1 + think=true + ctx=16384 + max_tokens=16384`. Initial health check: 7/237 rows, 7 ok, 0 errors.

## 2026-05-20 07:32 MSK

- Completed `gemma4:e4b + gold_mimic_broad_v1 + think=true`: 234 ok, 3 errors.
- The broad gold-mimic prompt improved broad scopes but not direct: all micro-F1 0.526, direct_context 0.498, context 0.280, weak 0.191, but best direct only 0.471. Direct leader remains `direct_core_recall_v2` at 0.578.
- Quick ensemble check across `direct_core_recall_v2`, `direct_business_migration_fixed`, and `gold_mimic_broad_v1` did not beat the direct core prompt. Best score-level ensemble was below the single best direct run.
- Diagnostic post-filter on `direct_core_recall_v2` suggests a factor allowlist / low-support suppression can raise direct micro-F1 from 0.578 to roughly 0.60-0.61 on this eval set by removing false positives from rare factors.
- Added `prompts/v4_direct_search/direct_core_filtered_v3.md`, a prompt that keeps the successful core rules and explicitly suppresses noisy rare factors unless literally stated.
- Updated `scripts/evaluate_v4_strength_metrics.py` to include `direct_core_filtered_v3` outputs.
- Started `gemma4:e4b + direct_core_filtered_v3 + think=true + ctx=16384 + max_tokens=16384`. Initial health check: 10/237 raw records, 10 ok, 0 errors.

## 2026-05-20 08:01 MSK

- Completed `gemma4:e4b + direct_core_filtered_v3 + think=true`: 237 ok, 0 errors.
- The low-support suppression prompt did not help direct metrics. Best direct was 0.555 at threshold 0.80, below `direct_core_recall_v2` at 0.578. It likely removed some useful recall together with false positives.
- Current leaders remain: direct = `direct_core_recall_v2` threshold 0.85, F1 0.578; direct_context = `direct_core_recall_v2`, F1 0.522; all/context/weak = `gold_mimic_broad_v1`, all F1 0.526, context 0.280, weak 0.191.
- Added `prompts/v4_direct_search/direct_core_gold_calibrated_v4.md`: it keeps the core prompt but adds narrow gold-style calibration for broad direct labels without switching fully to broad mode.
- Updated `scripts/evaluate_v4_strength_metrics.py` to include `direct_core_gold_calibrated_v4` outputs.
- Started `gemma4:e4b + direct_core_gold_calibrated_v4 + think=true + ctx=16384 + max_tokens=16384`. Initial health check: 9/237 records, 9 ok, 0 errors.

## 2026-05-20 08:32 MSK

- Completed `gemma4:e4b + direct_core_gold_calibrated_v4 + think=true`: 237 ok, 0 errors.
- This did not beat the leader. Best direct was 0.529 at threshold 0.80. It improved all to 0.505, but over-broadened direct predictions and reduced precision/recall balance.
- Current leaders after 6 completed runs: direct = `direct_core_recall_v2` threshold 0.85, F1 0.578; direct_context = `direct_core_recall_v2`, F1 0.522; all/context/weak = `gold_mimic_broad_v1`, all F1 0.526, context 0.280, weak 0.191.
- Added `prompts/v4_direct_search/direct_core_gold_fewshot_v5.md`: same core direct logic, with a small set of explicit gold-style few-shot examples for the most unintuitive labels.
- Updated `scripts/evaluate_v4_strength_metrics.py` to include `direct_core_gold_fewshot_v5` outputs.
- Started `gemma4:e4b + direct_core_gold_fewshot_v5 + think=true + ctx=16384 + max_tokens=16384`. Initial health check: 7/237 records, 7 ok, 0 errors.

## 2026-05-20 09:00 MSK

- Completed `gemma4:e4b + direct_core_gold_fewshot_v5 + think=true`: 236 ok, 1 error.
- It improved broad/social-signal scopes, but did not beat the direct-only leader. Best direct was 0.564 micro-F1 at threshold 0.90: precision 0.623, recall 0.516, labels/news 0.77.
- Scope leaders for this run: all 0.591, direct_context 0.591, context 0.320, weak 0.173. This is now the best broad/direct_context/context prompt, while weak still remains best on `gold_mimic_broad_v1` at 0.191.
- Current direct-only leader remains `gemma4:e4b + direct_core_recall_v2` at threshold 0.85: precision 0.573, recall 0.584, micro-F1 0.578.
- Started `qwen3.5:9b + direct_core_recall_v2 + think=true + ctx=16384 + max_tokens=16384` because qwen is still the main candidate that could beat Gemma if the thinking JSON failure mode does not repeat on the shorter best prompt.

## 2026-05-20 09:30 MSK

- Stopped `qwen3.5:9b + direct_core_recall_v2 + think=true` after 13 rows because the previous failure mode repeated: 10/13 rows were empty responses, each failed row took roughly 2-3 minutes, and the model was still consuming the full 16k output budget on thinking without reliable final JSON.
- Moved the qwen partial/raw artifacts to `.aborted`, so they will not contaminate completed metrics.
- Tried `gemma3n:e4b + direct_core_recall_v2 + think=true`; Ollama returned HTTP 400 for every row, so this tag is not a meaningful model-quality experiment. Moved those artifacts to `.aborted` and regenerated `v4_strength_metrics.csv`.
- Started `gemma4:e2b + direct_core_recall_v2 + think=true + ctx=16384 + max_tokens=16384` as the next bounded model comparison. Initial health: 10/237 rows, 9 ok, 1 error.

## 2026-05-20 10:00 MSK

- Completed `gemma4:e2b + direct_core_recall_v2 + think=true`: 227 ok, 10 errors. It is much weaker than `gemma4:e4b`; best direct micro-F1 was only 0.292 at default/low thresholds, with precision 0.734 and recall 0.182. The model is too sparse and misses most gold direct labels.
- Found a concrete prompt-quality issue: `direct_core_gold_fewshot_v5` had its few-shot Russian examples stored as mojibake. That run still improved broad scopes, but the calibration text itself was corrupted.
- Added `prompts/v4_direct_search/direct_core_clean_fewshot_v6.md`: a clean UTF-8 targeted few-shot prompt focused on the main direct misses (`consumer_price_index`, `industrial_production_index`, `enterprises_count`, `crime_count`, `mortality_rate`) while keeping strict migration rules.
- Updated `scripts/evaluate_v4_strength_metrics.py` to include `direct_core_clean_fewshot_v6`.
- Started `gemma4:e4b + direct_core_clean_fewshot_v6 + think=true + ctx=16384 + max_tokens=16384`. Initial health: 7/237 rows, 7 ok, 0 errors.

## 2026-05-20 10:30 MSK

- Completed `gemma4:e4b + direct_core_clean_fewshot_v6 + think=true`: 235 ok, 2 errors.
- Clean UTF-8 examples improved broad/weak behavior but did not improve direct. Best direct was 0.534 at threshold 0.80: precision 0.498, recall 0.575. It is too wide for direct-only evaluation.
- Best scopes for v6: all 0.568, direct_context 0.558, context 0.297, weak 0.200. So it is useful for social-signal recall, but not for the direct leaderboard.
- Current direct leader remains `gemma4:e4b + direct_core_recall_v2` at threshold 0.85: precision 0.573, recall 0.584, micro-F1 0.578.
- Added `prompts/v4_direct_search/direct_core_targeted_v7.md`: keeps v2-style directness, adds clean targeted examples for actual direct misses, and adds negative examples to suppress migration/CPI/mortality/hospital false positives.
- Updated `scripts/evaluate_v4_strength_metrics.py` to include `direct_core_targeted_v7`.
- Started `gemma4:e4b + direct_core_targeted_v7 + think=true + ctx=16384 + max_tokens=16384`. Initial health: 8/237 rows, 8 ok, 0 errors.

## 2026-05-20 11:00 MSK

- Completed `gemma4:e4b + direct_core_targeted_v7 + think=true`: 234 ok, 3 errors.
- New direct-only leader: threshold 0.85, precision 0.575, recall 0.588, micro-F1 0.582. This is only a small improvement over `direct_core_recall_v2` at 0.578, but macro-F1 improved more clearly from 0.359 to 0.410.
- Scope summary for v7: direct_context 0.550, all 0.554, context 0.248, weak 0.195. For broad all/direct_context, `direct_core_gold_fewshot_v5` still leads at about 0.591.
- Factor comparison against v2: v7 improved `consumer_price_index` F1 0.542 -> 0.613, `hospitals` 0.444 -> 0.615, `life_expectancy` 0.286 -> 0.500, but regressed `enterprises_count` 0.595 -> 0.521 and `mortality_rate` 0.609 -> 0.538.
- Added `prompts/v4_direct_search/direct_core_balanced_v8.md`: preserves v7's price/medicine gains, restores stronger enterprise rules, and narrows over-broad industrial/mortality predictions.
- Updated `scripts/evaluate_v4_strength_metrics.py` to include `direct_core_balanced_v8`.
- Started `gemma4:e4b + direct_core_balanced_v8 + think=true + ctx=16384 + max_tokens=16384`. Initial health: 10/237 rows, 10 ok, 0 errors.

## 2026-05-20 11:30 MSK

- Completed `gemma4:e4b + direct_core_balanced_v8 + think=true`: 235 ok, 2 errors.
- v8 did not beat the direct leader. Best direct was 0.546 at threshold 0.80: precision 0.472, recall 0.647. It increased recall but lost too much precision.
- Best scopes for v8: all 0.563, direct_context 0.554, context 0.283, weak 0.199. This remains below `direct_core_gold_fewshot_v5` for broad scopes and below `direct_core_targeted_v7` for direct.
- Current direct leader remains `gemma4:e4b + direct_core_targeted_v7` at threshold 0.85: precision 0.575, recall 0.588, micro-F1 0.582.
- Checked `qwen3.6:27b`; it is not installed locally, but the Ollama library has the tag. Started pulling `qwen3.6:27b` and stopped the parallel lower-priority `qwen3.5:27b` pull to avoid bandwidth/disk contention.
- Started watcher `qwen36_27b_direct_core_targeted_v7_think_ctx16384_tok16384_n237`: once `qwen3.6:27b` finishes downloading, it will run the full 237-row experiment with `direct_core_targeted_v7`, `think=true`, `ctx=16384`, `max_tokens=16384`, then regenerate `v4_strength_metrics.csv`.

## 2026-05-20 13:58 MSK

- `qwen3.6:27b` downloaded successfully and the watcher started the full `direct_core_targeted_v7` experiment.
- Current qwen progress: 18/237 rows, 18 ok, 0 errors. It is technically healthy, but extremely slow: about 8,513 seconds for 18 rows.
- Interim qwen direct-only metrics on the first 18 completed ids: at threshold 0.90, precision 0.500, recall 0.750, micro-F1 0.600. On the same ids, `gemma4:e4b + v7` at threshold 0.90 has micro-F1 0.519. This is too small a sample to trust, but it suggests qwen may be worth letting run if time is acceptable.
- Completed-run leaders remain unchanged: direct = `gemma4:e4b + direct_core_targeted_v7` at threshold 0.85, F1 0.582; all/direct_context/context = `gemma4:e4b + direct_core_gold_fewshot_v5`, F1 about 0.591/0.591/0.320; weak = `gemma4:e4b + direct_core_clean_fewshot_v6`, F1 0.200.

## 2026-05-20 14:10 MSK

- Stopped `qwen3.6:27b` at the user's request.
- The run had reached 36/237 rows with 0 errors, but was too slow for an interactive loop. Partial/raw artifacts were renamed with `.stopped`, the stale queue PID was removed, and `v4_strength_metrics.csv` was regenerated so the stopped partial run is not counted as a completed metric candidate.

## 2026-05-20 16:18 MSK

- Started a full high-budget `qwen3.5:9b + direct_core_targeted_v7 + think=true` run at the user's request.
- Config: `ctx=65536`, `max_tokens=65536`, `text_limit=6000`, `batch_size=1`, `timeout=2400`, 237 eval rows.
- Initial health check: first completed row is ok with non-empty final JSON content, so the previous empty-response failure mode is not appearing immediately under the larger output budget.
- Active tag: `qwen35_9b_direct_core_targeted_v7_think_ctx65536_tok65536_txt6000_n237`.

## 2026-05-20 16:42 MSK

- Investigated the `qwen3.5:9b` empty-response row. The current raw log only preserved final `message.content`; it did not preserve `message.thinking`, so the old error only showed `LLM returned an empty response` with empty `raw_text`.
- Updated `analyzer/llm_client.py` to retain the last full Ollama response in `LLMClient.last_response_data`.
- Updated `scripts/run_v4_social_signal_prompt_experiment.py` to write a sibling `.ollama_response.jsonl` file for every request, including full raw response, `content_chars`, `thinking_chars`, `done_reason`, duration counters, prompt/eval token counters, and error metadata.
- Stopped the earlier `qwen35_9b...ctx65536...` run at 4 rows because it was not diagnostic enough without the full response log.
- Started a larger diagnostic/full run: `qwen3.5:9b + direct_core_targeted_v7 + think=true`, `ctx=131072`, `max_tokens=131072`, `text_limit=6000`, `timeout=7200`, tag `qwen35_9b_direct_core_targeted_v7_think_ctx131072_tok131072_txt6000_n237`.
- Initial status: model loaded with context 131072, using about 14GB; first item is still running, so the new `.ollama_response.jsonl` will appear after the first completed or failed response.
