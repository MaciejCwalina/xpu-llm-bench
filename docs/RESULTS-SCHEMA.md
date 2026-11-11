# Results schema

A run writes these files under `--output-dir/<spec-name>/`:

- `manifest.json`: identity, fingerprint, boot evidence, host facts, hold state, and `completed_rung_ids` (the resume cursor).
- `results.jsonl`: one JSON object per measured rung, in schedule order.
- `sessions.json`: every live conversation verbatim, used by `--resume` to re-prefill. Never edit it.
- `greedy_outputs.jsonl`: greedy window outputs (`probe_id`, token ids, text). This is the MTP on/off reference.
- `logprob_scores.jsonl`: teacher-forced probe scores. A dense run's file is the logprob baseline.

`schemas/result.schema.json` is generated from the Pydantic models in
`xpu_llm_bench.results` (`python scripts/gen_schema.py`). `tests/test_schema.py`
fails if it drifts. Every timestamp is recorded twice: UTC and America/New_York,
both with explicit offsets. Definitions of every metric are in
[METHODOLOGY.md](METHODOLOGY.md) §4.

## Manifest (`RunManifest`)

| Field | Meaning |
|---|---|
| `run_id`, `spec_name`, `spec_path`, `profile`, `tool_version` | What produced the run |
| `started_*`, `finished_*`, `status`, `abort_reason` | `in_progress` / `completed` / `aborted` (RAM floor, OOM, SPEC_ASSERT, engine error) |
| `prompt_seed`, `k` | Fixed workload seed; declared k (null if undeclared) |
| `fingerprint` | `build_fingerprint`, `launch_fingerprint`, image id, per-mount hashes, StartedAt, redacted env |
| `boot` | Parsed boot evidence (attention_boot, QSA_WRAP / QSA_MEMORY_PLAN / banner, graphs, capture sizes, KV dtype and tokens, speculative config) |
| `host` | sysfs GPU ids/driver, kernel, oneAPI / level-zero package versions (no xpu-smi) |
| `hold` | bench-hold status before, our acquire, status after, release |
| `corpus_files` | sha256 of each corpus file when `prompts.corpus_dir` is used |
| `first_greedy_divergence` | First probe and token where MTP-on output left the reference |
| `image`, `hardware`, `config`, `attention`, `kv_format`, `graph`, `speculative`, `decode_method` | Spec-declared config, plus the method id `fixed-tokens-chunks-v2` and the formula text |
| `completed_rung_ids`, `output_jsonl`, `notes` | Resume cursor, path, notes |

## Rung row (`RungResult`)

Fixed CSV / wide-report column order (`report.COLUMNS`). Steps/s comes
immediately before tok/s:

```
run_id, rung_id, phase, users, ctx_per_user, ctx_tokens, prompt_type, load_mode, prompt_seed, k, build_fingerprint, attention_path, attention_boot, graphs_on, capture_sizes, decode_batch_tokens, kv_format, steps_s_per_user, decode_tok_s_per_user, steps_s_total, decode_tok_s_total, tokens_per_step, accepted_drafts_per_step, hit_rate, acceptance_rate_vllm, verify_steps, verify_steps_server, tokens_delivered, decode_window_s, prefill_tok_s, ttft_ms, tpot_ms, itl_p50_ms, itl_p95_ms, itl_p99_ms, e2e_ms, queue_time_ms, running_reqs, waiting_reqs, running_max, waiting_max, prefix_cache_hit_rate, kv_cache_usage_pct, kv_cache_usage_max_pct, preemptions, mem_available_min_gib, rss_kib, other_vllm_stacks, gpu_mem, eager_fallback, quality, errors, timeouts, exit_reason, flags, valid, reason
```

The default markdown table (`report.SUMMARY_COLUMNS`) uses:
`rung_id, users, ctx_tokens, attention_path, steps_s_per_user, decode_tok_s_per_user, decode_tok_s_total, tokens_per_step, hit_rate, prefill_tok_s, ttft_ms, itl_p95_ms, kv_cache_usage_max_pct, mem_available_min_gib, quality, flags`.

Fields on the row that are not in the table: `session_seeds`, `temperature`,
`launch_fingerprint`, `image_id`, `boot_started_at`, `verify_steps_stream`,
`draft_tokens`, `accepted_drafts`, `drafts`, `prefill_tokens`,
`attention_evidence`, and timestamps.

- `rung_id` is `{prompt_type}:{load_short}:{users}u:{ctx}:{phase}`, where phase is `S` (session-1 growth), `C` (joiners), or `meas` (grid shape).
- `ctx_tokens` lists the context of every live decoding session. `ctx_per_user` is the rung's nominal target.
- `quality` holds needle, code-edit, format/repetition, greedy reference/divergence, and logprob scores. Unchecked items are null.
- `valid` is `valid`, `warning` (see METHODOLOGY §6), or `invalid`.
- `exit_reason` is set only on failed rungs.

## Flags

| Flag | Meaning |
|---|---|
| `NOT_AS_LABELED` | Log evidence contradicts the spec label (attention path, graphs, k, KV dtype). |
| `QSA_FALLBACK` | QSA requested/wrapped, but runtime lines in the window show the dense/skip path. |
| `MIXED_PATH` | Both QSA and skip/dense runtime lines in the window. |
| `GRAPH_MISS` | Decode batch tokens users*(k+1) not in the logged/declared capture sizes, or graphs off. |
| `SPEC_COUNTER_BROKEN` | T != k*D, A > T, sum(A_p) != A, or A disagrees with delivered - D. |
| `STEP_COUNT_MISMATCH` | Stream-counted verify steps and server drafts differ by > 5%. |
| `DERIVED_STEPS` | Stream gave no per-step token counts; verify steps taken from server drafts. |
| `K_UNDECLARED` | Spec did not declare k; k was taken from draft_tokens/drafts. |
| `SHORT_WINDOW` | Fewer verify steps per session than ladder.min_window_steps. |
| `EOS_IN_WINDOW` | Model emitted EOS inside the window (ignore_eos kept decoding). |
| `PREEMPTED` | Preemptions > 0 during the rung. |
| `CONCURRENT_CLIENT` | More running/waiting requests than this runner's sessions (another client). |
| `FOREIGN_STACK` | Another vLLM stack (different container) is resident on the host. |
| `PROTECTED_PORT` | Server bound to a protected production port (only possible on imported rows). |
| `RAM_FLOOR` | MemAvailable hit the floor; runner guard aborted. |
| `OOM` | Out-of-memory (container OOMKilled or OOM lines in logs). |
| `SPEC_ASSERT` | Known assert: Expected spec_token == num_spec_decodes * (num_speculative_tokens + 1). |
| `ENGINE_ERROR` | Engine error / traceback in logs or HTTP 5xx. |
| `TIMEOUT` | Client timeout. |
| `CLIENT_ERROR` | Request failed without a server-side cause in the logs. |
| `REBOOTED` | Container StartedAt changed mid-run (boot-once violated). |
| `RESUMED_REFILL` | First rung after --resume on a new boot: sessions were re-prefilled. |
| `QUALITY_FAIL` | A quality check failed on this rung (see quality fields). |
| `CONVERTED` | Value converted from a legacy definition (import-legacy). |
| `DRY_RUN` | Placeholder row from --dry-run. |

## Comparison

Rows pair on `(users, ctx_per_user, prompt_type, load_mode, phase)`. A pair is
NON-COMPARABLE if any of these hold:
- k, seed or temperature differ
- per-session contexts drift by more than 5%
- attention paths differ
- either row has errors or a flag (other than `K_UNDECLARED`)

## Legacy import

`bench import-legacy src.jsonl dest.jsonl [--correct-summed-ratio K | --correct-mtp6-hit-rate] [--port N]`
maps old fields (`users`, `ctx`, `tok/s` / `total_tps` / `per_user_tps`,
`acc` / `acceptance`, `mem_min_kb`) into this schema. Every imported row is
flagged `CONVERTED`. A protected production port adds `PROTECTED_PORT`. The summed-ratio
correction multiplies by (k+1)/(2k).
