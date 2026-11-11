# xpu-llm-bench

Consistent, comparable benchmark runs for LLM serving on an Intel XPU bench host (written against 2× Intel Arc Pro B60, vLLM 0.30 XPU, MTP / QSA / K8V4 / DFlash2).

Same methodology, same metrics, same report columns — so a run from Tuesday can be put next to a run from Friday without guessing what was measured.

## Quick start

```bash
# Site config: the real production names/ports of your bench host (untracked).
cp bench-site.example.yaml bench-site.local.yaml   # then edit it

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# See the progressive rung order, a time estimate, and KV/RAM feasibility
# (no GPUs, no containers):
bench plan specs/qsa-vs-dense.yaml --profile quick

# Dry-run prints the plan and the exact safety commands it would run, and writes
# placeholder rows (still refuses protected production ports / container names):
bench run specs/qsa-vs-dense.yaml --dry-run --output-dir /tmp/xpu-llm-bench-demo

# Against a live OpenAI-compatible endpoint (test container only):
#   export OPENAI_API_KEY=...    # optional; read from the env, never from the spec
bench run specs/qsa-full-fix-mtp6.yaml --profile quick --output-dir results

# After an abort, continue from the last completed rung:
bench run specs/qsa-full-fix-mtp6.yaml --profile quick --output-dir results --resume

# Tables and A/B deltas (non-comparable rows are flagged loudly):
bench report results/qsa-full-fix-mtp6/results.jsonl
bench compare results/qsa-vs-dense__dense/results.jsonl results/qsa-vs-dense__qsa/results.jsonl

# Older ad-hoc rung JSONL; convert the old summed ratio for a given k (k=6 → ×7/12):
bench import-legacy old.jsonl results/imported.jsonl --correct-summed-ratio 6 --port 8102
```

Tests and lint (no GPUs; a mock OpenAI server is used):

```bash
ruff check src tests
pytest
```

## What a run records

**Headline speed is verify steps/s per user, shown next to decode tok/s.**
The true hit rate is `(delivered − verify_steps) / (k × verify_steps)`, with k
taken from the spec (MTP-6 = 6, DFlash2 = 7, MTP-3 = 3), never hard-coded.

Each rung records:
- **Speed:** prefill tok/s, TTFT, TPOT, ITL p50/p95/p99, end-to-end latency, queue time
- **Server state:** running/waiting (mean and max), prefix-cache hit rate, KV usage %, preemptions
- **Host memory:** MemAvailable minimum, RSS of the API server, engine and each TP worker, and any other resident vLLM stack
- **What actually ran:** the attention path, graphs, k and KV dtype, read from server logs rather than flags, plus a build fingerprint (image id + hash of every mount)
- **Seed:** the fixed prompt seed
- **Quality:** logprob vs a dense baseline, MTP on/off greedy divergence, needle, code-edit unit test, format/repetition
- **Outcome:** an exit reason (`RAM_FLOOR` / `OOM` / `SPEC_ASSERT` / `ENGINE_ERROR`) and flags

The ladder is progressive, with one boot per config:
1. Session 1 grows 8K → 32K → 64K → 131K in one conversation. It is never refilled and never shrunk.
2. Sessions 2…8 then join one at a time while session 1 stays live, and every live session decodes in each window.

There are three prompt types (code edit, writing, chat) and two load levels (all decoding together, or staggered arrivals). `--resume` continues from the last completed rung. The `quick` profile (1/2/4/8 users × 8K/32K × 3 types) is about 10 minutes. Details are in [docs/METHODOLOGY.md](docs/METHODOLOGY.md); columns and flags are in [docs/RESULTS-SCHEMA.md](docs/RESULTS-SCHEMA.md).

## Example specs

| Spec | Role |
|---|---|
| [`specs/prod-dense-baseline.yaml`](specs/prod-dense-baseline.yaml) | Dense baseline, MTP-6, full matrix |
| [`specs/qsa-full-fix-mtp6.yaml`](specs/qsa-full-fix-mtp6.yaml) | QSA + current full-fix set, MTP-6 |
| [`specs/qsa-vs-dense.yaml`](specs/qsa-vs-dense.yaml) | Prioritized A/B: `K8V4_QSA=1` vs `0` |
| [`specs/qsa-width-clamp.yaml`](specs/qsa-width-clamp.yaml) | Prioritized A/B: width clamp before/after |
| [`specs/dflash2-k7.yaml`](specs/dflash2-k7.yaml) | DFlash2 k=7, capture sizes 8..64 step 8 |
| [`specs/mtp3.yaml`](specs/mtp3.yaml) | Planned MTP-3 (k=3), capture sizes 4..32 |
| [`specs/mtp-off-reference.yaml`](specs/mtp-off-reference.yaml) | Spec decode off: greedy reference for the MTP on/off check |
| [`specs/one-fix-at-a-time.yaml`](specs/one-fix-at-a-time.yaml) | Template for a single-fix A/B |

Unknown real values (image digests, served model name, launch args, some env flags, how the runner reaches the bench host) are marked `TODO` in those files.

## Safety

These rules are enforced in code, and the tests check them:
- Refuses any endpoint on a protected production port and any protected production container or service, plus CI-runner containers. The real names and ports come from an untracked site config (`bench-site.local.yaml`, `~/.config/xpu-llm-bench/site.yaml`, or `$XPU_BENCH_SITE`; see [`bench-site.example.yaml`](bench-site.example.yaml)). Site values only add protection; the built-in neutral defaults (`vllm-production`, port 8001) always apply. `$BENCH_HOST` can name the host for SSH access.
- Never starts, stops or restarts a container.
- `bench-hold`: checks `status` first and **refuses if anyone else holds it**, because `acquire` silently overwrites. Also refuses an active operator quiet hold (optional file, from site config). It then acquires, confirms, renews while it still owns the hold, and releases only its own hold.
- Aborts at a 2 GiB `MemAvailable` floor (`--resume` continues).
- Never runs `xpu-smi`, not even discovery.
- API keys come from the environment only.

## License

No license has been chosen yet; until one is added, all rights are reserved.
