# xpu-llm-bench

Consistent, comparable benchmark runs for LLM serving on an Intel XPU bench host (written against 2× Intel Arc Pro B60, vLLM 0.30 XPU, MTP / QSA / K8V4 / DFlash2).

Same methodology, same metrics, same report columns — so a run from Tuesday can be put next to a run from Friday without guessing what was measured.

## Prerequisites

The harness does **not** boot, configure or stop vLLM. You bring a running
test container; the harness measures it and checks it is what the spec says.

1. **Clone and install** (Python 3.11+):

   ```bash
   git clone <this repo> xpu-llm-bench && cd xpu-llm-bench
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   ```

2. **Write the site config** (see [Site config](#site-config) below). A live
   `bench run` refuses to start without one unless you pass `--no-site-config`.

3. **Boot a test container yourself**, on a port that is not a production
   port, for example:

   ```bash
   docker run -d --name vllm-bench-qsa ... <image> vllm serve <model> --port 8102 ...
   ```

   Never point the harness at a production container or port; it refuses
   protected names and ports, but it is your container to launch.

4. **Point a spec at it.** Copy a spec from `specs/` and set:
   - `endpoint.base_url` / `metrics_url` / `health_url` to the container's port,
     and `endpoint.model` to the served model name;
   - `server.container_name` to the container's name (it must exist and be
     running: preflight runs `docker inspect` on it);
   - `server.image`, `server.env`, `server.launch_args` to what you launched,
     or `null` / empty to skip a check.

   **`server.env`, `server.launch_args` and `server.image` are expectations,
   not settings.** The harness never applies them. At preflight it compares
   them with `docker inspect` (env, command line, image) and, on any
   mismatch, flags every row `NOT_AS_LABELED` and records the evidence in
   `manifest.json` → `launch_check`.

   Every `TODO` or `<...>` value must be replaced (or set to `null`) before a
   live run; preflight lists the remaining ones by field path and stops.

5. **Check, dry-run, then run:**

   ```bash
   bench plan my-spec.yaml               # order, time, GPU/RAM feasibility (exit 1 if it cannot fit)
   bench run my-spec.yaml --dry-run      # plan + the exact safety commands; writes only <out>/<spec>/dry-run/
   bench run my-spec.yaml --output-dir results
   bench report results/<spec-name>      # markdown on stdout
   ```

### A/B specs (variants)

Each variant must run against **its own booted container**. Variants share
one endpoint by default, so `bench run` refuses to run them back to back
unless you choose one of:

- `--variant NAME`: run one variant; boot its container first, then run the
  next variant with its own container;
- `--pause-between-variants`: the runner stops before each variant and tells
  you which container (name, env, args) to boot, then waits for Enter;
- a per-variant `endpoint:` (`base_url`, `metrics_url`, `health_url`) so
  both containers can be up at once on different ports.

Each variant's container is verified against that variant's `env` /
`launch_args` / `image`; a container that does not match is `NOT_AS_LABELED`.

## Site config

The production details of your bench host live in an untracked YAML file,
searched in this order (the current directory is **not** searched):

1. `$XPU_BENCH_SITE` (a path; an empty value disables site config),
2. `<repo root>/bench-site.local.yaml` (gitignored),
3. `~/.config/xpu-llm-bench/site.yaml`.

`bench plan` and `bench run` print which file was loaded. Start from
[`bench-site.example.yaml`](bench-site.example.yaml) and fill in:

| Field | Required | What to put there |
|---|---|---|
| `host` | yes | Display name of the bench host (written to results as `hardware.host`). |
| `ssh_host` | if `access.mode: ssh` | An `~/.ssh/config` alias for the bench host. Never an IP in a spec. |
| `protected_containers` | yes | Exact names of every production container the runner must never touch. |
| `protected_patterns` | no | Case-insensitive regexes for other protected services (gateways, git hosting, CI). |
| `forbidden_ports` | yes | Host ports of every production endpoint; endpoints and containers on them are refused. |
| `hold_command` | no | Path of the `bench-hold` CLI on the bench host (default: `bench-hold` on PATH). |
| `quiet_hold_file` | no | Operator quiet-hold file (read only); an active quiet hold blocks a run. |

Any value still containing `<...>`, `TODO`, `example` or `changeme` is
rejected, as is an empty `host`, `protected_containers` or `forbidden_ports`.
Site values only ever add protection; the built-in neutral defaults
(`vllm-production`, port 8001, CI-runner names) always apply.

## Quick start

```bash
# See the progressive rung order, a time estimate, and GPU/RAM feasibility
# (no GPUs, no containers):
bench plan specs/qsa-vs-dense.yaml --profile quick

# Dry run: the plan plus the exact safety commands it would run, printed to
# stdout. It writes only under <out>/<spec>/dry-run/ (manifest status
# dry_run, ignored by --resume, no sessions.json) and still refuses
# protected production ports / container names:
bench run specs/qsa-vs-dense.yaml --dry-run --output-dir /tmp/xpu-llm-bench-demo

# Against your running test container (see Prerequisites):
#   export OPENAI_API_KEY=...    # optional; read from the env, never from the spec
bench run my-qsa-full-fix.yaml --profile quick --output-dir results

# After an abort, continue from the last completed rung:
bench run my-qsa-full-fix.yaml --profile quick --output-dir results --resume

# A/B: one variant at a time, each against its own container:
bench run my-qsa-vs-dense.yaml --variant qsa --output-dir results
bench run my-qsa-vs-dense.yaml --variant dense --output-dir results

# Report (markdown on stdout; --output FILE / --csv FILE to write files) and A/B deltas:
bench report results/my-qsa-full-fix
bench compare results/qsa-vs-dense__dense/results.jsonl results/qsa-vs-dense__qsa/results.jsonl

# Older rung JSONL (scaffold rows or speed-fix ladder-rungs.jsonl):
bench import-legacy old.jsonl results/imported.jsonl --prompt-type code_edits
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

The ladder is progressive, with one boot per config, for every profile
(including `custom`):
1. Session 1 grows 8K → 32K → 64K → 131K in one conversation. It is never refilled and never shrunk.
2. Sessions 2…8 then join one at a time while session 1 stays live, and every live session decodes in each window.

The old grid order (every user at once at the first rung, then step down) is
refused unless a spec sets both `ladder.shape: grid` and `ladder.allow_grid: true`,
and then `bench plan` / `bench run` print a loud warning.

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

Unknown real values (image digests, served model name, launch args, some env flags, how the runner reaches the bench host) are marked `TODO` in those files. A live run refuses to start until each is set (or set to `null`); `--dry-run` lists them as warnings.

## Safety

These rules are enforced in code, and the tests check them:
- Refuses any endpoint on a protected production port and any protected production container or service, plus CI-runner containers. The real names and ports come from an untracked site config (`bench-site.local.yaml`, `~/.config/xpu-llm-bench/site.yaml`, or `$XPU_BENCH_SITE`; see [`bench-site.example.yaml`](bench-site.example.yaml)). Site values only add protection; the built-in neutral defaults (`vllm-production`, port 8001) always apply. `$BENCH_HOST` can name the host for SSH access.
- Never starts, stops or restarts a container.
- `bench-hold`: checks `status` first and **refuses if anyone else holds it**, because `acquire` silently overwrites. Also refuses an active operator quiet hold (optional file, from site config). It then acquires, confirms, renews while it still owns the hold, and releases only its own hold.
- Preflight stops (exit 3, manifest `status: refused`) on: no site config, `TODO`/`<...>` spec values, a missing or stopped test container, a planner GPU/RAM failure (using the server's measured RSS), or a bench-hold held by someone else.
- Aborts at a 2 GiB `MemAvailable` floor (`--resume` continues).
- Never runs `xpu-smi`, not even discovery.
- API keys come from the environment only.

## License

No license has been chosen yet; until one is added, all rights are reserved.
