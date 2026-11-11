"""Result schema: one manifest per run, one JSONL row per measured rung."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from xpu_llm_bench import USER_TZ, __version__

AttentionPath = Literal["qsa", "dense", "fallback", "mixed", "unknown"]
Validity = Literal["valid", "invalid", "warning"]

# Row flags. A flagged row is excluded from headline numbers and from clean A/B.
FLAGS: dict[str, str] = {
    "NOT_AS_LABELED": "Log evidence contradicts the spec label (attention path, graphs, k, KV dtype).",
    "QSA_FALLBACK": "QSA requested/wrapped, but runtime lines in the window show the dense/skip path.",
    "MIXED_PATH": "Both QSA and skip/dense runtime lines in the window.",
    "GRAPH_MISS": "Decode batch tokens users*(k+1) not in the logged/declared capture sizes, or graphs off.",
    "SPEC_COUNTER_BROKEN": "T != k*D, A > T, sum(A_p) != A, or A disagrees with delivered - D.",
    "STEP_COUNT_MISMATCH": "Stream-counted verify steps and server drafts differ by > 5%.",
    "DERIVED_STEPS": "Stream gave no per-step token counts; verify steps taken from server drafts.",
    "K_UNDECLARED": "Spec did not declare k; k was taken from draft_tokens/drafts.",
    "SHORT_WINDOW": "Fewer verify steps per session than ladder.min_window_steps.",
    "EOS_IN_WINDOW": "Model emitted EOS inside the window (ignore_eos kept decoding).",
    "PREEMPTED": "Preemptions > 0 during the rung.",
    "CONCURRENT_CLIENT": "More running/waiting requests than this runner's sessions (another client).",
    "FOREIGN_STACK": "Another vLLM stack (different container) is resident on the host.",
    "PROTECTED_PORT": "Server bound to a protected production port (only possible on imported rows).",
    "RAM_FLOOR": "MemAvailable hit the floor; runner guard aborted.",
    "OOM": "Out-of-memory (container OOMKilled or OOM lines in logs).",
    "SPEC_ASSERT": "Known assert: Expected spec_token == num_spec_decodes * (num_speculative_tokens + 1).",
    "ENGINE_ERROR": "Engine error / traceback in logs or HTTP 5xx.",
    "TIMEOUT": "Client timeout.",
    "CLIENT_ERROR": "Request failed without a server-side cause in the logs.",
    "REBOOTED": "Container StartedAt changed mid-run (boot-once violated).",
    "RESUMED_REFILL": "First rung after --resume on a new boot: sessions were re-prefilled.",
    "QUALITY_FAIL": "A quality check failed on this rung (see quality fields).",
    "CONVERTED": "Value converted from a legacy definition (import-legacy).",
    "DRY_RUN": "Placeholder row from --dry-run.",
}


def utc_now() -> datetime:
    return datetime.now(UTC)


def et_now(now: datetime | None = None) -> datetime:
    src = now or utc_now()
    return src.astimezone(ZoneInfo(USER_TZ))


def stamp_pair(now: datetime | None = None) -> tuple[str, str]:
    """Return (utc_iso, america_new_york_iso) with offsets."""
    utc = (now or utc_now()).astimezone(UTC)
    et = et_now(utc)
    return utc.isoformat(), et.isoformat()


class HardwareSnapshot(BaseModel):
    host: str = "<bench-host>"
    gpu_model: str = "Intel Arc Pro B60"
    gpu_count: int = 2
    tensor_parallel: int = 2
    driver: str | None = None  # TODO: collect if available without xpu-smi
    oneapi: str | None = None  # TODO
    host_ram_gib: float | None = None
    mem_available_floor_gib: float = 2.0


class GraphSnapshot(BaseModel):
    mode: str
    per_gpu: bool = False
    capture_sizes: list[int] = Field(default_factory=list)
    graphs_on: bool = True
    eager_fallback: bool = False


class SpecDecodeSnapshot(BaseModel):
    method: str
    num_speculative_tokens: int | None = None
    dflash_k: int | None = None


class AttentionSnapshot(BaseModel):
    requested: str | None = None
    actual: AttentionPath = "unknown"
    evidence_lines: list[str] = Field(default_factory=list)


class DecodeMethod(BaseModel):
    method_id: str = "fixed-tokens-chunks-v2"
    window: str = "fixed_output_tokens"
    output_tokens: int
    counters: list[str] = Field(default_factory=list)
    formula: str = (
        "Per session: window = first token -> last token of a fixed max_tokens output "
        "(ignore_eos). delivered E_i = tokens after the first (the first is the prefill step); "
        "verify steps S_i = token-bearing SSE chunks after the first (one chunk per engine step). "
        "decode tok/s = E_i / window; steps/s = S_i / window. "
        "true hit rate = (E - S) / (k * S) with E, S summed over sessions; "
        "S_server = delta(spec_decode_num_drafts_total) cross-checks S (STEP_COUNT_MISMATCH > 5%). "
        "accepted_drafts_per_step = E/S - 1 (from the stream); acceptance_rate_vllm = A/T is kept "
        "for continuity only."
    )


class QualityResult(BaseModel):
    needle_pass: bool | None = None
    needle_detail: str | None = None
    code_edit_pass: bool | None = None
    code_edit_detail: str | None = None
    format_ok: bool | None = None
    format_problems: list[str] = Field(default_factory=list)
    repetition_max: float | None = None
    greedy_ref: str | None = None  # match | diverged | context_diverged | no_reference
    greedy_first_divergence: int | None = None  # token index in the session output
    logprob_ok: bool | None = None
    logprob_worst_delta: float | None = None  # current - baseline mean logprob (nats/token)
    logprob_scores: dict[str, float] = Field(default_factory=dict)


class RungResult(BaseModel):
    run_id: str
    rung_id: str
    users: int
    ctx_tokens: list[int]  # per live session, in session order
    ctx_per_user: int  # representative / target context for this rung
    prompt_type: str
    load_mode: str
    phase: str = "meas"  # S (session 1 grows) | C (sessions join) | meas (grid)
    # Identity of what ran
    prompt_seed: int | None = None
    session_seeds: list[int] = Field(default_factory=list)
    temperature: float | None = None
    k: int | None = None
    build_fingerprint: str | None = None
    launch_fingerprint: str | None = None
    image_id: str | None = None
    boot_started_at: str | None = None
    attention_boot: str | None = None
    capture_sizes: list[int] = Field(default_factory=list)
    decode_batch_tokens: int | None = None
    # Prefill / TTFT
    prefill_tok_s: float | None = None
    prefill_tokens: int | None = None
    ttft_ms: float | None = None
    # Decode
    decode_tok_s_per_user: float | None = None
    decode_tok_s_total: float | None = None
    steps_s_per_user: float | None = None
    steps_s_total: float | None = None
    tpot_ms: float | None = None
    decode_window_s: float | None = None
    # MTP / speculation. hit_rate is the TRUE hit rate:
    # (delivered - verify_steps) / (k * verify_steps).
    accepted_drafts_per_step: float | None = None
    hit_rate: float | None = None
    acceptance_rate_vllm: float | None = None  # A/T from the counters, for continuity
    tokens_per_step: float | None = None
    tokens_delivered: int | None = None
    verify_steps: int | None = None
    verify_steps_stream: int | None = None
    verify_steps_server: int | None = None
    draft_tokens: int | None = None
    accepted_drafts: int | None = None
    drafts: int | None = None
    # Latency
    itl_p50_ms: float | None = None
    itl_p95_ms: float | None = None
    itl_p99_ms: float | None = None
    e2e_ms: float | None = None
    # Queue / cache / KV
    queue_time_ms: float | None = None
    running_reqs: float | None = None
    waiting_reqs: float | None = None
    running_max: float | None = None
    waiting_max: float | None = None
    kv_cache_usage_max_pct: float | None = None
    prefix_cache_hit_rate: float | None = None
    kv_cache_usage_pct: float | None = None
    preemptions: int | None = None
    # Memory
    mem_available_min_gib: float | None = None
    gpu_mem: str | None = None  # free-form; never collected via xpu-smi
    rss_kib: dict[str, int] = Field(default_factory=dict)  # api_server, engine, worker_tp0, ...
    other_vllm_stacks: list[str] = Field(default_factory=list)
    # Path / graphs actually observed
    attention_path: AttentionPath = "unknown"
    attention_evidence: list[str] = Field(default_factory=list)
    graphs_on: bool | None = None
    eager_fallback: bool | None = None
    kv_format: str | None = None
    # Validity
    valid: Validity = "valid"
    reason: str | None = None
    errors: list[str] = Field(default_factory=list)
    timeouts: int = 0
    exit_reason: str | None = None  # RAM_FLOOR | OOM | SPEC_ASSERT | ENGINE_ERROR | TIMEOUT | CLIENT_ERROR
    flags: list[str] = Field(default_factory=list)
    quality: QualityResult = Field(default_factory=QualityResult)
    started_utc: str | None = None
    started_et: str | None = None
    finished_utc: str | None = None
    finished_et: str | None = None

    @property
    def comparable_key(self) -> tuple[Any, ...]:
        return (
            self.users,
            self.ctx_per_user,
            self.prompt_type,
            self.load_mode,
            self.phase,
        )


class RunManifest(BaseModel):
    run_id: str
    spec_name: str
    spec_path: str | None = None
    profile: str
    tool_version: str = __version__
    started_utc: str
    started_et: str
    finished_utc: str | None = None
    finished_et: str | None = None
    status: Literal["in_progress", "completed", "aborted"] = "in_progress"
    abort_reason: str | None = None
    image: str | None = None
    image_digest: str | None = None
    server_source_commit: str | None = None
    hardware: HardwareSnapshot
    config: dict[str, Any] = Field(default_factory=dict)
    attention: AttentionSnapshot = Field(default_factory=AttentionSnapshot)
    kv_format: str | None = None
    graph: GraphSnapshot | None = None
    speculative: SpecDecodeSnapshot | None = None
    decode_method: DecodeMethod | None = None
    completed_rung_ids: list[str] = Field(default_factory=list)
    output_jsonl: str | None = None
    notes: list[str] = Field(default_factory=list)
    prompt_seed: int | None = None
    k: int | None = None
    fingerprint: dict[str, Any] = Field(default_factory=dict)
    boot: dict[str, Any] = Field(default_factory=dict)
    host: dict[str, Any] = Field(default_factory=dict)
    corpus_files: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    hold: dict[str, Any] = Field(default_factory=dict)
    first_greedy_divergence: dict[str, Any] | None = None

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.model_dump(), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> RunManifest:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def write_jsonl_row(path: str | Path, row: RungResult | dict[str, Any]) -> None:
    payload = row.model_dump() if isinstance(row, RungResult) else row
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=False) + "\n")


def read_jsonl(path: str | Path) -> list[RungResult]:
    rows: list[RungResult] = []
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(RungResult.model_validate(json.loads(line)))
    return rows
