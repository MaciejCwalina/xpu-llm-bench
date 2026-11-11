"""Parse what the server actually booted with, and why it failed, from container logs.

Only log evidence counts. ``Using Xe2 K8/V4 attention backend.`` is printed by
both dense K8V4 and QSA builds, so it is never sufficient on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from xpu_llm_bench.spec import LogPatterns

_TS = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?(Z|[+-]\d{2}:\d{2})\s")
_CAPTURE_SIZES = re.compile(r"cudagraph_capture_sizes['\"]?\s*[:=]\s*\[([\d,\s]*)\]")
_CG_MODE = re.compile(r"cudagraph_mode['\"]?\s*[:=]\s*['\"<]?(?:CUDAGraphMode\.)?([A-Z_]+)")
_KV_DTYPE = re.compile(r"kv_cache_dtype['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9_]+)")
_KV_TOKENS = re.compile(r"KV cache size:\s*([\d,]+) tokens")
_SPEC = re.compile(r"SpeculativeConfig\(method='([^']+)', model='([^']*)', num_spec_tokens=(\d+)\)")
_MAX_LEN = re.compile(r"max_seq_len=(\d+)")
_VLLM_VER = re.compile(r"Initializing a V1 LLM engine \(v([^)]+)\)")
_CAPTURING = re.compile(r"Capturing CUDA graphs \(([^)]+)\)[^|]*\|[^|]*\|\s*(\d+)/(\d+)")
_GRAPH_DONE = re.compile(r"Graph capturing finished in [\d.]+ secs?, took ([\d.]+) GiB")
_ENFORCE_EAGER = re.compile(r"'enforce_eager':\s*True|--enforce-eager")
_FUSED_FALLBACK = re.compile(r"falling back to rebuilding attention metadata between draft steps")

_OOM = re.compile(
    r"out of memory|OutOfMemoryError|OUT_OF_DEVICE_MEMORY|UR_RESULT_ERROR_OUT_OF_(DEVICE|HOST)_MEMORY|Killed process \d+",
    re.IGNORECASE,
)
_ENGINE_ERR = re.compile(
    r"EngineDeadError|EngineCore (?:encountered|failed)|RuntimeError:|AssertionError|Traceback \(most recent call last\)"
)


def strip_ts(line: str) -> tuple[float | None, str]:
    m = _TS.match(line)
    if not m:
        return None, line
    iso = m.group(1) + (m.group(3) if m.group(3) != "Z" else "+00:00")
    try:
        t = datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return None, line
    frac = m.group(2)
    if frac:
        t += float("0" + frac)
    return t, line[m.end():]


def lines_since(text: str, since_epoch: float | None) -> list[str]:
    """Lines from ``docker logs --timestamps`` at or after since_epoch (timestamps stripped).

    Lines without a timestamp follow the previous line's decision; lines before
    the first timestamp (or logs with no timestamps at all) are kept, because
    their age is unknown.
    """
    out: list[str] = []
    keep = True
    for raw in text.splitlines():
        t, body = strip_ts(raw)
        if t is not None and since_epoch is not None:
            keep = t >= since_epoch
        if keep:
            out.append(body)
    return out


@dataclass
class BootEvidence:
    vllm_version: str | None = None
    attention_boot: str = "unknown"  # qsa | dense | unknown
    banner_lines: list[str] = field(default_factory=list)
    qsa_wrap_lines: list[str] = field(default_factory=list)
    qsa_memory_plan_lines: list[str] = field(default_factory=list)
    qsa_graph_lines: list[str] = field(default_factory=list)
    fused_draft_fallback: bool = False
    graphs_on: bool | None = None
    enforce_eager: bool = False
    cudagraph_mode: str | None = None
    capture_sizes_logged: list[int] = field(default_factory=list)
    capture_counts: dict[str, str] = field(default_factory=dict)  # mode -> "n/N" (last seen)
    graph_done_lines: list[str] = field(default_factory=list)
    kv_dtype: str | None = None
    kv_tokens: int | None = None
    kv_line: str | None = None
    spec_method: str | None = None
    spec_k: int | None = None
    spec_line: str | None = None
    max_model_len: int | None = None
    evidence: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "vllm_version": self.vllm_version,
            "attention_boot": self.attention_boot,
            "graphs_on": self.graphs_on,
            "enforce_eager": self.enforce_eager,
            "cudagraph_mode": self.cudagraph_mode,
            "capture_sizes_logged": self.capture_sizes_logged,
            "capture_counts": self.capture_counts,
            "kv_dtype": self.kv_dtype,
            "kv_tokens": self.kv_tokens,
            "spec_method": self.spec_method,
            "spec_k": self.spec_k,
            "max_model_len": self.max_model_len,
            "fused_draft_fallback": self.fused_draft_fallback,
        }


def _keep(ev: list[str], line: str, cap: int = 40) -> None:
    if len(ev) < cap:
        ev.append(line[:400])


def parse_boot(text: str | list[str], patterns: LogPatterns | None = None) -> BootEvidence:
    pats = patterns or LogPatterns()
    banner = re.compile(pats.dense_banner)
    wrap = re.compile(pats.qsa_wrap)
    plan = re.compile(pats.qsa_memory_plan)
    lines = text.splitlines() if isinstance(text, str) else list(text)
    ev = BootEvidence()
    for raw in lines:
        _, line = strip_ts(raw)
        # tqdm progress bars arrive as one long line with \r-free segments
        if (m := _VLLM_VER.search(line)) and ev.vllm_version is None:
            ev.vllm_version = m.group(1)
        if banner.search(line):
            ev.banner_lines.append(line.strip()[:400])
            _keep(ev.evidence, line.strip())
        if wrap.search(line):
            ev.qsa_wrap_lines.append(line.strip()[:400])
            _keep(ev.evidence, line.strip())
        if plan.search(line):
            ev.qsa_memory_plan_lines.append(line.strip()[:400])
            _keep(ev.evidence, line.strip())
        if "QSA_GRAPH" in line:
            ev.qsa_graph_lines.append(line.strip()[:400])
        if _FUSED_FALLBACK.search(line):
            ev.fused_draft_fallback = True
            _keep(ev.evidence, line.strip())
        if _ENFORCE_EAGER.search(line):
            ev.enforce_eager = True
        if (m := _CAPTURE_SIZES.search(line)) and not ev.capture_sizes_logged:
            ev.capture_sizes_logged = [int(x) for x in m.group(1).replace(" ", "").split(",") if x]
        if (m := _CG_MODE.search(line)) and ev.cudagraph_mode is None:
            ev.cudagraph_mode = m.group(1)
        if (m := _KV_DTYPE.search(line)) and ev.kv_dtype is None:
            ev.kv_dtype = m.group(1)
        if m := _KV_TOKENS.search(line):
            ev.kv_tokens = int(m.group(1).replace(",", ""))
            ev.kv_line = line.strip()[:400]
            _keep(ev.evidence, line.strip())
        if m := _SPEC.search(line):
            ev.spec_method, ev.spec_k = m.group(1), int(m.group(3))
            ev.spec_line = m.group(0)
        if (m := _MAX_LEN.search(line)) and ev.max_model_len is None:
            ev.max_model_len = int(m.group(1))
        for cm in _CAPTURING.finditer(line):
            ev.capture_counts[cm.group(1)] = f"{cm.group(2)}/{cm.group(3)}"
        if _GRAPH_DONE.search(line):
            ev.graph_done_lines.append(line.strip()[:400])
            _keep(ev.evidence, line.strip())
    if ev.qsa_wrap_lines and ev.qsa_memory_plan_lines:
        ev.attention_boot = "qsa"
    elif ev.qsa_wrap_lines or ev.qsa_memory_plan_lines:
        ev.attention_boot = "unknown"  # half-wired QSA; do not guess
    elif ev.banner_lines:
        ev.attention_boot = "dense"
    if ev.enforce_eager:
        ev.graphs_on = False
    elif ev.graph_done_lines:
        ev.graphs_on = True
    elif lines:
        ev.graphs_on = False if ev.cudagraph_mode == "NONE" else None
    if ev.spec_line:
        _keep(ev.evidence, ev.spec_line)
    return ev


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


@dataclass
class Failure:
    flag: str  # RAM_FLOOR | OOM | SPEC_ASSERT | ENGINE_ERROR | TIMEOUT | CLIENT_ERROR
    detail: str
    lines: list[str] = field(default_factory=list)


def classify_failure(
    log_lines: list[str],
    errors: list[str],
    timed_out: bool = False,
    ram_floor_hit: bool = False,
    oom_killed: bool = False,
    patterns: LogPatterns | None = None,
) -> Failure | None:
    """Most specific cause first. Returns None if nothing failed."""
    pats = patterns or LogPatterns()
    spec_assert = re.compile(pats.spec_assert)
    text_lines = [strip_ts(x)[1] for x in log_lines] + list(errors)
    if ram_floor_hit:
        return Failure("RAM_FLOOR", "MemAvailable reached the floor; runner guard aborted the rung")
    hits = [x for x in text_lines if spec_assert.search(x)]
    if hits:
        return Failure("SPEC_ASSERT", "spec_token assert (Expected spec_token == num_spec_decodes * (k + 1))", hits[:5])
    hits = [x for x in text_lines if _OOM.search(x)]
    if oom_killed or hits:
        return Failure("OOM", "container OOMKilled" if oom_killed else "out-of-memory in logs", hits[:5])
    hits = [x for x in text_lines if _ENGINE_ERR.search(x)]
    if hits:
        return Failure("ENGINE_ERROR", "engine error in logs", hits[:5])
    if timed_out:
        return Failure("TIMEOUT", "client timeout")
    if errors:
        return Failure("CLIENT_ERROR", "; ".join(errors)[:400])
    return None
