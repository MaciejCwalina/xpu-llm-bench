"""Importer for older ad-hoc rung JSONL files.

Legacy rows typically have fields like users, ctx, tok/s, acc / acceptance.
A previous script reported the summed ratio 2A/((k+1)D). Pass
`--correct-summed-ratio K` to convert to the per-slot rate A/(kD)
(factor (k+1)/(2k); `--correct-mtp6-hit-rate` is the k=6 shortcut, 7/12).
Imported rows carry the CONVERTED flag and never enter headline tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xpu_llm_bench.results import RungResult, stamp_pair, write_jsonl_row
from xpu_llm_bench.safety import site_forbidden_ports, url_port

MTP6_HIT_RATE_CORRECTION = 7.0 / 12.0


def summed_ratio_correction(k: int) -> float:
    """The old script reported 2A/((k+1)D); the per-slot rate is A/(kD).

    Multiply the old value by (k+1)/(2k): 7/12 for k=6, 8/14 for k=7, 4/6 for k=3.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    return (k + 1) / (2.0 * k)


def _first(data: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    return None


def _as_int(val: Any) -> int | None:
    if val is None:
        return None
    if isinstance(val, dict):
        # e.g. ctx: {1: 8192, 2: 8192}
        nums = [int(v) for v in val.values() if v is not None]
        return max(nums) if nums else None
    return int(val)


def _as_float(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, dict):
        nums = [float(v) for v in val.values() if v is not None]
        if not nums:
            return None
        return sum(nums) / len(nums)
    return float(val)


def import_row(
    data: dict[str, Any],
    run_id: str = "legacy",
    correct_mtp6_hit_rate: bool = False,
    prompt_type: str = "unknown",
    load_mode: str = "closed_all_decoding",
    correct_k: int | None = None,
    port: int | None = None,
) -> RungResult:
    users = _as_int(_first(data, "users", "n_users", "live_count"))
    if users is None and isinstance(data.get("live"), list):
        users = len(data["live"])
    if users is None and isinstance(data.get("per_user_tps"), dict):
        users = len(data["per_user_tps"])
    ctx = _as_int(_first(data, "ctx_per_user", "ctx", "context", "context_tokens"))
    tok_total = _as_float(_first(data, "decode_tok_s_total", "total_tps", "tok/s", "toks", "tps"))
    tok_user = _as_float(_first(data, "decode_tok_s_per_user", "per_user_tps", "tok_s_per_user"))
    if tok_user is None and tok_total is not None and users:
        tok_user = tok_total / users
    if tok_total is None and tok_user is not None and users:
        tok_total = tok_user * users
    hit = _as_float(_first(data, "hit_rate", "acceptance", "acc"))
    flags = ["CONVERTED"]
    if correct_mtp6_hit_rate and correct_k is None:
        correct_k = 6
    if hit is not None and correct_k:
        hit = hit * summed_ratio_correction(correct_k)
    bad_ports = site_forbidden_ports()
    legacy_url = str(_first(data, "url", "base_url") or "")
    legacy_port = url_port(legacy_url) if "://" in legacy_url else None
    if port in bad_ports or _first(data, "port") in bad_ports or legacy_port in bad_ports:
        flags.append("PROTECTED_PORT")
    acc_step = _as_float(_first(data, "accepted_drafts_per_step", "acc_per_step"))
    label = str(_first(data, "rung_id", "label", "rung") or "legacy")
    valid = "valid"
    reason = "imported from legacy jsonl"
    if data.get("error"):
        valid = "invalid"
        reason = str(data["error"])
    utc, et = stamp_pair()
    return RungResult(
        run_id=run_id,
        rung_id=str(label),
        users=users or 0,
        ctx_tokens=[ctx] * (users or 1) if ctx else [],
        ctx_per_user=ctx or 0,
        prompt_type=str(_first(data, "prompt_type") or prompt_type),
        load_mode=str(_first(data, "load_mode") or load_mode),
        prefill_tok_s=_as_float(_first(data, "prefill_tok_s")),
        ttft_ms=_as_float(_first(data, "ttft_ms"))
        or (
            (float(min(data["ttft_s"].values())) * 1000.0)
            if isinstance(data.get("ttft_s"), dict) and data["ttft_s"]
            else _as_float(_first(data, "ttft_s")) * 1000.0 if _first(data, "ttft_s") else None
        ),
        decode_tok_s_per_user=tok_user,
        decode_tok_s_total=tok_total,
        accepted_drafts_per_step=acc_step,
        hit_rate=hit,
        mem_available_min_gib=(
            float(data["mem_min_kb"]) / (1024 * 1024)
            if data.get("mem_min_kb") is not None
            else _as_float(_first(data, "mem_available_min_gib"))
        ),
        preemptions=_as_int(_first(data, "preemptions")),
        attention_path=str(_first(data, "attention_path") or "unknown"),  # type: ignore[arg-type]
        valid=valid,  # type: ignore[arg-type]
        reason=reason,
        errors=[str(data["error"])] if data.get("error") else [],
        k=correct_k,
        flags=flags,
        started_utc=utc,
        started_et=et,
        finished_utc=utc,
        finished_et=et,
    )


def import_legacy_file(
    src: str | Path,
    dest: str | Path,
    run_id: str = "legacy",
    correct_mtp6_hit_rate: bool = False,
    prompt_type: str = "unknown",
    correct_k: int | None = None,
    port: int | None = None,
) -> int:
    count = 0
    dest_path = Path(dest)
    if dest_path.exists():
        dest_path.unlink()
    for line in Path(src).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = import_row(
            json.loads(line),
            run_id=run_id,
            correct_mtp6_hit_rate=correct_mtp6_hit_rate,
            prompt_type=prompt_type,
            correct_k=correct_k,
            port=port,
        )
        write_jsonl_row(dest_path, row)
        count += 1
    return count
