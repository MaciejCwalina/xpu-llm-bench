"""Importer for older ad-hoc rung JSONL files.

Two legacy shapes are understood:

* scaffold-era rows: users, ctx, tok/s, acc / acceptance;
* speed-fix ladder rows (``ladder-rungs.jsonl``): label, prompt_tokens,
  rates, total_toks, per_user_steps_s, first_token_s, mem.min_kb,
  tokens_per_step, drafts_delta, draft_tokens_delta, accepted_tokens_delta.

hit_rate is the TRUE hit rate (E-S)/(k*S) = (tokens_per_step - 1)/k. It is
derived from tokens_per_step (or delivered tokens and drafts) with k =
draft_tokens/drafts. It is NEVER copied from vLLM's acceptance counter:
counter values go to acceptance_rate_vllm, and a row with nothing to derive
the hit rate from gets hit_rate=null and the HIT_RATE_UNKNOWN flag.

A previous script reported the summed ratio 2A/((k+1)D). Pass
`--correct-summed-ratio K` to convert that counter value to the per-slot rate
A/(kD) (factor (k+1)/(2k); `--correct-mtp6-hit-rate` is the k=6 shortcut,
7/12); the corrected value lands in acceptance_rate_vllm.
Imported rows carry the CONVERTED flag and never enter headline tables.
"""

from __future__ import annotations

import json
import re
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


def _mean(vals: Any) -> float | None:
    if isinstance(vals, dict):
        vals = list(vals.values())
    if not isinstance(vals, list):
        return _as_float(vals)
    nums = [float(v) for v in vals if v is not None]
    return sum(nums) / len(nums) if nums else None


def _label_ctx(label: str) -> int | None:
    m = re.search(r"(\d{3,})\s*$", label)
    return int(m.group(1)) if m else None


def _phase(label: str, users: int | None) -> str:
    low = label.lower()
    if "join" in low:
        return "C"
    if "start" in low or "grow" in low or users == 1:
        return "S"
    return "meas"


def import_row(
    data: dict[str, Any],
    run_id: str = "legacy",
    correct_mtp6_hit_rate: bool = False,
    prompt_type: str = "unknown",
    load_mode: str = "closed_all_decoding",
    correct_k: int | None = None,
    port: int | None = None,
    attention_path: str | None = None,
) -> RungResult:
    label = str(_first(data, "rung_id", "label", "rung") or "legacy")
    prompt_list = data.get("prompt_tokens") if isinstance(data.get("prompt_tokens"), list) else None
    users = _as_int(_first(data, "users", "n_users", "live_count"))
    if users is None and prompt_list:
        users = len(prompt_list)
    if users is None and isinstance(data.get("live"), list):
        users = len(data["live"])
    if users is None and isinstance(data.get("rates"), list):
        users = len(data["rates"])
    if users is None and isinstance(data.get("per_user_tps"), dict):
        users = len(data["per_user_tps"])
    ctx = _as_int(_first(data, "ctx_per_user", "ctx", "context", "context_tokens"))
    if ctx is None and prompt_list:
        ctx = _label_ctx(label) or max(int(v) for v in prompt_list)
    ctx_tokens = [int(v) for v in prompt_list] if prompt_list else ([ctx] * (users or 1) if ctx else [])

    tok_total = _as_float(_first(data, "decode_tok_s_total", "total_toks", "total_tps", "tok/s", "toks", "tps"))
    tok_user = _mean(_first(data, "decode_tok_s_per_user", "rates", "per_user_tps", "tok_s_per_user"))
    if tok_user is None and tok_total is not None and users:
        tok_user = tok_total / users
    if tok_total is None and tok_user is not None and users:
        tok_total = tok_user * users
    steps_list = data.get("per_user_steps_s")
    steps_user = _mean(steps_list) if steps_list is not None else _as_float(_first(data, "steps_s_per_user"))
    steps_total = (
        float(sum(float(v) for v in steps_list)) if isinstance(steps_list, list) and steps_list
        else _as_float(_first(data, "steps_s_total"))
    )

    flags = ["CONVERTED"]
    # Speculation counters.
    drafts = _as_float(_first(data, "drafts_delta", "drafts"))
    draft_tokens = _as_float(_first(data, "draft_tokens_delta", "drafted_delta", "draft_tokens"))
    accepted = _as_float(_first(data, "accepted_tokens_delta", "accepted_delta", "accepted_drafts"))
    tps = _as_float(_first(data, "tokens_per_step"))
    window = _as_float(_first(data, "overlap_s", "decode_window_s"))
    k: int | None = None
    if drafts and draft_tokens:
        k = int(round(draft_tokens / drafts))
    elif isinstance(data.get("per_pos_index"), list) and data["per_pos_index"]:
        k = len(data["per_pos_index"])
    elif correct_k:
        k = correct_k
    elif correct_mtp6_hit_rate:
        k = 6
    if tps is None and drafts and tok_total is not None and window:
        tps = (tok_total * window) / drafts  # delivered tokens / verify steps
    hit = None
    reason = "imported from legacy jsonl"
    if tps is not None and k:
        hit = (tps - 1.0) / k
        reason += f"; hit_rate=(tokens_per_step-1)/k with k={k}"
    else:
        flags.append("HIT_RATE_UNKNOWN")
        reason += "; hit_rate not derivable (no tokens_per_step/drafts), left empty"

    # vLLM's acceptance counter: kept for continuity, never used as hit_rate.
    acc_counter = _as_float(_first(data, "acceptance_rate", "acceptance", "acc", "hit_rate"))
    if correct_mtp6_hit_rate and correct_k is None:
        correct_k = 6
    if acc_counter is not None and correct_k:
        acc_counter = acc_counter * summed_ratio_correction(correct_k)

    bad_ports = site_forbidden_ports()
    legacy_url = str(_first(data, "url", "base_url") or "")
    legacy_port = url_port(legacy_url) if "://" in legacy_url else None
    if port in bad_ports or _first(data, "port") in bad_ports or legacy_port in bad_ports:
        flags.append("PROTECTED_PORT")
    acc_step = _as_float(_first(data, "accepted_drafts_per_step", "acc_per_step", "mean_accepted_per_step"))
    if acc_step is None and tps is not None:
        acc_step = tps - 1.0

    ttft_ms = _as_float(_first(data, "ttft_ms"))
    for key in ("first_token_s", "ttft_s"):
        val = data.get(key)
        if ttft_ms is not None or val is None:
            continue
        if isinstance(val, dict) and val:
            # Worst (slowest) first token in the rung: the joiner waits behind the live decode.
            ttft_ms = max(float(v) for v in val.values()) * 1000.0
        elif not isinstance(val, dict):
            ttft_ms = float(val) * 1000.0

    mem = data.get("mem") if isinstance(data.get("mem"), dict) else {}
    min_kb = _first(data, "mem_min_kb") or mem.get("min_kb")
    mem_gib = float(min_kb) / (1024 * 1024) if min_kb is not None else _as_float(_first(data, "mem_available_min_gib"))
    kv_pct = _as_float(_first(data, "kv_cache_usage_perc"))

    valid = "valid"
    errors: list[str] = []
    if data.get("error"):
        errors.append(str(data["error"]))
    if isinstance(data.get("errors"), dict) and data["errors"]:
        errors.extend(f"{k_}: {v}" for k_, v in data["errors"].items())
    if data.get("ok") is False or (data.get("status") not in (None, "ok")):
        errors.append(f"legacy status {data.get('status')!r}")
    if errors:
        valid = "invalid"
        reason = "; ".join(errors)
    attn = _first(data, "attention_path")
    if attn is None and attention_path:
        attn = attention_path
    if attn is not None:
        # Legacy rows carry no log evidence: the path is declared, not proven.
        flags.append("ATTENTION_DECLARED")
        reason += f"; attention_path={attn} declared, not proven by logs"
    utc, et = stamp_pair()
    return RungResult(
        run_id=run_id,
        rung_id=label,
        users=users or 0,
        ctx_tokens=ctx_tokens,
        ctx_per_user=ctx or 0,
        prompt_type=str(_first(data, "prompt_type") or prompt_type),
        load_mode=str(_first(data, "load_mode") or load_mode),
        phase=str(_first(data, "phase") or _phase(label, users)),
        prefill_tok_s=_as_float(_first(data, "prefill_tok_s")),
        prefill_tokens=_as_int(_first(data, "prefill_new_tokens", "prefill_tokens")),
        ttft_ms=ttft_ms,
        decode_tok_s_per_user=tok_user,
        decode_tok_s_total=tok_total,
        steps_s_per_user=steps_user,
        steps_s_total=steps_total,
        decode_window_s=window,
        accepted_drafts_per_step=acc_step,
        tokens_per_step=tps,
        hit_rate=hit,
        acceptance_rate_vllm=acc_counter,
        drafts=int(drafts) if drafts is not None else None,
        draft_tokens=int(draft_tokens) if draft_tokens is not None else None,
        accepted_drafts=int(accepted) if accepted is not None else None,
        kv_cache_usage_pct=kv_pct * 100.0 if kv_pct is not None and kv_pct <= 1.0 else kv_pct,
        mem_available_min_gib=mem_gib,
        preemptions=_as_int(_first(data, "preemptions")),
        attention_path=str(attn or "unknown"),  # type: ignore[arg-type]
        valid=valid,  # type: ignore[arg-type]
        reason=reason,
        errors=errors,
        k=k,
        flags=flags,
        started_utc=utc,
        started_local=et,
        finished_utc=utc,
        finished_local=et,
    )


def import_legacy_file(
    src: str | Path,
    dest: str | Path,
    run_id: str = "legacy",
    correct_mtp6_hit_rate: bool = False,
    prompt_type: str = "unknown",
    correct_k: int | None = None,
    port: int | None = None,
    attention_path: str | None = None,
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
            attention_path=attention_path,
        )
        write_jsonl_row(dest_path, row)
        count += 1
    return count
