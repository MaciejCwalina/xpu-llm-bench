"""Build the synthetic example run in tests/fixtures/example_run and render it.

Every value is SYNTHETIC and only illustrates the report layout: an auto
context ladder up to a 262144-token max_model_len (8K ... 260K) for 1/2/4/8
users and three prompt types. 1 and 2 users reach the top rung; 4 users stop
at 163K and 8 users at 64K (KV capacity ceilings, recorded as CTX_CEILING).
A few rows are flagged and a few quality checks fail. Usage:

    python scripts/make_example_report.py [OUT.html]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from xpu_llm_bench import launch
from xpu_llm_bench.html_report import render_html
from xpu_llm_bench.results import (
    AttentionSnapshot,
    GraphSnapshot,
    HardwareSnapshot,
    QualityResult,
    RungResult,
    RunManifest,
    SpecDecodeSnapshot,
)

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
OUT_DIR = FIX / "example_run"
RUN_ID = "example-synthetic-20260101T000000Z"
MAX_MODEL_LEN = 262144
# auto ladder with auto_step_tokens 32768 (the default step is 65536)
CTXS = [8192, 32768, 65536, 131072, 163840, 196608, 229376, 260096]
USERS = [1, 2, 4, 8]
OUTPUT = 256
KV_CAPACITY = 720896  # tokens (synthetic "KV cache size" boot line)
PROMPTS = (("long_form", 49.0), ("code_edits", 52.0), ("multi_turn_chat", 46.0))
K = 6


def fits(users: int, ctx: int) -> bool:
    return users * (ctx + OUTPUT) <= KV_CAPACITY


def ceilings() -> dict[str, dict]:
    out = {}
    for pt, _ in PROMPTS:
        for users in USERS:
            ok = [c for c in CTXS if fits(users, c)]
            stop = next((c for c in CTXS if not fits(users, c)), None)
            if stop is None:
                continue
            out[f"{pt}|closed_all_decoding|{users}"] = {
                "family": pt, "load_mode": "closed_all_decoding", "users": users,
                "max_ok_ctx": max(ok), "stopped_at": stop, "reason": "kv_capacity",
                "detail": f"KV need {users * (stop + OUTPUT):,} tokens ({users} x {stop:,} + output) exceeds the "
                          f"server KV capacity {KV_CAPACITY:,} tokens (boot_log); not sent",
                "rung_id": f"{pt}:closed:{users}u:{stop}:C",
            }
    return out


def _quality(pt: str, users: int, ctx: int) -> tuple[QualityResult, bool]:
    q = QualityResult(needle_pass=True, format_ok=True)
    if pt == "code_edits":
        q.code_edit_pass = True
    if users == 1:
        q.greedy_ref = "match"
        if ctx <= 131072:
            q.logprob_ok = True
    # a few synthetic failures, so the pass counts are not all-green
    if pt == "long_form" and users == 2 and ctx == 196608:
        q.needle_pass = False
    if pt == "code_edits" and users == 8 and ctx == 32768:
        q.code_edit_pass = False
    if pt == "multi_turn_chat" and users == 1 and ctx == 229376:
        q.greedy_ref = "diverged"
    if pt == "long_form" and users == 4 and ctx == 8192:
        q.format_ok = False
    failed = q.needle_pass is False or q.code_edit_pass is False or q.format_ok is False or q.greedy_ref == "diverged"
    return q, failed


def rows() -> list[RungResult]:
    out: list[RungResult] = []
    for pt, base_tps in PROMPTS:
        for ui, users in enumerate(USERS):
            for ctx in CTXS:
                phase = "S" if users == 1 else "C"
                rid = f"{pt}:closed:{users}u:{ctx}:{phase}"
                if not fits(users, ctx):
                    c = ceilings()[f"{pt}|closed_all_decoding|{users}"]
                    out.append(RungResult(
                        run_id=RUN_ID, rung_id=rid, phase=phase, users=users, ctx_tokens=[], ctx_per_user=ctx,
                        prompt_type=pt, load_mode="closed_all_decoding", prompt_seed=20260101, k=K,
                        valid="invalid", reason=c["detail"], exit_reason="CTX_CEILING", flags=["CTX_CEILING"],
                        started_utc="2026-01-01T00:00:00+00:00", started_local="2026-01-01T00:00:00+00:00",
                    ))
                    break  # this user count stops here; the next user count continues
                lg = math.log2(ctx / 8192)
                ctx_f = 1.0 / (1.0 + 0.16 * lg + 0.025 * lg * lg)
                user_f = 1.0 / (1.0 + 0.28 * ui)
                tps = base_tps * ctx_f * user_f
                tok_per_step = 3.3 - 0.12 * lg - 0.1 * ui
                steps = tps / tok_per_step
                accepted = tok_per_step - 1
                flags: list[str] = []
                hit = accepted / K
                if pt == "code_edits" and users == 4 and ctx == 65536:
                    flags.append("QSA_FALLBACK")
                if pt == "multi_turn_chat" and users == 2 and ctx == 32768:
                    flags.append("NOT_AS_LABELED")
                if pt == "multi_turn_chat" and users == 8 and ctx == 8192:
                    flags.append("HIT_RATE_UNKNOWN")
                    hit = None
                q, failed = _quality(pt, users, ctx)
                if failed:
                    flags.append("QUALITY_FAIL")
                prefill = 1600 / (1 + 0.22 * lg) / (1 + 0.1 * ui)
                fill = [ctx - 37 * (s % 3) for s in range(users)]  # filled to within 0.5% of the target
                out.append(
                    RungResult(
                        run_id=RUN_ID,
                        rung_id=rid,
                        phase=phase,
                        users=users,
                        ctx_tokens=fill,
                        ctx_per_user=ctx,
                        ctx_fill_pct=round(100.0 * sum(fill) / len(fill) / ctx, 2),
                        prompt_type=pt,
                        load_mode="closed_all_decoding",
                        prompt_seed=20260101,
                        k=K,
                        attention_path="fallback" if "QSA_FALLBACK" in flags else "qsa",
                        capture_sizes=[8, 16, 32, 64],
                        decode_batch_tokens=users * (K + 1),
                        kv_format="example_kv",
                        decode_tok_s_per_user=round(tps, 2),
                        decode_tok_s_total=round(tps * users, 2),
                        steps_s_per_user=round(steps, 2),
                        steps_s_total=round(steps * users, 2),
                        tokens_per_step=round(tok_per_step, 3),
                        accepted_drafts_per_step=round(accepted, 3),
                        hit_rate=None if hit is None else round(hit, 4),
                        prefill_tok_s=round(prefill, 1),
                        ttft_ms=round(min(ctx, 32768) / prefill * 1000 * (1 + 0.4 * ui), 1),
                        tpot_ms=round(1000 / tps, 2),
                        itl_p50_ms=round(1000 / steps, 2),
                        itl_p99_ms=round(1600 / steps, 2),
                        kv_cache_usage_max_pct=round(min(99.0, 100.0 * users * (ctx + OUTPUT) / KV_CAPACITY), 1),
                        mem_available_min_gib=round(9.5 - 0.6 * ui - 0.4 * lg, 2),
                        quality=q,
                        flags=flags,
                        valid="warning" if {"QSA_FALLBACK", "NOT_AS_LABELED"} & set(flags) else "valid",
                        started_utc="2026-01-01T00:00:00+00:00",
                        started_local="2026-01-01T00:00:00+00:00",
                    )
                )
    return out


def manifest() -> RunManifest:
    info = json.loads((FIX / "docker_inspect_example.json").read_text())[0]
    rec = launch.capture(
        inspect_info=info,
        container="vllm-bench-example",
        boot_log=(FIX / "boot_log_example.txt").read_text(),
        endpoint="http://127.0.0.1:8102/v1",
    )
    rec.update(captured_utc="2026-01-01T00:00:45+00:00", captured_local="2026-01-01T00:00:45+00:00",
               display_timezone="UTC")
    return RunManifest(
        run_id=RUN_ID,
        spec_name="example-synthetic",
        profile="full",
        started_utc="2026-01-01T00:00:45+00:00",
        started_local="2026-01-01T00:00:45+00:00",
        finished_utc="2026-01-01T01:10:00+00:00",
        finished_local="2026-01-01T01:10:00+00:00",
        display_timezone="UTC",
        status="completed",
        image="example-registry/vllm-xpu:example",
        hardware=HardwareSnapshot(host="example-host", gpu_model="example GPU", gpu_count=2, tensor_parallel=2),
        attention=AttentionSnapshot(requested="qsa", actual="qsa",
                                    evidence_lines=["(synthetic) attention backend line from the boot log"]),
        kv_format="example_kv",
        graph=GraphSnapshot(mode="FULL_DECODE_ONLY", capture_sizes=[8, 16, 32, 64],
                            graphs_on=True, graphs_basis="boot_log", graphs_declared=True),
        speculative=SpecDecodeSnapshot(method="mtp", num_speculative_tokens=K),
        k=K,
        prompt_seed=20260101,
        fingerprint={"container": "vllm-bench-example", "image_tag": "example-registry/vllm-xpu:example",
                     "build_fingerprint": "synthetic-example-fingerprint"},
        launch=rec,
        notes=["SYNTHETIC example data: illustrates the report layout only."],
        max_model_len=MAX_MODEL_LEN,
        max_model_len_source="/v1/models",
        context_rungs=CTXS,
        context_note="auto (synthetic, auto_step_tokens 32768): up to 260096 (max_model_len=262144 minus 1280 headroom)",
        kv_capacity_tokens=KV_CAPACITY,
        kv_capacity_source="boot_log",
        ctx_ceilings=ceilings(),
        pre_run_load={"checked": True, "probe_s": 2.0, "requests_running": 0.0, "requests_waiting": 0.0,
                      "generation_tokens_in_probe": 0.0, "kv_cache_usage_pct": 0.0, "server_busy": False,
                      "gpu_loaded": False, "loaded": False},
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rs = rows()
    (OUT_DIR / "results.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in rs), encoding="utf-8")
    man = manifest()
    man.save(OUT_DIR / "manifest.json")
    html = render_html(rs, man, title="xpu-llm-bench example report (SYNTHETIC data, layout illustration only)")
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out:
        out.write_text(html, encoding="utf-8")
        print(f"wrote {out}")
    print(f"wrote {OUT_DIR}/results.jsonl ({len(rs)} rows) and manifest.json")


if __name__ == "__main__":
    main()
