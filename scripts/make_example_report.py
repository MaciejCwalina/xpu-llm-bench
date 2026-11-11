"""Build the synthetic example run in tests/fixtures/example_run and render it.

Every value is SYNTHETIC and only illustrates the report layout: a ladder of
1/2/4/8 users x 8K/32K/64K/131K context for two prompt types, with a few
flagged rows. Usage:

    python scripts/make_example_report.py [OUT.html]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from xpu_llm_bench import launch
from xpu_llm_bench.html_report import render_html
from xpu_llm_bench.results import (
    AttentionSnapshot,
    GraphSnapshot,
    HardwareSnapshot,
    RungResult,
    RunManifest,
    SpecDecodeSnapshot,
)

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
OUT_DIR = FIX / "example_run"
RUN_ID = "example-synthetic-20260101T000000Z"
CTXS = [8192, 32768, 65536, 131072]
USERS = [1, 2, 4, 8]
K = 6


def rows() -> list[RungResult]:
    out: list[RungResult] = []
    for pt, base_tps in (("code_edits", 52.0), ("multi_turn_chat", 46.0)):
        for ui, users in enumerate(USERS):
            for ci, ctx in enumerate(CTXS):
                if users == 8 and ctx == 131072:
                    continue  # does not fit: no row, so the 8-user line stops at 64K
                ctx_f = 1.0 / (1.0 + 0.22 * ci + 0.05 * ci * ci)
                user_f = 1.0 / (1.0 + 0.28 * ui)
                tps = base_tps * ctx_f * user_f
                tok_per_step = 3.3 - 0.25 * ci - 0.1 * ui
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
                out.append(
                    RungResult(
                        run_id=RUN_ID,
                        rung_id=f"{pt}:closed:{users}u:{ctx}:{'S' if users == 1 else 'C'}",
                        phase="S" if users == 1 else "C",
                        users=users,
                        ctx_tokens=[ctx] * users,
                        ctx_per_user=ctx,
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
                        prefill_tok_s=round(1600 / (1 + 0.35 * ci) / (1 + 0.1 * ui), 1),
                        ttft_ms=round(ctx / (1600 / (1 + 0.35 * ci)) * 1000 * (1 + 0.4 * ui), 1),
                        tpot_ms=round(1000 / tps, 2),
                        itl_p50_ms=round(1000 / steps, 2),
                        itl_p99_ms=round(1600 / steps, 2),
                        kv_cache_usage_max_pct=round(min(99.0, 3.0 * users * (ctx / 8192) ** 0.8), 1),
                        mem_available_min_gib=round(9.5 - 0.6 * ui - 0.5 * ci, 2),
                        flags=flags,
                        valid="valid",
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
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rs = rows()
    (OUT_DIR / "results.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in rs), encoding="utf-8")
    man = manifest()
    man.save(OUT_DIR / "manifest.json")
    html = render_html(rs, man, title="xpu-llm-bench example report (synthetic data)")
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out:
        out.write_text(html, encoding="utf-8")
        print(f"wrote {out}")
    print(f"wrote {OUT_DIR}/results.jsonl ({len(rs)} rows) and manifest.json")


if __name__ == "__main__":
    main()
