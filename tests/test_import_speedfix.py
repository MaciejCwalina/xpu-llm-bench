"""F7: speed-fix ladder rows (anonymized real sample) import with every field
mapped; hit_rate is derived, never copied from the acceptance counter; compare
pairs rows on users + context + prompt type."""

import json
from pathlib import Path

import pytest

from xpu_llm_bench.import_legacy import import_legacy_file, import_row
from xpu_llm_bench.report import compare_runs, format_compare
from xpu_llm_bench.results import read_jsonl

FIX = Path(__file__).resolve().parent / "fixtures" / "legacy_speedfix_rungs.jsonl"
RAW = [json.loads(line) for line in FIX.read_text().splitlines() if line.strip()]


def _rows(tmp_path: Path, **kw):
    dest = tmp_path / "imp.jsonl"
    assert import_legacy_file(FIX, dest, run_id="speedfix", **kw) == len(RAW)
    return read_jsonl(dest)


def test_fields_mapped(tmp_path: Path) -> None:
    rows = _rows(tmp_path, prompt_type="code_edits")
    r0, r3 = rows[0], rows[3]
    raw0, raw3 = RAW[0], RAW[3]
    assert r0.users == 1 and r0.ctx_per_user == 8192 and r0.phase == "S"
    assert r3.users == 4 and r3.ctx_per_user == 8192 and r3.phase == "C"
    assert r3.ctx_tokens == raw3["prompt_tokens"]
    assert r3.decode_tok_s_total == pytest.approx(raw3["total_toks"])
    assert r3.decode_tok_s_per_user == pytest.approx(sum(raw3["rates"]) / 4)
    assert r3.steps_s_per_user == pytest.approx(sum(raw3["per_user_steps_s"]) / 4)
    assert r3.steps_s_total == pytest.approx(sum(raw3["per_user_steps_s"]))
    assert r3.ttft_ms == pytest.approx(max(raw3["first_token_s"].values()) * 1000)
    assert r3.mem_available_min_gib == pytest.approx(raw3["mem"]["min_kb"] / 1024 / 1024)
    assert r3.tokens_per_step == pytest.approx(raw3["tokens_per_step"])
    assert r3.drafts == raw3["drafts_delta"] and r3.draft_tokens == raw3["draft_tokens_delta"]
    assert r3.accepted_drafts == raw3["accepted_tokens_delta"]
    assert r3.k == 6 and r3.prompt_type == "code_edits"
    assert r0.prefill_tok_s == pytest.approx(raw0["prefill_tok_s"])
    assert all(r.valid == "valid" and "CONVERTED" in r.flags for r in rows)


def test_hit_rate_from_tokens_per_step_not_counter(tmp_path: Path) -> None:
    for row, raw in zip(_rows(tmp_path), RAW, strict=True):
        assert row.hit_rate == pytest.approx((raw["tokens_per_step"] - 1) / 6)
        assert row.acceptance_rate_vllm == pytest.approx(raw["acceptance"])
    # A broken counter must not leak into hit_rate.
    bad = dict(RAW[1], acceptance=1.07, acceptance_rate=1.07)
    row = import_row(bad)
    assert row.hit_rate == pytest.approx((RAW[1]["tokens_per_step"] - 1) / 6)
    assert row.acceptance_rate_vllm == pytest.approx(1.07)


def test_hit_rate_from_delivered_and_drafts_when_no_tokens_per_step() -> None:
    raw = {k: v for k, v in RAW[0].items() if k not in {"tokens_per_step", "mean_accepted_per_step"}}
    row = import_row(raw)
    delivered = raw["total_toks"] * raw["overlap_s"]
    assert row.hit_rate == pytest.approx((delivered / raw["drafts_delta"] - 1) / 6, rel=1e-6)


def test_counter_only_row_leaves_hit_rate_empty_and_flags() -> None:
    row = import_row({"label": "x", "users": 1, "ctx": 8192, "acceptance": 0.6})
    assert row.hit_rate is None and "HIT_RATE_UNKNOWN" in row.flags
    assert row.acceptance_rate_vllm == 0.6


def test_errors_make_row_invalid() -> None:
    row = import_row(dict(RAW[0], errors={"2": "timeout"}, ok=False, status="error"))
    assert row.valid == "invalid" and any("timeout" in e for e in row.errors)


def test_compare_keys_on_users_ctx_prompt_type(tmp_path: Path) -> None:
    a = _rows(tmp_path)
    b = _rows(tmp_path)
    pairs = compare_runs(a, b)
    # 5 rows: (1u,8K) twice (two runs merged in the sample), 2u, 3u, 4u -> no collapse.
    assert len(pairs) == len(a)
    assert {p.key[:3] for p in pairs} == {(1, 8192, "unknown"), (2, 8192, "unknown"),
                                          (3, 8192, "unknown"), (4, 8192, "unknown")}
    # Two imported rows pair up, but the pair is caveated, never a clean A/B.
    assert all(len(p.reasons) == 1 and "both rows imported" in p.reasons[0] for p in pairs)
    assert not any("flagged CONVERTED" in r for p in pairs for r in p.reasons)
    text = format_compare(pairs)
    assert "| 4 | 8192 | unknown |" in text


def test_compare_prompt_type_and_phase_mismatch_is_flagged(tmp_path: Path) -> None:
    a = _rows(tmp_path, prompt_type="code_edits")[:1]
    b = [a[0].model_copy(update={"phase": "C", "flags": []})]
    pairs = compare_runs(a, b)
    assert len(pairs) == 1 and any("phase mismatch" in r for r in pairs[0].reasons)
    c = [a[0].model_copy(update={"prompt_type": "long_form"})]
    assert len(compare_runs(a, c)) == 2  # different prompt types never pair
