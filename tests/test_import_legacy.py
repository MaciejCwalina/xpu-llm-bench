from pathlib import Path

from xpu_llm_bench.import_legacy import MTP6_HIT_RATE_CORRECTION, import_legacy_file
from xpu_llm_bench.results import read_jsonl

FIX = Path(__file__).resolve().parent / "fixtures"


def test_import_maps_fields(tmp_path: Path) -> None:
    dest = tmp_path / "out.jsonl"
    n = import_legacy_file(FIX / "legacy_rungs.jsonl", dest, run_id="old")
    assert n == 3
    rows = read_jsonl(dest)
    assert rows[0].users == 1
    assert rows[0].ctx_per_user == 8192
    assert rows[0].decode_tok_s_total == 60.0
    assert rows[0].hit_rate == 0.48
    assert rows[0].ttft_ms == 1500
    assert abs(rows[0].mem_available_min_gib - 3.0) < 1e-6
    assert rows[1].users == 2
    assert rows[1].ctx_per_user == 8192
    assert rows[2].valid == "invalid"
    assert rows[2].errors


def test_mtp6_correction(tmp_path: Path) -> None:
    dest = tmp_path / "corr.jsonl"
    import_legacy_file(
        FIX / "legacy_rungs.jsonl", dest, correct_mtp6_hit_rate=True, run_id="old"
    )
    rows = read_jsonl(dest)
    assert rows[0].hit_rate == 0.48 * MTP6_HIT_RATE_CORRECTION
    assert abs(MTP6_HIT_RATE_CORRECTION - 7 / 12) < 1e-12
