from xpu_llm_bench.report import (
    COLUMNS,
    compare_runs,
    format_compare,
    non_comparable_reasons,
    rows_to_csv,
    rows_to_markdown,
)
from xpu_llm_bench.results import RungResult


def _row(**kwargs) -> RungResult:
    base = dict(
        run_id="r",
        rung_id="code_edits:closed:1u:8192:meas",
        users=1,
        ctx_tokens=[8192],
        ctx_per_user=8192,
        prompt_type="code_edits",
        load_mode="closed_all_decoding",
        decode_tok_s_per_user=50.0,
        decode_tok_s_total=50.0,
        steps_s_per_user=10.0,
        hit_rate=0.4,
        attention_path="qsa",
        valid="valid",
    )
    base.update(kwargs)
    return RungResult.model_validate(base)


def test_column_order_is_fixed() -> None:
    md = rows_to_markdown([_row()])
    header = md.splitlines()[0]
    order = [c.strip() for c in header.strip("|").split("|")]
    assert order == COLUMNS
    csv = rows_to_csv([_row()])
    assert csv.splitlines()[0].split(",") == COLUMNS


def test_compare_flags_non_comparable() -> None:
    a = _row()
    b = _row(
        run_id="r2",
        decode_tok_s_per_user=60.0,
        decode_tok_s_total=60.0,
        attention_path="fallback",
        prompt_type="long_form",
        eager_fallback=True,
    )
    reasons = non_comparable_reasons(a, b)
    assert any("prompt_type" in r for r in reasons)
    assert any("path" in r for r in reasons)
    assert any("fallback" in r for r in reasons)
    assert any("eager" in r for r in reasons)
    text = format_compare(compare_runs([a], [b]))
    assert "NON-COMPARABLE" in text


def test_compare_delta_when_clean() -> None:
    a = _row()
    b = _row(run_id="r2", decode_tok_s_per_user=60.0, decode_tok_s_total=60.0, hit_rate=0.5)
    pairs = compare_runs([a], [b])
    assert pairs[0].reasons == []
    assert pairs[0].deltas["decode_tok_s_per_user"] == 10.0
    assert abs(pairs[0].deltas["hit_rate"] - 0.1) < 1e-9
    table_rows = [ln for ln in format_compare(pairs).splitlines() if ln.startswith("| 1 |")]
    assert table_rows
    assert "NON-COMPARABLE" not in table_rows[0]


def test_ram_floor_flag() -> None:
    a = _row()
    b = _row(run_id="r2", valid="invalid", reason="MemAvailable ram floor hit")
    assert any("floor" in r.lower() for r in non_comparable_reasons(a, b))
