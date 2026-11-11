from pathlib import Path

from xpu_llm_bench.planner import format_plan, make_plan
from xpu_llm_bench.spec import load_spec

SPECS = Path(__file__).resolve().parents[1] / "specs"


def test_quick_is_shorter_than_full() -> None:
    quick = make_plan(load_spec(SPECS / "qsa-full-fix-mtp6.yaml", profile="quick"))
    full = make_plan(load_spec(SPECS / "qsa-full-fix-mtp6.yaml", profile="full"))
    assert quick.estimated_seconds < full.estimated_seconds
    assert len(quick.rungs) < len(full.rungs)
    assert quick.feasibility.peak_live_tokens > 0
    text = format_plan(quick)
    assert "Plan for" in text
    assert "Peak live tokens" in text
    # Every rung is listed in order
    for rung in quick.rungs:
        assert rung.rung_id in text


def test_plan_flags_missing_issues_empty_for_default() -> None:
    plan = make_plan(load_spec(SPECS / "qsa-vs-dense.yaml"))
    # No shrink / duplicate issues
    assert not any("shrink" in i for i in plan.issues)
