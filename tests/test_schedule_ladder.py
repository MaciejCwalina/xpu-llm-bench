from pathlib import Path

from xpu_llm_bench.planner import make_plan
from xpu_llm_bench.schedule import build_schedule, validate_progressive
from xpu_llm_bench.spec import RunSpec, load_spec

SPECS = Path(__file__).resolve().parents[1] / "specs"


def _spec(**ladder):
    data = {
        "name": "d",
        "profile": "custom",
        "prompts": {"types": ["code_edits", "long_form", "multi_turn_chat"]},
        "ladder": {
            "shape": "session1_then_join",
            "context_rungs_tokens": [8192, 32768, 65536, 131072],
            "concurrency_steps": [1, 2, 4, 8],
            **ladder,
        },
    }
    return RunSpec.model_validate(data)


def test_session1_grows_then_sessions_join() -> None:
    rungs = [r for r in build_schedule(_spec()) if r.prompt_type == "code_edits"]
    assert validate_progressive(rungs) == []
    s_phase = [r for r in rungs if r.phase == "S"]
    c_phase = [r for r in rungs if r.phase == "C"]
    assert [r.ctx_per_user for r in s_phase] == [8192, 32768, 65536, 131072]
    assert all(r.live_sessions == (1,) and r.grow_sessions == (1,) for r in s_phase)
    # S phase completes before any session joins
    assert rungs.index(c_phase[0]) > rungs.index(s_phase[-1])
    assert [r.users for r in c_phase] == [2, 4, 8]
    # joiners are added one at a time (each grow step lists them in order)
    assert [r.grow_sessions for r in c_phase] == [(2,), (3, 4), (5, 6, 7, 8)]
    # everyone decodes: earlier sessions stay genuinely live
    assert all(r.decoding_sessions == r.live_sessions for r in c_phase)
    # session 1 stays at its high-water context; joiners at 8K
    assert c_phase[-1].ctx_targets == (131072,) + (8192,) * 7
    ids = [r.rung_id for r in rungs]
    assert len(set(ids)) == len(ids)


def test_quick_profile_is_about_ten_minutes() -> None:
    spec = load_spec(SPECS / "qsa-full-fix-mtp6.yaml", profile="quick")
    assert spec.ladder.shape == "session1_then_join"
    plan = make_plan(spec)
    # 3 prompt types x (8K, 32K growth + 2/4/8 joins) = 15 rungs
    assert len(plan.rungs) == 15
    assert {r.prompt_type for r in plan.rungs} == {"code_edits", "long_form", "multi_turn_chat"}
    assert {r.users for r in plan.rungs} == {1, 2, 4, 8}
    assert 5 <= plan.estimated_minutes <= 14
    assert not any("shrink" in i for i in plan.issues)


def test_grid_shape_still_available() -> None:
    rungs = build_schedule(_spec(shape="grid", allow_grid=True, context_rungs_tokens=[8192, 32768]))
    assert all(r.phase == "meas" for r in rungs)
    assert validate_progressive(rungs) == []
