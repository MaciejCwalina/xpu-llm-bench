"""F1: every shipped spec, under every profile and variant, plans the
progressive ladder: session 1 grows, then users join. Never shrink, never refill."""

from pathlib import Path

import pytest

from xpu_llm_bench.planner import GRID_WARNING, make_plan
from xpu_llm_bench.schedule import build_schedule, validate_progressive
from xpu_llm_bench.spec import RunSpec, apply_variant, load_spec

SPECS = sorted((Path(__file__).resolve().parents[1] / "specs").glob("*.yaml"))


def _expanded():
    for path in SPECS:
        for profile in (None, "quick", "full"):
            spec = load_spec(path, profile=profile)
            yield f"{path.stem}:{profile or 'file'}", spec
            for var in spec.variants:
                yield f"{path.stem}:{profile or 'file'}:{var.name}", apply_variant(spec, var)


CASES = list(_expanded())


def test_specs_found() -> None:
    assert len(SPECS) >= 8


@pytest.mark.parametrize("name,spec", CASES, ids=[c[0] for c in CASES])
def test_spec_plans_monotonic_growth(name: str, spec: RunSpec) -> None:
    assert spec.ladder.shape == "session1_then_join", name
    rungs = build_schedule(spec)
    assert rungs
    assert validate_progressive(rungs) == []
    top = max(spec.ladder.context_rungs_tokens)
    groups: dict[tuple[str, str], list] = {}
    for r in rungs:
        groups.setdefault((r.family, r.load_mode), []).append(r)
    for key, seq in groups.items():
        assert all(r.phase in {"S", "C"} for r in seq), key
        # The first rung is one user: the run never starts wide.
        assert seq[0].users == 1 and seq[0].phase == "S", key
        phases = [r.phase for r in seq]
        assert phases == sorted(phases, key=lambda p: p != "S"), f"{key}: S rungs must all precede C rungs"
        s_rungs = [r for r in seq if r.phase == "S"]
        c_rungs = [r for r in seq if r.phase == "C"]
        # Session 1 grows through strictly increasing contexts.
        assert all(r.users == 1 for r in s_rungs)
        s_ctx = [r.target_for(1) for r in s_rungs]
        assert s_ctx == sorted(set(s_ctx)) and s_ctx[-1] == top, key
        # Users then join one rung at a time; session 1 stays at the top context.
        c_users = [r.users for r in c_rungs]
        assert c_users == sorted(set(c_users)), key
        for r in c_rungs:
            assert r.target_for(1) == top, key
        # Never shrink, never refill: a session is only grown to a NEW high-water mark.
        high: dict[int, int] = {}
        live_count = 0
        for r in seq:
            assert len(r.live_sessions) >= live_count, f"{key}: live sessions dropped at {r.rung_id}"
            live_count = len(r.live_sessions)
            for sid in r.live_sessions:
                target = r.target_for(sid)
                assert target >= high.get(sid, 0), f"{key}: session {sid} shrinks at {r.rung_id}"
            for sid in r.grow_sessions:
                assert r.target_for(sid) > high.get(sid, 0), f"{key}: session {sid} refilled at {r.rung_id}"
            for sid in r.live_sessions:
                high[sid] = max(high.get(sid, 0), r.target_for(sid))


def test_grid_is_refused_unless_explicit() -> None:
    base = {"name": "g", "profile": "custom", "ladder": {"shape": "grid"}}
    with pytest.raises(ValueError, match="allow_grid"):
        RunSpec.model_validate(base)
    spec = RunSpec.model_validate({**base, "ladder": {"shape": "grid", "allow_grid": True}})
    plan = make_plan(spec)
    assert plan.issues[0] == GRID_WARNING


def test_custom_profile_defaults_to_progressive() -> None:
    spec = RunSpec.model_validate({"name": "c", "profile": "custom"})
    assert spec.ladder.shape == "session1_then_join"
    rungs = build_schedule(spec)
    assert rungs[0].users == 1 and rungs[0].ctx_per_user == min(spec.ladder.context_rungs_tokens)
