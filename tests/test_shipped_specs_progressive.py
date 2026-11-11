"""F1: every shipped spec, under every profile and variant, plans the
progressive ladder: session 1 grows, then users join. Never shrink, never refill."""

from pathlib import Path

import pytest

from xpu_llm_bench.planner import GRID_WARNING, make_plan
from xpu_llm_bench.schedule import build_schedule, validate_progressive
from xpu_llm_bench.spec import RunSpec, apply_variant, ladder_contexts, load_spec, resolve_contexts

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
    ctxs = ladder_contexts(spec)
    top = ctxs[-1]
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
        # Session 1 grows through strictly increasing contexts up to the top rung.
        assert all(r.users == 1 for r in s_rungs)
        s_ctx = [r.target_for(1) for r in s_rungs]
        assert s_ctx == sorted(set(s_ctx)) and s_ctx[-1] == top, key
        # Then each user count climbs on its own: users never go down, and a
        # user count's contexts strictly increase up to the top rung, with
        # every one of its users decoding at the same context.
        c_users = [r.users for r in c_rungs]
        assert c_users == sorted(c_users), key
        by_users: dict[int, list] = {}
        for r in c_rungs:
            by_users.setdefault(r.users, []).append(r)
            assert len(r.decoding_sessions) == r.users == len(r.live_sessions), key
            assert {r.target_for(s) for s in r.live_sessions} == {r.ctx_per_user}, key
        for n, rs in by_users.items():
            cs = [r.ctx_per_user for r in rs]
            c_top = max(c for c in ctxs if c <= (spec.ladder.join_max_context_tokens or top))
            assert cs == sorted(set(cs)) and cs[-1] == c_top, (key, n)
            assert len({r.live_sessions for r in rs}) == 1, f"{key}: {n} users keep one group of sessions"
        # Never shrink, never refill: a session is only grown to a NEW high-water mark.
        high: dict[int, int] = {}
        for r in seq:
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
    assert rungs[0].users == 1 and rungs[0].ctx_per_user == min(ladder_contexts(spec))


@pytest.mark.parametrize("name,spec", CASES, ids=[c[0] for c in CASES])
def test_full_and_default_profiles_reach_max_context(name: str, spec: RunSpec) -> None:
    """No shipped spec caps the ladder at 131K: full / file profiles climb to
    the server's max_model_len; quick stays short."""
    ctxs, _ = resolve_contexts(spec.ladder, 262144)
    if spec.profile == "quick":
        assert max(ctxs) <= 32768, name
    elif spec.ladder.max_context is None:
        assert max(ctxs) > 196608, (name, ctxs)
        assert max(ctxs) <= 262144 - spec.ladder.output_tokens_per_rung
