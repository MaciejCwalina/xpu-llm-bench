from xpu_llm_bench.schedule import build_schedule, validate_progressive
from xpu_llm_bench.spec import RunSpec


def _spec(**kwargs) -> RunSpec:
    data = {
        "name": "sched",
        "profile": "custom",
        "prompts": {"types": ["code_edits", "long_form"], "seed": 1},
        "ladder": {
            "context_rungs_tokens": [8192, 32768],
            "concurrency_steps": [1, 2, 4, 8],
            "load_modes": ["closed_all_decoding"],
            "output_tokens_per_rung": 32,
        },
    }
    data.update(kwargs)
    return RunSpec.model_validate(data)


def test_never_shrinks_and_covers_matrix() -> None:
    # The old grid order, explicitly opted into.
    rungs = build_schedule(
        _spec(
            ladder={
                "shape": "grid",
                "allow_grid": True,
                "context_rungs_tokens": [8192, 32768],
                "concurrency_steps": [1, 2, 4, 8],
                "load_modes": ["closed_all_decoding"],
                "output_tokens_per_rung": 32,
            }
        )
    )
    errors = validate_progressive(rungs)
    assert errors == []
    # Two prompt types × (4 concurrencies at 8K + 4 at 32K)
    assert len(rungs) == 2 * 8
    keys = {(r.prompt_type, r.users, r.ctx_per_user) for r in rungs}
    for ptype in ("code_edits", "long_form"):
        for users in (1, 2, 4, 8):
            for ctx in (8192, 32768):
                assert (ptype, users, ctx) in keys
    # Context is non-decreasing within a family for session 1
    s1 = [r.ctx_per_user for r in rungs if r.prompt_type == "code_edits" and 1 in r.live_sessions]
    assert s1 == sorted(s1)
    # At 8K we grow sessions as concurrency rises; at 32K we grow all once
    first_32 = next(r for r in rungs if r.prompt_type == "code_edits" and r.ctx_per_user == 32768)
    assert first_32.users == 8
    assert first_32.measures_prefill
    assert first_32.grow_sessions == first_32.live_sessions
    later = [r for r in rungs if r.prompt_type == "code_edits" and r.ctx_per_user == 32768 and r.users < 8]
    assert later
    assert all(not r.grow_sessions for r in later)
    # Quiesced sessions remain live
    one_at_32 = next(r for r in later if r.users == 1)
    assert len(one_at_32.live_sessions) == 8
    assert one_at_32.decoding_sessions == (1,)


def test_prompt_types_are_separate_families() -> None:
    rungs = build_schedule(_spec())
    code_ids = {s for r in rungs if r.family == "code_edits" for s in r.live_sessions}
    # Families may reuse the same numeric ids (reset is allowed across types)
    assert rungs[0].family != rungs[-1].family or rungs[0].prompt_type == rungs[-1].prompt_type
    assert code_ids
