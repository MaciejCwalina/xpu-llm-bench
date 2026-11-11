"""Context ladder to the server's max context, per-user-count ceilings,
exact rung fill, and the pre-run load check (all against the mock server)."""

from pathlib import Path

import pytest

from test_runner_harness import _runner, _spec
from xpu_llm_bench.cli import main
from xpu_llm_bench.results import RunManifest, read_jsonl
from xpu_llm_bench.runner import RunRefused, ceiling_reason, kv_capacity_from_metrics
from xpu_llm_bench.spec import LadderSpec, resolve_contexts


def _ladder(**over):
    base = {"context_rungs_tokens": [256, 512, 768], "concurrency_steps": [1, 2, 4], "output_tokens_per_rung": 64}
    return {**base, **over}


# -- auto ladder ---------------------------------------------------------------


def test_auto_ladder_resolution() -> None:
    lad = LadderSpec()
    assert resolve_contexts(lad, 262144)[0] == [8192, 32768, 65536, 131072, 196608, 260096]
    assert resolve_contexts(lad, 131072)[0] == [8192, 32768, 65536, 129024]
    assert resolve_contexts(lad, 1048576)[0][-3:] == [262144, 524288, 1046528]
    # unknown server max: shows up to 131K and says so
    ctxs, note = resolve_contexts(lad, None)
    assert ctxs == [8192, 32768, 65536, 131072] and "not known yet" in note
    # explicit list and max_context still win
    assert resolve_contexts(LadderSpec(max_context=200000), 262144)[0][-1] == 199680
    pinned = LadderSpec(context_rungs_tokens=[8192, 131072, 262144])
    assert resolve_contexts(pinned, 196608)[0] == [8192, 131072]
    assert resolve_contexts(LadderSpec(auto_step_mode="double"), 600000)[0][4:] == [262144, 524288, 598016]


def test_live_run_resolves_auto_from_v1_models(mock_server, tmp_path: Path) -> None:
    mock_server["state"].max_model_len = 1100
    spec = _spec(
        mock_server["base"],
        ladder=_ladder(context_rungs_tokens="auto", auto_base_tokens=[256, 512], auto_step_tokens=256,
                       auto_step_mode="add", ctx_headroom_tokens=64, concurrency_steps=[1]),
    )
    man = _runner(spec, tmp_path / "o", mock_server, tmp_path).run()
    assert man.status == "completed"
    assert man.max_model_len == 1100 and man.max_model_len_source == "/v1/models"
    assert man.context_rungs == [256, 512, 768, 1036]
    rows = read_jsonl(tmp_path / "o" / "results.jsonl")
    assert [r.ctx_per_user for r in rows] == [256, 512, 768, 1036]


# -- ceilings ------------------------------------------------------------------


def test_kv_capacity_ceiling_stops_one_user_count_and_run_continues(mock_server, tmp_path: Path) -> None:
    state = mock_server["state"]
    state.cache_blocks = (150, 16)  # 2400 tokens of KV
    spec = _spec(mock_server["base"], ladder=_ladder())
    man = _runner(spec, tmp_path / "o", mock_server, tmp_path).run()
    assert man.status == "completed"
    assert man.kv_capacity_tokens == 2400 and man.kv_capacity_source == "metrics cache_config_info"
    rows = read_jsonl(tmp_path / "o" / "results.jsonl")
    by_id = {r.rung_id: r for r in rows}
    # 1 and 2 users reach the top; 4 users stop at 768 (4 x 768 + output > 2400)
    assert by_id["multi_turn_chat:closed:2u:768:C"].valid != "invalid"
    ceil = by_id["multi_turn_chat:closed:4u:768:C"]
    assert ceil.exit_reason == "CTX_CEILING" and "CTX_CEILING" in ceil.flags and "not sent" in ceil.reason
    c = man.ctx_ceilings["multi_turn_chat|closed_all_decoding|4"]
    assert c["max_ok_ctx"] == 512 and c["stopped_at"] == 768 and c["reason"] == "kv_capacity"
    assert set(man.ctx_ceilings) == {"multi_turn_chat|closed_all_decoding|4"}


def test_server_context_rejection_is_a_ceiling_not_an_abort(mock_server, tmp_path: Path) -> None:
    mock_server["state"].reject_above = 600
    spec = _spec(mock_server["base"], ladder=_ladder(concurrency_steps=[1, 2]))
    man = _runner(spec, tmp_path / "o", mock_server, tmp_path).run()
    assert man.status == "completed", man.abort_reason
    rows = read_jsonl(tmp_path / "o" / "results.jsonl")
    ids = [r.rung_id for r in rows]
    # 1u stops at 768, then 2 users still run 256 and 512 and stop at 768 too
    assert ids == [
        "multi_turn_chat:closed:1u:256:S", "multi_turn_chat:closed:1u:512:S", "multi_turn_chat:closed:1u:768:S",
        "multi_turn_chat:closed:2u:256:C", "multi_turn_chat:closed:2u:512:C", "multi_turn_chat:closed:2u:768:C",
    ]
    assert "CTX_CEILING" in rows[2].flags and "CTX_CEILING" in rows[5].flags
    assert {v["reason"] for v in man.ctx_ceilings.values()} == {"server_rejected_context"}
    assert man.ctx_ceilings["multi_turn_chat|closed_all_decoding|1"]["max_ok_ctx"] == 512


def test_stop_at_ceiling_false_keeps_the_old_abort(mock_server, tmp_path: Path) -> None:
    mock_server["state"].cache_blocks = (150, 16)
    spec = _spec(mock_server["base"], ladder=_ladder(stop_at_ceiling=False))
    man = _runner(spec, tmp_path / "o", mock_server, tmp_path).run()
    assert not man.ctx_ceilings


def test_ceiling_reason_and_kv_metric_parsing() -> None:
    from xpu_llm_bench.results import RungResult

    def row(**kw):
        base = dict(run_id="r", rung_id="x", users=1, ctx_tokens=[1], ctx_per_user=1, prompt_type="code_edits",
                    load_mode="closed_all_decoding", started_utc="", started_local="", valid="invalid")
        return RungResult(**{**base, **kw})

    assert ceiling_reason(row(exit_reason="RAM_FLOOR")) == "ram_floor"
    assert ceiling_reason(row(exit_reason="OOM")) == "oom"
    assert ceiling_reason(row(exit_reason="CLIENT_ERROR", errors=["This model's maximum context length is 8"])) == (
        "server_rejected_context")
    assert ceiling_reason(row(exit_reason="ENGINE_ERROR", errors=["boom"])) is None
    assert ceiling_reason(row(valid="valid", exit_reason="OOM")) is None
    text = 'vllm:cache_config_info{block_size="64",cache_dtype="auto",num_gpu_blocks="4096"} 1.0\n'
    assert kv_capacity_from_metrics(text) == 262144
    assert kv_capacity_from_metrics("vllm:num_requests_running 0\n") is None


# -- exact fill ----------------------------------------------------------------


def test_rungs_fill_to_target_with_the_server_tokenizer(mock_server, tmp_path: Path) -> None:
    # A bad chars-per-token guess used to under-fill (an "8K" rung at 7,113 tokens).
    spec = _spec(mock_server["base"], prompts={"types": ["multi_turn_chat"], "seed": 11, "chars_per_token_guess": 2.5},
                 ladder=_ladder(context_rungs_tokens=[2048, 4096], concurrency_steps=[1, 2]))
    man = _runner(spec, tmp_path / "o", mock_server, tmp_path).run()
    assert man.status == "completed"
    rows = read_jsonl(tmp_path / "o" / "results.jsonl")
    for r in rows:
        assert r.ctx_fill_pct is not None and abs(r.ctx_fill_pct - 100) <= 1.0, (r.rung_id, r.ctx_tokens)
        assert "CTX_FILL_OFF" not in r.flags


# -- pre-run load ----------------------------------------------------------------


def test_pre_existing_load_warns_asks_and_flags(mock_server, tmp_path: Path, capsys) -> None:
    state = mock_server["state"]
    state.running = 2.0  # someone else's requests on the server
    asked: list[str] = []
    spec = _spec(mock_server["base"], ladder=_ladder(context_rungs_tokens=[256], concurrency_steps=[1]))
    man = _runner(spec, tmp_path / "o", mock_server, tmp_path,
                  confirm_fn=lambda msg: asked.append(msg) or True).run()
    assert len(asked) == 1 and "server busy: 2 running" in asked[0]
    assert man.pre_run_load["server_busy"] and man.pre_run_load["confirmed"]
    rows = read_jsonl(tmp_path / "o" / "results.jsonl")
    assert rows and all("PRE_EXISTING_LOAD" in r.flags and r.valid == "warning" for r in rows)
    assert "pre-existing load" in capsys.readouterr().err

    state.running = 3.0
    before = state.chat_requests
    with pytest.raises(RunRefused, match="declined"):
        _runner(spec, tmp_path / "o2", mock_server, tmp_path, confirm_fn=lambda msg: False).run()
    assert state.chat_requests == before

    state.running = 1.0
    man3 = _runner(spec, tmp_path / "o3", mock_server, tmp_path, assume_yes=True,
                   confirm_fn=lambda msg: pytest.fail("--yes must not ask")).run()
    assert man3.status == "completed" and man3.pre_run_load["loaded"]


def test_idle_server_is_not_flagged(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"], ladder=_ladder(context_rungs_tokens=[256], concurrency_steps=[1]))
    man = _runner(spec, tmp_path / "o", mock_server, tmp_path,
                  confirm_fn=lambda msg: pytest.fail("idle server must not ask")).run()
    assert man.pre_run_load["checked"] and not man.pre_run_load["loaded"]
    assert not any("PRE_EXISTING_LOAD" in r.flags for r in read_jsonl(tmp_path / "o" / "results.jsonl"))


# -- report on an empty run ------------------------------------------------------


def test_report_on_refused_or_empty_run_says_why(tmp_path: Path, capsys) -> None:
    run = tmp_path / "run"
    run.mkdir()
    man = RunManifest(run_id="r1", spec_name="s", profile="quick", started_utc="", started_local="",
                      hardware={}, status="refused", abort_reason="operator declined to run with pre-existing load")
    man.save(run / "manifest.json")
    assert main(["report", str(run)]) == 1
    err = capsys.readouterr().err
    assert "run r1 was refused before any rung ran: operator declined" in err and "Errno" not in err
    assert main(["report", str(tmp_path / "nothing")]) == 2
    assert "no results.jsonl and no manifest.json" in capsys.readouterr().err
