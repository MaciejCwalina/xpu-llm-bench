"""F4: the planner's RAM/GPU check can actually fail, and uses measured RSS when given."""

from pathlib import Path

from test_runner_harness import _runner, _spec
from xpu_llm_bench.cli import main
from xpu_llm_bench.planner import feasibility, make_plan, measured_host_gib_from_rows
from xpu_llm_bench.procmem import ProcSnapshot
from xpu_llm_bench.results import RungResult, RunManifest
from xpu_llm_bench.schedule import build_schedule
from xpu_llm_bench.spec import RunSpec


def _s(**over) -> RunSpec:
    data = {"name": "f", "profile": "custom", "ladder": {"context_rungs_tokens": [8192, 131072],
                                                          "concurrency_steps": [1, 2, 4, 8]}}
    for k, v in over.items():
        data[k] = {**data.get(k, {}), **v} if isinstance(v, dict) else v
    return RunSpec.model_validate(data)


def test_default_specs_fit() -> None:
    f = feasibility(_s(), build_schedule(_s()))
    assert f.ok and f.gpu_status == "ok" and f.host_status == "ok" and f.host_basis == "estimate"


def test_host_ram_estimate_can_fail() -> None:
    spec = _s(planner={"assumed_host_process_gib": 24.0})
    f = feasibility(spec, build_schedule(spec))
    assert f.host_status == "fail" and not f.ok
    assert any(n.startswith("FAIL host RAM") for n in f.notes)
    plan = make_plan(spec)
    assert any(i.startswith("FAIL host RAM") for i in plan.issues)


def test_host_ram_tight_warns() -> None:
    spec = _s(planner={"assumed_host_process_gib": 22.0})  # 23 GiB usable, < 2 GiB headroom
    f = feasibility(spec, build_schedule(spec))
    assert f.host_status == "tight" and f.ok


def test_measured_rss_overrides_estimate() -> None:
    spec = _s()
    f = feasibility(spec, build_schedule(spec), measured_host_gib=23.5)
    assert f.host_basis == "measured" and f.host_status == "fail"


def test_host_kv_share_counts() -> None:
    spec = _s(planner={"host_kv_fraction": 1.0, "assumed_kv_bytes_per_token": 200_000})
    f = feasibility(spec, build_schedule(spec))
    assert f.host_status == "fail"


def test_gpu_can_fail_and_warn() -> None:
    big = _s(planner={"assumed_kv_bytes_per_token": 200_000})
    assert feasibility(big, build_schedule(big)).gpu_status == "fail"
    warm = _s(planner={"assumed_weight_gib": 36.0})
    assert feasibility(warm, build_schedule(warm)).gpu_status == "tight"


def _row(rss: dict) -> RungResult:
    return RungResult(run_id="r", rung_id="x", users=1, ctx_tokens=[1], ctx_per_user=1, prompt_type="code_edits",
                      load_mode="closed_all_decoding", rss_kib=rss, started_utc="", started_et="")


def test_measured_from_rows() -> None:
    rows = [_row({"api_server": 1024 * 1024, "engine": 2 * 1024 * 1024}), _row({"engine": 1024 * 1024})]
    assert measured_host_gib_from_rows(rows) == 3.0
    assert measured_host_gib_from_rows([_row({})]) is None


def test_plan_cli_exits_1_on_fail_and_reads_measured(tmp_path: Path, capsys) -> None:
    spec = tmp_path / "s.yaml"
    spec.write_text("name: s\nprofile: custom\nladder: {context_rungs_tokens: [8192], concurrency_steps: [1]}\n")
    assert main(["plan", str(spec)]) == 0
    res = tmp_path / "results.jsonl"
    res.write_text(_row({"engine": 24 * 1024 * 1024}).model_dump_json() + "\n")
    assert main(["plan", str(spec), "--measured", str(res)]) == 1
    out = capsys.readouterr().out
    assert "FAIL host RAM" in out and "(measured)" in out


def test_live_preflight_refuses_on_measured_rss(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"])
    r = _runner(spec, tmp_path / "o", mock_server, tmp_path)
    r._proc_fn = lambda: ProcSnapshot(ours={"engine": 30 * 1024 * 1024})
    import pytest

    from xpu_llm_bench.runner import RunRefused

    with pytest.raises(RunRefused, match="FAIL host RAM"):
        r.run()
    assert mock_server["state"].chat_requests == 0
    assert RunManifest.load(tmp_path / "o" / "manifest.json").status == "refused"
