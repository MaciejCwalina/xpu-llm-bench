"""Planner feasibility: GPU fit, and host RAM against live MemAvailable (which
already counts the booted server and every other resident process)."""

from pathlib import Path

import pytest

from test_runner_harness import _runner, _spec
from xpu_llm_bench.cli import main
from xpu_llm_bench.planner import (
    baseline_mem_available_from_rows,
    feasibility,
    local_mem_available_gib,
    make_plan,
)
from xpu_llm_bench.results import RungResult
from xpu_llm_bench.schedule import build_schedule
from xpu_llm_bench.spec import RunSpec

SPECS = Path(__file__).resolve().parents[1] / "specs"


def _s(**over) -> RunSpec:
    data = {"name": "f", "profile": "custom",
            "hardware": {"gpu_count": 2, "gpu_mem_gib": 24},
            # stop_at_ceiling false: the whole-run check (every rung must fit).
            "ladder": {"context_rungs_tokens": [8192, 131072], "concurrency_steps": [1, 2, 4, 8],
                       "stop_at_ceiling": False}}
    for k, v in over.items():
        data[k] = {**data.get(k, {}), **v} if isinstance(v, dict) else v
    return RunSpec.model_validate(data)


def _f(spec, mem=None):
    return feasibility(spec, build_schedule(spec), mem)


def test_without_a_reading_host_is_unverified_and_informational() -> None:
    f = _f(_s())
    assert f.host_status == "unverified" and f.failures == []
    from xpu_llm_bench.planner import format_plan

    out = format_plan(make_plan(_s()))
    assert "Host RAM (info only, never blocks)" in out and "no MemAvailable reading" in out


def test_host_ram_estimate_never_blocks() -> None:
    assert _f(_s(), mem=128.0).host_status == "ok"
    f = _f(_s(), mem=6.0)
    assert f.host_status == "over" and f.failures == []
    assert not any("host RAM" in n for n in f.notes)  # one Host RAM line in the plan, no duplicate note
    out = __import__("xpu_llm_bench.planner", fromlist=["format_plan"]).format_plan(make_plan(_s(), mem_available_gib=6.0))
    assert out.count("Host RAM") == 1 and "(OVER)" in out and "FAIL" not in out


def test_qsa_4_users_at_8k_estimate_is_informational() -> None:
    """Reality: QSA at 4 users / 8K joiners hit the 2 GiB MemAvailable floor,
    starting from ~4.8 GiB available. The estimate shows it but never blocks:
    the live RAM floor stops that user count (CTX_CEILING) instead."""
    from xpu_llm_bench.spec import load_spec

    q = load_spec(SPECS / "qsa-vs-dense.yaml", profile="quick")
    f = feasibility(q, build_schedule(q), 4.8)
    assert f.host_status in {"over", "tight"} and f.failures == []


def test_floor_and_per_token_rate_are_configurable(monkeypatch, tmp_path: Path) -> None:
    from xpu_llm_bench import site
    from xpu_llm_bench.spec import load_spec

    p = tmp_path / "site.yaml"
    p.write_text(
        "host: lab\nprotected_containers: [prod-a]\nforbidden_ports: [9100]\n"
        "hardware: {mem_available_floor_gib: 4.0, gpu_count: 1, gpu_mem_gib: 80}\n"
        "planner: {host_bytes_per_token: 1024}\n"
    )
    monkeypatch.setenv(site.SITE_ENV, str(p))
    site.clear_cache()
    spec_file = tmp_path / "s.yaml"
    spec_file.write_text("name: s\nprofile: custom\nladder: {context_rungs_tokens: [8192], concurrency_steps: [1, 2]}\n")
    spec = load_spec(spec_file)
    assert spec.hardware.mem_available_floor_gib == 4.0 and spec.planner.host_bytes_per_token == 1024
    assert spec.hardware.gpu_count == 1 and spec.hardware.gpu_mem_gib == 80
    f = _f(spec, mem=4.5)
    assert f.host_budget_gib == pytest.approx(0.5)


def test_host_kv_share_counts() -> None:
    spec = _s(planner={"host_kv_fraction": 1.0, "host_bytes_per_token": 0, "assumed_kv_bytes_per_token": 200_000})
    assert _f(spec, mem=20.0).host_status == "over"


def test_gpu_can_fail_warn_or_be_unknown() -> None:
    f = _f(_s(planner={"assumed_kv_bytes_per_token": 200_000}))
    assert f.gpu_status == "fail" and f.failures == []  # an estimate: warns, never blocks
    assert any(n.startswith("WARN GPU (estimate, does not block)") for n in f.notes)
    assert _f(_s(planner={"assumed_weight_gib": 36.0}, ladder={"context_rungs_tokens": [8192]})).gpu_status == "tight"
    unknown = RunSpec.model_validate({"name": "u", "profile": "custom"})
    f = _f(unknown)
    assert f.gpu_status == "unknown" and any("gpu_count / gpu_mem_gib not set" in n for n in f.notes)


def test_ceiling_estimates_per_user_count() -> None:
    from xpu_llm_bench.planner import format_plan

    spec = _s(ladder={"context_rungs_tokens": "auto", "stop_at_ceiling": True})
    plan = make_plan(spec, max_model_len=262144, kv_capacity_tokens=700_000)
    by_users = {c.users: c for c in plan.feasibility.ceilings if c.load_mode == "closed_all_decoding"
                and c.family == "code_edits"}
    assert by_users[1].stops_at is None and by_users[2].stops_at is None
    assert by_users[4].top_ok == 131072 and by_users[4].stops_at == 196608 and by_users[4].reason == "kv_capacity"
    assert by_users[8].top_ok == 65536
    out = format_plan(plan)
    assert "Context ceiling per user count" in out and "4u: up to 131K, stops at 196K (kv_capacity)" in out
    assert "Context rungs: 8K, 32K, 64K, 131K, 196K, 260K" in out


def _row(mem: float | None) -> RungResult:
    return RungResult(run_id="r", rung_id="x", users=1, ctx_tokens=[1], ctx_per_user=1, prompt_type="code_edits",
                      load_mode="closed_all_decoding", mem_available_min_gib=mem, started_utc="", started_local="")


def test_baseline_from_rows_and_local_meminfo(tmp_path: Path) -> None:
    assert baseline_mem_available_from_rows([_row(None), _row(5.5), _row(3.0)]) == 5.5
    assert baseline_mem_available_from_rows([_row(None)]) is None
    mi = tmp_path / "meminfo"
    mi.write_text("MemTotal: 26214400 kB\nMemAvailable: 4194304 kB\n")
    assert local_mem_available_gib(str(mi)) == 4.0
    assert local_mem_available_gib(str(tmp_path / "nope")) is None


def test_plan_cli_never_fails_on_estimates(tmp_path: Path, capsys) -> None:
    spec = tmp_path / "s.yaml"
    spec.write_text("name: s\nprofile: custom\naccess: {mode: ssh}\n"
                    "ladder: {context_rungs_tokens: [8192], concurrency_steps: [1, 2, 4]}\n")
    assert main(["plan", str(spec)]) == 0
    assert "no MemAvailable reading" in capsys.readouterr().out
    assert main(["plan", str(spec), "--mem-available", "2.5"]) == 0
    assert "(OVER)" in capsys.readouterr().out
    res = tmp_path / "results.jsonl"
    res.write_text(_row(2.2).model_dump_json() + "\n")
    assert main(["plan", str(spec), "--measured", str(res)]) == 0
    assert "measured baseline" in capsys.readouterr().out
    assert main(["plan", str(spec), "--max-model-len", "262144", "--kv-capacity-tokens", "100000"]) == 0
    assert "Context ceiling per user count" in capsys.readouterr().out


def test_live_preflight_records_mem_but_estimate_never_refuses(mock_server, tmp_path: Path) -> None:
    from xpu_llm_bench.ram import LocalMemInfo

    spec = _spec(mock_server["base"])
    spec.hardware.mem_available_floor_gib = 2.0
    spec.planner.host_bytes_per_token = 10 * 1024 ** 2  # estimate far over budget
    mem = tmp_path / "meminfo"
    mem.write_text("MemAvailable: 3145728 kB\n")  # 3 GiB: 1 GiB above the floor
    r = _runner(spec, tmp_path / "o", mock_server, tmp_path)
    r.ram_source = LocalMemInfo(str(mem))
    man = r.run()
    assert man.status == "completed" and mock_server["state"].chat_requests > 0
    assert man.launch_check["feasibility"]["host"] == "over"
    assert man.launch_check["feasibility"]["mem_available_gib"] == pytest.approx(3.0, abs=0.01)
    assert any(n.startswith("INFO host RAM estimate (never blocks)") for n in man.notes)

    # A live reading already under the floor is a warning too: the run starts,
    # and each user count that hits the floor stops there (CTX_CEILING).
    mem.write_text("MemAvailable: 2050000 kB\n")
    r2 = _runner(spec, tmp_path / "o2", mock_server, tmp_path)
    r2.ram_source = LocalMemInfo(str(mem))
    man2 = r2.run()
    assert man2.status == "completed" and not man2.abort_reason
    assert any("under the floor before the first rung" in n for n in man2.notes)
    assert not man2.ctx_ceilings  # the floor guard is off for this run, every rung measured
