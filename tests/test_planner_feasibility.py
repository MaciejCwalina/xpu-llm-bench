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
from xpu_llm_bench.results import RungResult, RunManifest
from xpu_llm_bench.runner import RunRefused
from xpu_llm_bench.schedule import build_schedule
from xpu_llm_bench.spec import RunSpec

SPECS = Path(__file__).resolve().parents[1] / "specs"


def _s(**over) -> RunSpec:
    data = {"name": "f", "profile": "custom",
            "hardware": {"gpu_count": 2, "gpu_mem_gib": 24},
            "ladder": {"context_rungs_tokens": [8192, 131072], "concurrency_steps": [1, 2, 4, 8]}}
    for k, v in over.items():
        data[k] = {**data.get(k, {}), **v} if isinstance(v, dict) else v
    return RunSpec.model_validate(data)


def _f(spec, mem=None):
    return feasibility(spec, build_schedule(spec), mem)


def test_without_a_reading_host_is_unverified_never_ok() -> None:
    f = _f(_s())
    assert f.host_status == "unverified" and f.ok  # a warning, not a pass and not a failure
    assert any(n.startswith("WARN host RAM: no MemAvailable reading") for n in f.notes)
    assert "UNVERIFIED" in __import__("xpu_llm_bench.planner", fromlist=["format_plan"]).format_plan(make_plan(_s()))


def test_live_mem_available_decides() -> None:
    assert _f(_s(), mem=64.0).host_status == "ok"
    f = _f(_s(), mem=6.0)
    assert f.host_status == "fail" and not f.ok
    assert any(n.startswith("FAIL host RAM") for n in f.notes)


def test_qsa_4_users_at_8k_is_not_ok() -> None:
    """Reality: QSA at 4 users / 8K joiners hit the 2 GiB MemAvailable floor,
    starting from ~4.8 GiB available with the server booted."""
    spec = _s(ladder={"context_rungs_tokens": [8192], "concurrency_steps": [1, 2, 4]})
    assert _f(spec, mem=4.8).host_status in {"tight", "fail"}
    quick = RunSpec.model_validate(
        {**__import__("yaml").safe_load((SPECS / "qsa-vs-dense.yaml").read_text()), "profile": "quick"}
    )
    from xpu_llm_bench.spec import apply_profile, load_spec  # noqa: F401

    q = load_spec(SPECS / "qsa-vs-dense.yaml", profile="quick")
    assert feasibility(q, build_schedule(q), 4.8).host_status == "fail"
    assert quick.name == "qsa-vs-dense"


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
    assert _f(spec, mem=20.0).host_status == "fail"


def test_gpu_can_fail_warn_or_be_unknown() -> None:
    assert _f(_s(planner={"assumed_kv_bytes_per_token": 200_000})).gpu_status == "fail"
    assert _f(_s(planner={"assumed_weight_gib": 36.0})).gpu_status == "tight"
    unknown = RunSpec.model_validate({"name": "u", "profile": "custom"})
    f = _f(unknown)
    assert f.gpu_status == "unknown" and any("gpu_count / gpu_mem_gib not set" in n for n in f.notes)


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


def test_plan_cli_exit_codes(tmp_path: Path, capsys) -> None:
    spec = tmp_path / "s.yaml"
    spec.write_text("name: s\nprofile: custom\naccess: {mode: ssh}\n"
                    "ladder: {context_rungs_tokens: [8192], concurrency_steps: [1, 2, 4]}\n")
    assert main(["plan", str(spec)]) == 0
    assert "UNVERIFIED" in capsys.readouterr().out
    assert main(["plan", str(spec), "--mem-available", "2.5"]) == 1
    assert "FAIL host RAM" in capsys.readouterr().out
    res = tmp_path / "results.jsonl"
    res.write_text(_row(2.2).model_dump_json() + "\n")
    assert main(["plan", str(spec), "--measured", str(res)]) == 1
    assert "measured baseline" in capsys.readouterr().out


def test_live_preflight_uses_live_mem_available(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"])
    spec.hardware.mem_available_floor_gib = 2.0
    mem = tmp_path / "meminfo"
    mem.write_text("MemAvailable: 2050000 kB\n")  # ~1.96 GiB: already under the 2 GiB floor
    r = _runner(spec, tmp_path / "o", mock_server, tmp_path)
    from xpu_llm_bench.ram import LocalMemInfo

    r.ram_source = LocalMemInfo(str(mem))
    with pytest.raises(RunRefused, match="FAIL host RAM"):
        r.run()
    assert mock_server["state"].chat_requests == 0
    man = RunManifest.load(tmp_path / "o" / "manifest.json")
    assert man.status == "refused" and man.launch_check["feasibility"]["mem_available_gib"] == pytest.approx(1.96, abs=0.01)
