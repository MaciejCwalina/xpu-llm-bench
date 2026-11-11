"""Nothing in the defaults is specific to one bench host: timestamps, hardware,
bench-hold, KV/graph/spec-decode labels, hold names, URLs, A/B refusal codes,
legacy imports and dry-run evidence."""

import json
import time
from pathlib import Path

import pytest
import yaml

from test_runner_harness import _runner, _spec
from xpu_llm_bench import site
from xpu_llm_bench.cli import main
from xpu_llm_bench.hold import BenchHold
from xpu_llm_bench.import_legacy import import_row
from xpu_llm_bench.results import RungResult, RunManifest, local_tz_name, read_jsonl, stamp_pair
from xpu_llm_bench.runner import Runner, RunRefused
from xpu_llm_bench.schedule import build_schedule
from xpu_llm_bench.spec import RunSpec, load_spec

SPECS = Path(__file__).resolve().parents[1] / "specs"
SRC = Path(__file__).resolve().parents[1] / "src"


# -- timestamps ---------------------------------------------------------------
def test_no_hardcoded_timezone_in_source() -> None:
    for f in SRC.rglob("*.py"):
        assert "America/New_York" not in f.read_text(), f


def test_timezone_env_then_site_then_system(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XPU_BENCH_TZ", raising=False)
    assert local_tz_name() == "system-local"
    p = tmp_path / "site.yaml"
    p.write_text("host: lab\nprotected_containers: [prod-a]\nforbidden_ports: [9100]\ntimezone: Asia/Tokyo\n")
    monkeypatch.setenv(site.SITE_ENV, str(p))
    site.clear_cache()
    assert local_tz_name() == "Asia/Tokyo"
    assert stamp_pair()[1].endswith("+09:00")
    monkeypatch.setenv("XPU_BENCH_TZ", "UTC")
    assert local_tz_name() == "UTC" and stamp_pair()[1].endswith("+00:00")


def test_old_started_et_files_still_load() -> None:
    row = RungResult.model_validate({"run_id": "r", "rung_id": "x", "users": 1, "ctx_tokens": [1], "ctx_per_user": 1,
                                     "prompt_type": "code_edits", "load_mode": "closed_all_decoding",
                                     "started_utc": "u", "started_et": "2026-01-01T00:00:00-05:00"})
    assert row.started_local == "2026-01-01T00:00:00-05:00"


# -- hardware / labels -------------------------------------------------------
def test_defaults_are_neutral() -> None:
    s = RunSpec.model_validate({"name": "n"})
    hw = s.hardware
    assert (hw.gpu_model, hw.gpu_count, hw.gpu_mem_gib, hw.tensor_parallel, hw.host_ram_gib) == (
        "unspecified", None, None, None, None)
    assert s.server.kv_format is None and s.server.graph.mode is None and s.server.speculative.method is None
    assert s.safety.hold_command is None and s.safety.hold_name is None and s.safety.hold_reason is None
    assert s.endpoint.health_url == "http://127.0.0.1:8000/health"


def test_site_hardware_merges_but_spec_wins(monkeypatch, tmp_path: Path) -> None:
    p = tmp_path / "site.yaml"
    p.write_text("host: lab\nprotected_containers: [prod-a]\nforbidden_ports: [9100]\n"
                 "hardware: {gpu_model: Some GPU, gpu_count: 4, gpu_mem_gib: 48}\n")
    monkeypatch.setenv(site.SITE_ENV, str(p))
    site.clear_cache()
    f = tmp_path / "s.yaml"
    f.write_text("name: s\nhardware: {gpu_count: 2}\n")
    hw = load_spec(f).hardware
    assert (hw.gpu_model, hw.gpu_count, hw.gpu_mem_gib, hw.host) == ("Some GPU", 2, 48, "lab")


def test_undeclared_kv_and_graphs_are_not_flagged(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"], server={"container_name": "vllm-bench-mock", "speculative": {},
                                              "graph": {}, "attention_requested": "auto"})
    assert spec.server.kv_format is None and spec.server.graph.mode is None
    _runner(spec, tmp_path / "o", mock_server, tmp_path).run()
    rows = read_jsonl(tmp_path / "o" / "results.jsonl")
    assert rows and all("NOT_AS_LABELED" not in r.flags for r in rows)
    man = RunManifest.load(tmp_path / "o" / "manifest.json")
    assert man.graph.graphs_declared is None and man.display_timezone


# -- bench-hold is optional ------------------------------------------------------
def test_live_run_without_hold_command_runs_and_warns(mock_server, tmp_path: Path, capsys) -> None:
    spec = _spec(mock_server["base"])
    man = _runner(spec, tmp_path / "o", mock_server, tmp_path, hold=None).run()
    assert man.status == "completed" and "skipped" in man.hold
    assert "no safety.hold_command configured" in capsys.readouterr().err


def test_quiet_hold_still_refuses_without_hold_command(mock_server, tmp_path: Path) -> None:
    quiet = tmp_path / "quiet"
    quiet.write_text(f"until={time.time() + 3600:.0f} by=operator\n")
    spec = _spec(mock_server["base"], safety={"quiet_hold_file": str(quiet)})
    with pytest.raises(RunRefused, match="quiet hold"):
        _runner(spec, tmp_path / "o", mock_server, tmp_path, hold=None).run()
    assert mock_server["state"].chat_requests == 0


# -- hold names are per run --------------------------------------------------------
def _hold_runner(spec, out, mock, tmp_path):
    return _runner(spec, out, mock, tmp_path, hold=None, dry_run=True)


def test_default_hold_name_and_reason_are_unique_per_run(mock_server, tmp_path: Path) -> None:
    a = _spec(mock_server["base"], name="copy-a", safety={"hold_command": "/opt/x/bench-hold"})
    b = _spec(mock_server["base"], name="copy-b", safety={"hold_command": "/opt/x/bench-hold"})
    ra, rb = _hold_runner(a, tmp_path / "a", mock_server, tmp_path), _hold_runner(b, tmp_path / "b", mock_server, tmp_path)
    ma, mb = ra.run(), rb.run()
    assert ra.hold.name == ma.run_id and rb.hold.name == mb.run_id and ra.hold.name != rb.hold.name
    assert "copy-a" in ra.hold.name and "copy-b" in rb.hold.name
    assert ma.run_id in ra.hold_reason and "copy-a" in ra.hold_reason
    # a second run of the same spec gets its own name too
    ra2 = _hold_runner(a, tmp_path / "a2", mock_server, tmp_path)
    ra2.run()
    assert ra2.hold.name != ra.hold.name


def test_explicit_hold_name_is_kept(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"], safety={"hold_command": "/opt/x/bench-hold", "hold_name": "team-lock",
                                              "hold_reason": "nightly"})
    r = _hold_runner(spec, tmp_path / "o", mock_server, tmp_path)
    r.run()
    assert r.hold.name == "team-lock" and r.hold_reason == "nightly"


def test_injected_hold_keeps_its_name(mock_server, tmp_path: Path) -> None:
    r = _runner(_spec(mock_server["base"]), tmp_path / "o", mock_server, tmp_path,
                hold=BenchHold("/bin/true", name="mine", dry_run=True))
    r.run()
    assert r.hold.name == "mine"


# -- URLs ---------------------------------------------------------------------------
def test_health_and_metrics_follow_base_url() -> None:
    ep = RunSpec.model_validate({"name": "u", "endpoint": {"base_url": "http://bench-host:9123/v1"}}).endpoint
    assert ep.health_url == "http://bench-host:9123/health" and ep.metrics_url == "http://bench-host:9123/metrics"
    ep = RunSpec.model_validate({"name": "u", "endpoint": {"base_url": "http://h:1/v1",
                                                           "health_url": "http://h:1/ready"}}).endpoint
    assert ep.health_url == "http://h:1/ready"


# -- A/B refusal exit code -------------------------------------------------------------
def test_shared_endpoint_ab_refusal_exits_3(tmp_path: Path, capsys) -> None:
    f = tmp_path / "ab.yaml"
    f.write_text(yaml.safe_dump({"name": "ab", "profile": "custom", "server": {"container_name": "vllm-bench-ab"},
                                 "variants": [{"name": "a", "env": {"X": "1"}}, {"name": "b", "env": {"X": "0"}}]}))
    assert main(["run", str(f), "--output-dir", str(tmp_path / "o"), "--no-site-config"]) == 3
    assert "--variant" in capsys.readouterr().err
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert "exit 3" in readme


# -- legacy import ----------------------------------------------------------------------
def test_attention_path_is_declared_not_proven() -> None:
    raw = {"users": 1, "ctx": 8192, "ok": True}
    row = import_row(dict(raw), attention_path="qsa")
    assert row.attention_path == "qsa" and "ATTENTION_DECLARED" in row.flags
    assert "declared, not proven" in (row.reason or "")
    plain = import_row(dict(raw))
    assert plain.attention_path == "unknown" and "ATTENTION_DECLARED" not in plain.flags


def test_import_legacy_cli_attention_path(tmp_path: Path) -> None:
    src = tmp_path / "old.jsonl"
    src.write_text(json.dumps({"users": 2, "ctx": 8192, "ok": True}) + "\n")
    out = tmp_path / "new.jsonl"
    assert main(["import-legacy", str(src), str(out), "--attention-path", "dense"]) == 0
    row = read_jsonl(out)[0]
    assert row.attention_path == "dense" and "ATTENTION_DECLARED" in row.flags
    with pytest.raises(SystemExit):
        main(["import-legacy", str(src), str(out), "--attention-path", "maybe"])


def test_both_converted_pair_is_caveated() -> None:
    from xpu_llm_bench.report import compare_runs

    a = import_row({"users": 1, "ctx": 8192, "ok": True})
    b = import_row({"users": 1, "ctx": 8192, "ok": True})
    (pair,) = compare_runs([a], [b])
    assert any("both rows imported" in r for r in pair.reasons)


# -- dry-run evidence --------------------------------------------------------------------
def test_dry_run_never_claims_graphs_on(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"], server={"container_name": "vllm-bench-mock",
                                              "graph": {"mode": "FULL_DECODE_ONLY"}})
    man = _runner(spec, tmp_path / "o", mock_server, tmp_path, dry_run=True).run()
    assert man.graph.graphs_on is None and man.graph.graphs_basis == "unknown"
    assert man.graph.graphs_declared is True
    rows = read_jsonl(tmp_path / "o" / "dry-run" / "results.jsonl") if (tmp_path / "o" / "dry-run" / "results.jsonl").exists() else []
    assert all(r.graphs_on is not True for r in rows)


def test_live_graphs_on_comes_from_boot_log(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"])
    man = _runner(spec, tmp_path / "o", mock_server, tmp_path).run()
    assert man.graph.graphs_basis in {"boot_log", "unknown"}
    if man.graph.graphs_basis == "unknown":
        assert man.graph.graphs_on is None


# -- generic spec --------------------------------------------------------------------------
def test_generic_spec_is_fork_free_and_progressive() -> None:
    spec = load_spec(SPECS / "generic-vllm.yaml")
    assert spec.ladder.shape == "session1_then_join"
    assert spec.server.env == {} and spec.server.launch_args == []
    assert spec.server.kv_format is None and spec.server.graph.mode is None and spec.server.speculative.method is None
    assert spec.hardware.tensor_parallel is None
    assert build_schedule(spec)


def test_shipped_specs_carry_no_host_facts() -> None:
    for f in SPECS.glob("*.yaml"):
        data = yaml.safe_load(f.read_text())
        hw = data.get("hardware") or {}
        assert not {"host", "gpu_model", "gpu_count", "gpu_mem_gib", "host_ram_gib", "mem_available_floor_gib"} & set(hw), f
        sf = data.get("safety") or {}
        assert not {"hold_name", "hold_reason", "hold_command"} & set(sf), f


def test_runner_class_exported() -> None:
    assert Runner
