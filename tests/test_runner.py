from pathlib import Path

from xpu_llm_bench.client import OpenAICompatClient
from xpu_llm_bench.hold import BenchHold
from xpu_llm_bench.ram import LocalMemInfo
from xpu_llm_bench.results import RunManifest, read_jsonl
from xpu_llm_bench.runner import Runner
from xpu_llm_bench.spec import RunSpec


def _spec(base_url: str, **extra) -> RunSpec:
    data = {
        "name": "mock-run",
        "profile": "custom",
        "endpoint": {
            "base_url": f"{base_url}/v1",
            "metrics_url": f"{base_url}/metrics",
            "health_url": f"{base_url}/health",
            "model": "mock",
        },
        "server": {
            "container_name": "vllm-bench-mock",
            "manage_container": False,
            "attention_requested": "qsa",
            "image": "mock:latest",
        },
        "hardware": {"mem_available_floor_gib": 0.001, "host_ram_gib": 25},
        "safety": {"allowed_containers": ["vllm-bench-mock"], "hold_name": "mock"},
        "prompts": {"types": ["code_edits"], "seed": 1, "chars_per_token_guess": 4.0},
        "ladder": {
            "context_rungs_tokens": [64, 128],
            "concurrency_steps": [1, 2],
            "load_modes": ["closed_all_decoding"],
            "output_tokens_per_rung": 4,
            "warmup": {"tokens": 1, "requests": 0},
        },
        "access": {"mode": "local"},
    }
    data.update(extra)
    return RunSpec.model_validate(data)


def test_runner_against_mock_server(mock_server, tmp_path: Path) -> None:
    mem = tmp_path / "meminfo"
    # ~8 GiB available
    mem.write_text("MemAvailable: 8388608 kB\n")
    spec = _spec(mock_server["base"])
    client = OpenAICompatClient(spec.endpoint.base_url, spec.endpoint.model, timeout_s=10)
    hold = BenchHold(command="/bin/true", dry_run=True)
    runner = Runner(
        spec,
        tmp_path / "out",
        dry_run=False,
        client=client,
        hold=hold,
        ram_source=LocalMemInfo(str(mem)),
        fetch_logs=lambda: "\n".join(mock_server["logs"]),
        collect_mem=lambda **kwargs: "kv_cache 12.5%",
        no_site_config=True,
        require_container=False,
    )
    manifest = runner.run()
    assert manifest.status == "completed"
    rows = read_jsonl(tmp_path / "out" / "results.jsonl")
    assert len(rows) == 3  # progressive: 1u@64, 1u@128 (session 1 grows), then 2u join at 64
    ids = [r.rung_id for r in rows]
    assert ids == sorted(ids, key=lambda x: ids.index(x))  # stable order
    assert {r.users for r in rows} == {1, 2}
    assert {r.ctx_per_user for r in rows} == {64, 128}
    assert all(r.prompt_type == "code_edits" for r in rows)
    # MTP accounting present
    measured = [r for r in rows if r.hit_rate is not None]
    assert measured
    for row in measured:
        assert 0 <= row.hit_rate <= 1
        assert row.accepted_drafts_per_step is not None
        assert row.decode_tok_s_total is not None
        assert row.ttft_ms is not None
        assert row.mem_available_min_gib is not None
        assert row.attention_path in {"qsa", "mixed", "unknown"}  # mock logs width=
    assert (tmp_path / "out" / "manifest.json").exists()
    loaded = RunManifest.load(tmp_path / "out" / "manifest.json")
    assert loaded.completed_rung_ids == [r.rung_id for r in rows]


def test_resume_skips_completed(mock_server, tmp_path: Path) -> None:
    mem = tmp_path / "meminfo"
    mem.write_text("MemAvailable: 8388608 kB\n")
    spec = _spec(mock_server["base"])
    out = tmp_path / "out"
    client = OpenAICompatClient(spec.endpoint.base_url, spec.endpoint.model, timeout_s=10)
    kwargs = dict(
        spec=spec,
        output_dir=out,
        client=client,
        hold=BenchHold(command="/bin/true", dry_run=True),
        ram_source=LocalMemInfo(str(mem)),
        fetch_logs=lambda: "QSA_GRAPH width=65536",
        collect_mem=lambda **kwargs: None,
        no_site_config=True,
        require_container=False,
    )
    first = Runner(**kwargs).run()
    rows1 = read_jsonl(out / "results.jsonl")
    assert first.status == "completed"
    # Resume should not duplicate rows
    second = Runner(**kwargs, resume=True).run()
    rows2 = read_jsonl(out / "results.jsonl")
    assert len(rows2) == len(rows1)
    assert second.completed_rung_ids == first.completed_rung_ids


def test_dry_run_writes_rows_without_http(tmp_path: Path) -> None:
    mem = tmp_path / "meminfo"
    mem.write_text("MemAvailable: 8388608 kB\n")
    spec = _spec("http://127.0.0.1:1")  # nothing listening
    runner = Runner(
        spec,
        tmp_path / "dry",
        dry_run=True,
        hold=BenchHold(command="/bin/true", dry_run=True),
        ram_source=LocalMemInfo(str(mem)),
    )
    manifest = runner.run()
    assert manifest.status == "dry_run"
    assert manifest.completed_rung_ids == [] and manifest.planned_rung_ids
    rows = read_jsonl(tmp_path / "dry" / "dry-run" / "results.jsonl")
    assert rows
    assert all(r.reason == "dry-run" for r in rows)


def test_ram_floor_aborts_and_is_resumable(tmp_path: Path) -> None:
    mem = tmp_path / "meminfo"
    mem.write_text("MemAvailable: 1024 kB\n")  # well under 2 GiB, and under 0.001? 1024 KiB is 0.001 GiB exactly... 1024 KiB = 1/1024 GiB ≈ 0.00098
    spec = _spec("http://127.0.0.1:1")
    runner = Runner(
        spec,
        tmp_path / "abort",
        dry_run=True,
        hold=BenchHold(command="/bin/true", dry_run=True),
        ram_source=LocalMemInfo(str(mem)),
    )
    import pytest

    from xpu_llm_bench.ram import RamFloorError

    with pytest.raises(RamFloorError):
        runner.run()
    manifest = RunManifest.load(tmp_path / "abort" / "dry-run" / "manifest.json")
    assert manifest.status == "aborted"
    assert "floor" in (manifest.abort_reason or "").lower()
