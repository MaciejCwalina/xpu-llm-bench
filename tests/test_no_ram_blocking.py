"""Rule: no RAM check blocks or aborts a run.

- The host-RAM estimate is a note only (plan exits 0; preflight adds an INFO note).
- Live MemAvailable under the floor before the first rung is a warning only.
- The only pre-run questions are "is the GPU already loaded" and "is the vLLM
  server already busy": warn, ask once (or --yes), run anyway.
"""

from pathlib import Path

import yaml

from xpu_llm_bench import cli
from xpu_llm_bench.results import RunManifest, read_jsonl
from xpu_llm_bench.runner import Runner


def _spec_file(tmp_path: Path, base: str) -> Path:
    spec = {
        "name": "tinyram",
        "profile": "custom",
        "endpoint": {"base_url": f"{base}/v1", "metrics_url": f"{base}/metrics", "health_url": f"{base}/health",
                     "model": "m"},
        "server": {"container_name": "vllm-bench-mock", "speculative": {"method": "mtp", "num_speculative_tokens": 6}},
        # A 4 GiB floor on a host with ~100 MiB available, and a host-RAM
        # estimate that is hopelessly over budget.
        "hardware": {"mem_available_floor_gib": 4.0, "gpu_count": 1, "gpu_mem_gib": 1},
        "planner": {"host_bytes_per_token": 1024 ** 3},
        "prompts": {"types": ["multi_turn_chat"], "seed": 3},
        "quality": {"code_edit": False},
        "ladder": {"context_rungs_tokens": [256, 512], "concurrency_steps": [1, 2, 4], "output_tokens_per_rung": 32,
                   "min_window_steps": 4, "metrics_poll_s": 0.05},
    }
    p = tmp_path / "tinyram.yaml"
    p.write_text(yaml.safe_dump(spec))
    return p


def test_simulated_bench_run_with_tiny_memavailable_proceeds(mock_server, tmp_path: Path, monkeypatch, capsys) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 1048576 kB\nMemAvailable: 102400 kB\n")  # 100 MiB, floor is 4 GiB
    state = mock_server["state"]
    state.running = 2.0  # the server is already busy with someone else's requests
    spec_path = _spec_file(tmp_path, mock_server["base"])

    # `bench plan`: estimates are notes, exit 0.
    assert cli.main(["plan", str(spec_path), "--mem-available", "0.1"]) == 0
    plan_out = capsys.readouterr().out
    assert "Host RAM (info only, never blocks)" in plan_out and "(OVER)" in plan_out and "FAIL" not in plan_out

    # `bench run` through the real CLI, with the fake server and a fake /proc/meminfo.
    from xpu_llm_bench import ram, runner

    real_init = Runner.__init__

    def init(self, *a, **kw):
        kw.setdefault("ram_source", ram.LocalMemInfo(str(meminfo)))
        kw.setdefault("require_container", False)
        kw.setdefault("fingerprint_fn", lambda: None)
        kw.setdefault("fetch_logs", lambda: "")
        kw.setdefault("host_fn", lambda: {"gpus": []})
        kw.setdefault("ps_fn", lambda: "")
        real_init(self, *a, **kw)

    monkeypatch.setattr(runner.Runner, "__init__", init)
    out = tmp_path / "out"
    rc = cli.main(["run", str(spec_path), "--output-dir", str(out), "--no-site-config", "--yes"])
    err = capsys.readouterr().err
    assert rc == 0, err
    man = RunManifest.load(out / "tinyram" / "manifest.json")
    assert man.status == "completed" and not man.abort_reason
    rows = read_jsonl(out / "tinyram" / "results.jsonl")
    # every rung ran and measured: nothing was skipped or stopped for RAM
    assert len(rows) == 6 and all(r.decode_tok_s_per_user for r in rows)
    assert not man.ctx_ceilings
    assert all(r.exit_reason is None for r in rows)
    # RAM: an estimate note and a live-floor warning, nothing else
    assert man.launch_check["feasibility"]["host"] == "over"
    assert any(n.startswith("INFO host RAM estimate (never blocks)") for n in man.notes)
    assert any("under the floor before the first rung" in n for n in man.notes)
    assert "under the floor before the first rung" in err
    # the only pre-run question: server busy / GPU loaded -> warned, answered by --yes, flagged
    assert man.pre_run_load["server_busy"] and man.pre_run_load["confirmed"]
    assert all("PRE_EXISTING_LOAD" in r.flags for r in rows)
    assert not any("RAM_FLOOR" in r.flags for r in rows)
