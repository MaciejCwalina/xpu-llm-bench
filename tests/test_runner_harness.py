import json
from pathlib import Path

import pytest

from xpu_llm_bench.client import OpenAICompatClient
from xpu_llm_bench.hold import BenchHold
from xpu_llm_bench.ram import LocalMemInfo, RamSource
from xpu_llm_bench.report import run_report
from xpu_llm_bench.results import RunManifest, read_jsonl
from xpu_llm_bench.runner import Runner
from xpu_llm_bench.safety import SafetyError
from xpu_llm_bench.spec import RunSpec


def _spec(base: str, **over) -> RunSpec:
    data = {
        "name": "h",
        "profile": "custom",
        "endpoint": {"base_url": f"{base}/v1", "metrics_url": f"{base}/metrics", "health_url": f"{base}/health", "model": "m"},
        "server": {
            "container_name": "vllm-bench-mock",
            "attention_requested": "qsa",
            "speculative": {"method": "mtp", "num_speculative_tokens": 6},
            "graph": {"capture_sizes": [7, 14, 21, 28]},
        },
        "hardware": {"mem_available_floor_gib": 0.001},
        "prompts": {"types": ["multi_turn_chat"], "seed": 11},
        "ladder": {
            "shape": "session1_then_join",
            "context_rungs_tokens": [256, 512],
            "concurrency_steps": [1, 2, 4],
            "output_tokens_per_rung": 64,
            "min_window_steps": 10,
            "metrics_poll_s": 0.05,
        },
        "quality": {"code_edit": False},
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k] = {**data[k], **v}
        else:
            data[k] = v
    return RunSpec.model_validate(data)


def _runner(spec, out, mock, tmp_path, **kw):
    mem = tmp_path / "meminfo"
    if not mem.exists():
        mem.write_text("MemAvailable: 8388608 kB\n")
    kw.setdefault("hold", BenchHold("/bin/true", dry_run=True))
    kw.setdefault("ram_source", LocalMemInfo(str(mem)))
    kw.setdefault("no_site_config", True)
    kw.setdefault("require_container", False)
    return Runner(
        spec,
        out,
        client=OpenAICompatClient(spec.endpoint.base_url, "m", timeout_s=10),
        fetch_logs=lambda: "\n".join(mock["logs"]),
        fingerprint_fn=lambda: None,
        proc_fn=lambda: __import__("xpu_llm_bench.procmem", fromlist=["ProcSnapshot"]).ProcSnapshot(),
        host_fn=lambda: {"gpus": []},
        **kw,
    )


def test_progressive_ladder_rows_and_metrics(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"])
    man = _runner(spec, tmp_path / "o", mock_server, tmp_path).run()
    assert man.status == "completed" and man.prompt_seed == 11 and man.k == 6
    rows = read_jsonl(tmp_path / "o" / "results.jsonl")
    assert [r.rung_id for r in rows] == [
        "multi_turn_chat:closed:1u:256:S",
        "multi_turn_chat:closed:1u:512:S",
        "multi_turn_chat:closed:2u:256:C",
        "multi_turn_chat:closed:4u:256:C",
    ]
    for r in rows:
        # Mock streams 1 prefill token then 4-token verify steps (1 bonus + 3 accepted).
        assert r.tokens_per_step == pytest.approx(4.0, rel=0.05)
        assert r.hit_rate == pytest.approx(3 / 6, rel=0.05)
        assert r.verify_steps == r.verify_steps_server
        assert r.steps_s_per_user and r.decode_tok_s_per_user
        assert r.prompt_seed == 11 and len(r.session_seeds) == r.users
        assert r.itl_p50_ms is not None and r.itl_p99_ms is not None and r.e2e_ms is not None
        assert r.queue_time_ms is not None and r.kv_cache_usage_pct is not None
        assert r.mem_available_min_gib == pytest.approx(8.0)
        assert "SPEC_COUNTER_BROKEN" not in r.flags and "CONCURRENT_CLIENT" not in r.flags
        assert r.decode_batch_tokens == r.users * 7
    # session 1 never shrinks, joiners at 256
    assert rows[0].ctx_tokens[0] <= rows[1].ctx_tokens[0] <= rows[3].ctx_tokens[0]
    assert len(rows[3].ctx_tokens) == 4
    # report: steps/s sits next to tok/s
    md = run_report(rows, RunManifest.load(tmp_path / "o" / "manifest.json"))
    header = next(ln for ln in md.splitlines() if ln.startswith("| rung_id"))
    cols = [c.strip() for c in header.strip("|").split("|")]
    assert cols.index("decode_tok_s_per_user") == cols.index("steps_s_per_user") + 1


def test_prompts_are_seeded_not_time_based(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"])
    _runner(spec, tmp_path / "a", mock_server, tmp_path).run()
    _runner(spec, tmp_path / "b", mock_server, tmp_path).run()
    sa = json.loads((tmp_path / "a" / "sessions.json").read_text())
    sb = json.loads((tmp_path / "b" / "sessions.json").read_text())
    # Compact format: user turns are (segment, n_chars, opener, sha256), regenerated on load.
    users = lambda d: [[t["sha256"] for t in s["turns"] if t["role"] == "user"] for s in d.values()]  # noqa: E731
    assert users(sa) == users(sb) and users(sa)[0]


def test_k_from_spec_drives_hit_rate_and_flags_mismatch(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"], server={"speculative": {"method": "dflash2", "num_speculative_tokens": 7}})
    _runner(spec, tmp_path / "o", mock_server, tmp_path).run()
    rows = read_jsonl(tmp_path / "o" / "results.jsonl")
    # The mock exports 6 draft tokens per step, so a k=7 label is caught.
    assert all("SPEC_COUNTER_BROKEN" in r.flags for r in rows)
    assert rows[0].hit_rate == pytest.approx(3 / 7, rel=0.05)


def test_k_undeclared_is_flagged(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"], server={"speculative": {"method": "mtp"}})
    _runner(spec, tmp_path / "o", mock_server, tmp_path).run()
    rows = read_jsonl(tmp_path / "o" / "results.jsonl")
    assert all("K_UNDECLARED" in r.flags for r in rows)
    assert rows[0].hit_rate == pytest.approx(0.5, rel=0.05)


def test_greedy_reference_match(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"])
    _runner(spec, tmp_path / "ref", mock_server, tmp_path).run()
    ref = tmp_path / "ref" / "greedy_outputs.jsonl"
    spec2 = _spec(mock_server["base"], quality={"greedy_reference": str(ref), "code_edit": False})
    _runner(spec2, tmp_path / "cmp", mock_server, tmp_path).run()
    rows = read_jsonl(tmp_path / "cmp" / "results.jsonl")
    assert all(r.quality.greedy_ref == "match" for r in rows)
    # Tamper with the reference: divergence index is recorded
    lines = [json.loads(x) for x in ref.read_text().splitlines()]
    lines[0]["text"] = "xy" + lines[0]["text"][2:]
    ref.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    _runner(spec2, tmp_path / "cmp2", mock_server, tmp_path).run()
    rows = read_jsonl(tmp_path / "cmp2" / "results.jsonl")
    assert rows[0].quality.greedy_ref == "diverged" and rows[0].quality.greedy_first_divergence == 1
    assert "QUALITY_FAIL" in rows[0].flags
    assert rows[1].quality.greedy_ref == "context_diverged"


def test_refuses_foreign_hold_before_any_request(mock_server, tmp_path: Path) -> None:
    class P:
        returncode, stdout, stderr = 0, "active by=speedfix-s5 reason=matrix until=x acquired=y queued=0\n", ""

    hold = BenchHold("/x/bench-hold", name="xpu-llm-bench", runner=lambda *a, **k: P())
    spec = _spec(mock_server["base"])
    with pytest.raises(SafetyError, match="speedfix-s5"):
        _runner(spec, tmp_path / "o", mock_server, tmp_path, hold=hold).run()
    assert mock_server["state"].chat_requests == 0


def test_engine_error_with_spec_assert_aborts_and_resume_continues(mock_server, tmp_path: Path) -> None:
    state = mock_server["state"]
    spec = _spec(mock_server["base"])
    state.fail_from = 3  # grow(1), measure(1), needle... -> fails inside rung 2
    man = _runner(spec, tmp_path / "o", mock_server, tmp_path).run()
    assert man.status == "aborted" and "SPEC_ASSERT" in (man.abort_reason or "")
    rows = read_jsonl(tmp_path / "o" / "results.jsonl")
    assert rows[-1].exit_reason == "SPEC_ASSERT" and rows[-1].valid == "invalid"
    done = list(man.completed_rung_ids)
    # Operator restarts the test container; resume from the last completed rung.
    state.fail_from = None
    mock_server["logs"].clear()
    man2 = _runner(spec, tmp_path / "o", mock_server, tmp_path, resume=True).run()
    assert man2.status == "completed"
    assert man2.completed_rung_ids[: len(done)] == done
    rows = read_jsonl(tmp_path / "o" / "results.jsonl")
    ok = [r for r in rows if r.valid != "invalid"]
    assert [r.rung_id for r in ok] == man2.completed_rung_ids
    # Session 1 resumed at its high-water context (never rebuilt smaller).
    first_after = next(r for r in rows if r.rung_id == man2.completed_rung_ids[len(done)])
    assert first_after.ctx_tokens[0] >= rows[0].ctx_tokens[0]


class Dropping(RamSource):
    def __init__(self, seq):
        self.seq = list(seq)

    def read_kib(self) -> int:
        return self.seq.pop(0) if len(self.seq) > 1 else self.seq[0]


def test_ram_floor_mid_run_is_a_row_flag_and_resumable(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"], hardware={"mem_available_floor_gib": 2.0})
    # plenty at start, then 1.5 GiB
    src = Dropping([8 * 1024 * 1024, 1536 * 1024])
    man = _runner(spec, tmp_path / "o", mock_server, tmp_path, ram_source=src).run()
    assert man.status == "aborted" and man.abort_reason.startswith("RAM_FLOOR")
    rows = read_jsonl(tmp_path / "o" / "results.jsonl")
    assert rows[-1].exit_reason == "RAM_FLOOR" and "RAM_FLOOR" in rows[-1].flags
    assert rows[-1].mem_available_min_gib is not None and rows[-1].mem_available_min_gib < 2.0


def test_dry_run_makes_no_requests_and_flags_rows(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"])
    r = _runner(spec, tmp_path / "o", mock_server, tmp_path, dry_run=True)
    man = r.run()
    assert man.status == "dry_run"
    assert mock_server["state"].chat_requests == 0
    rows = read_jsonl(tmp_path / "o" / "dry-run" / "results.jsonl")
    assert rows and all(row.flags == ["DRY_RUN"] for row in rows)
    assert "would_run" in man.hold


def test_dry_run_still_refuses_production_port(tmp_path: Path) -> None:
    spec = _spec("http://127.0.0.1:8001")
    with pytest.raises(SafetyError, match="8001"):
        Runner(spec, tmp_path / "o", dry_run=True, ram_source=LocalMemInfo("/proc/meminfo")).run()
