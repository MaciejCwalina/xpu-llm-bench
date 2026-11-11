"""F6: dry runs never look like progress, never clobber a real run, store no
prompt text, and print the safety commands; sessions.json stays small."""

import json
from pathlib import Path

import pytest

from test_runner_harness import _runner, _spec
from xpu_llm_bench.cli import main
from xpu_llm_bench.results import RunManifest, read_jsonl
from xpu_llm_bench.runner import Session


def test_dry_run_manifest_is_not_progress(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"])
    man = _runner(spec, tmp_path / "o", mock_server, tmp_path, dry_run=True).run()
    dry = tmp_path / "o" / "dry-run"
    assert man.status == "dry_run" and man.completed_rung_ids == [] and man.planned_rung_ids
    assert RunManifest.load(dry / "manifest.json").status == "dry_run"
    assert not (dry / "sessions.json").exists()
    assert not (tmp_path / "o" / "manifest.json").exists()
    # A real --resume afterwards runs every rung.
    real = _runner(spec, tmp_path / "o", mock_server, tmp_path, resume=True).run()
    assert real.status == "completed" and real.completed_rung_ids == man.planned_rung_ids


def test_resume_ignores_a_dry_run_manifest(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"])
    out = tmp_path / "o"
    out.mkdir()
    fake = _runner(spec, out, mock_server, tmp_path, dry_run=True).run()
    fake.completed_rung_ids = list(fake.planned_rung_ids)  # worst case: an old-style dry manifest
    fake.save(out / "manifest.json")
    real = _runner(spec, out, mock_server, tmp_path, resume=True).run()
    assert len(read_jsonl(out / "results.jsonl")) == len(fake.planned_rung_ids)
    assert real.run_id != fake.run_id


def test_dry_run_does_not_touch_a_real_run(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"])
    _runner(spec, tmp_path / "o", mock_server, tmp_path).run()
    before = (tmp_path / "o" / "results.jsonl").read_text()
    _runner(spec, tmp_path / "o", mock_server, tmp_path, dry_run=True).run()
    assert (tmp_path / "o" / "results.jsonl").read_text() == before
    assert RunManifest.load(tmp_path / "o" / "manifest.json").status == "completed"


def _grown(target: int = 131072) -> Session:
    s = Session(1, "code_edits", seed=7, chars_per_token=4.0)
    s.grow_to(8192)
    s.append_reply("first reply")
    s.grow_to(target)
    s.append_reply("second reply")
    return s


def test_sessions_state_is_compact_and_regenerates_exactly() -> None:
    s = _grown()
    blob = json.dumps(s.to_dict())
    text = sum(len(m["content"]) for m in s.messages)
    assert text > 400_000 and len(blob) < 4_000
    t = Session(1, "code_edits", seed=7, chars_per_token=4.0)
    t.load_state(json.loads(blob))
    assert t.messages == s.messages and t.segments == s.segments


def test_tampered_or_changed_corpus_refuses_resume() -> None:
    d = _grown(16384).to_dict()
    d["turns"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="resume refused"):
        Session(1, "code_edits", seed=7, chars_per_token=4.0).load_state(d)


def test_format1_sessions_still_load() -> None:
    s = Session(1, "code_edits", seed=7, chars_per_token=4.0)
    s.load_state({"messages": [{"role": "user", "content": "hi"}], "segments": 1, "cpt": 4.0})
    assert s.messages == [{"role": "user", "content": "hi"}]


def test_cli_dry_run_prints_safety_commands(tmp_path: Path, capsys) -> None:
    spec = tmp_path / "s.yaml"
    spec.write_text(
        "name: s\nprofile: custom\n"
        "endpoint: {base_url: 'http://127.0.0.1:1/v1', metrics_url: 'http://127.0.0.1:1/metrics', "
        "health_url: 'http://127.0.0.1:1/health', model: TODO-model}\n"
        "server: {container_name: vllm-bench-x}\n"
        "safety: {hold_command: /opt/x/bench-hold}\n"
        "hardware: {mem_available_floor_gib: 0.001}\n"
        "ladder: {context_rungs_tokens: [64, 128], concurrency_steps: [1, 2]}\n"
    )
    assert main(["run", str(spec), "--dry-run", "--output-dir", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "ignored by --resume" in out
    assert "/opt/x/bench-hold acquire" in out and "/opt/x/bench-hold release" in out
    assert "docker inspect vllm-bench-x" in out
    assert "endpoint.model" in out  # placeholder warning surfaces in a dry run
    assert (tmp_path / "out" / "s" / "dry-run" / "manifest.json").exists()
    assert not (tmp_path / "out" / "s" / "dry-run" / "sessions.json").exists()
