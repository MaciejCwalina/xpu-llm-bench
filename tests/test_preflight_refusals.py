"""F5: placeholder values, a missing container and a refused bench-hold all stop
the run cleanly, with the manifest marked refused (never left in_progress)."""

from pathlib import Path

import pytest

from test_runner_harness import _runner, _spec
from test_variants_ab import _fp
from xpu_llm_bench import cli
from xpu_llm_bench.hold import BenchHold
from xpu_llm_bench.results import RunManifest
from xpu_llm_bench.runner import RunRefused
from xpu_llm_bench.spec import spec_placeholders


def _manifest(path: Path) -> RunManifest:
    return RunManifest.load(path / "manifest.json")


def test_spec_placeholders_lists_field_paths() -> None:
    spec = _spec("http://127.0.0.1:1")
    spec.endpoint.model = "TODO-served-model-name"
    spec.server.image = "<vllm-xpu-image>:<tag>"
    spec.server.env = {"QSA_SHARED": "TODO"}
    spec.notes = ["TODO: free text is fine"]
    found = "\n".join(spec_placeholders(spec))
    assert "endpoint.model" in found and "server.image" in found and "server.env.QSA_SHARED" in found
    assert "notes" not in found


def test_placeholders_stop_live_run(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"])
    spec.server.image_digest = "TODO-image-digest"
    with pytest.raises(RunRefused, match=r"server.image_digest") as err:
        _runner(spec, tmp_path / "o", mock_server, tmp_path).run()
    assert "null" in str(err.value)
    man = _manifest(tmp_path / "o")
    assert man.status == "refused" and "image_digest" in man.abort_reason
    assert mock_server["state"].chat_requests == 0


def test_missing_container_stops_live_run(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"])
    r = _runner(spec, tmp_path / "o", mock_server, tmp_path, require_container=True)
    missing = _fp()
    missing.container_id = None
    missing.problems.append("docker inspect failed: No such object: vllm-bench-mock")
    r._fingerprint_fn = lambda: missing
    with pytest.raises(RunRefused, match="not found.*No such object"):
        r.run()
    assert _manifest(tmp_path / "o").status == "refused"
    assert mock_server["state"].chat_requests == 0


def test_stopped_container_stops_live_run(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"])
    r = _runner(spec, tmp_path / "o", mock_server, tmp_path, require_container=True)
    fp = _fp()
    fp.state_status = "exited"
    r._fingerprint_fn = lambda: fp
    with pytest.raises(RunRefused, match="exited"):
        r.run()


def test_hold_refusal_marks_manifest_refused(mock_server, tmp_path: Path) -> None:
    class P:
        returncode, stdout, stderr = 0, "active by=someone-else reason=matrix until=x acquired=y queued=0\n", ""

    hold = BenchHold("/x/bench-hold", name="xpu-llm-bench", runner=lambda *a, **k: P())
    spec = _spec(mock_server["base"])
    with pytest.raises(RunRefused, match="someone-else"):
        _runner(spec, tmp_path / "o", mock_server, tmp_path, hold=hold).run()
    man = _manifest(tmp_path / "o")
    assert man.status == "refused" and "someone-else" in man.abort_reason and man.finished_utc
    assert mock_server["state"].chat_requests == 0


def test_cli_reports_refusal_without_traceback(monkeypatch, tmp_path: Path, capsys) -> None:
    spec = tmp_path / "s.yaml"
    spec.write_text("name: s\nprofile: custom\nladder: {context_rungs_tokens: [64], concurrency_steps: [1]}\n")

    def boom(*a, **k):
        man = RunManifest.model_validate({
            "run_id": "r", "spec_name": "s", "profile": "custom", "started_utc": "", "started_et": "",
            "hardware": {}, "status": "refused", "output_jsonl": str(tmp_path / "results.jsonl"),
        })
        raise RunRefused("Refusing to start: bench-hold is held by 'someone-else'", man)

    monkeypatch.setattr(cli, "run_spec", boom)
    code = cli.main(["run", str(spec), "--no-site-config"])
    err = capsys.readouterr().err
    assert code == 3
    assert "run refused" in err and "someone-else" in err and "manifest marked refused" in err
    assert "Traceback" not in err
