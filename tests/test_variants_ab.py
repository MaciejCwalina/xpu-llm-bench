"""F3: each A/B variant runs against its own booted container, and the
container is verified (docker inspect) against the variant's expectations."""

from pathlib import Path

import pytest

from test_runner_harness import _runner, _spec
from xpu_llm_bench import runner as runner_mod
from xpu_llm_bench.fingerprint import Fingerprint, verify_expected
from xpu_llm_bench.results import read_jsonl
from xpu_llm_bench.spec import RunSpec, apply_variant, is_placeholder


def _fp(env=(), cmd=(), image="img:1", image_id="sha256:abc", digests=()) -> Fingerprint:
    return Fingerprint(
        container="vllm-bench-mock",
        container_id="c0ffee",
        image_tag=image,
        image_id=image_id,
        repo_digests=list(digests),
        started_at="2026-01-01T00:00:00Z",
        cmd=list(cmd),
        env=list(env),
        state_status="running",
    )


def test_verify_expected_env_args_image() -> None:
    fp = _fp(
        env=["K8V4_QSA=1", "HF_TOKEN=<redacted>"],
        cmd=["vllm", "serve", "/m", "--tensor-parallel-size", "2", "--max-num-seqs=8"],
    )
    ok = verify_expected(fp, {"K8V4_QSA": "1", "HF_TOKEN": "x"}, ["--tensor-parallel-size", "2", "--max-num-seqs", "8"], "img:1")
    assert ok == []
    bad = verify_expected(fp, {"K8V4_QSA": "0", "MISSING": "1"}, ["--enforce-eager"], "img:2")
    text = "\n".join(bad)
    assert "K8V4_QSA: expected '0'" in text and "MISSING" in text
    assert "--enforce-eager" in text and "img:2" in text


def test_verify_handles_shell_string_cmd_and_skips_placeholders() -> None:
    fp = _fp(cmd=["bash", "-c", "vllm serve /m --speculative-config '{\"k\": 6}' --tp 2"])
    assert verify_expected(fp, {}, ["--speculative-config", '{"k": 6}', "--tp", "2"]) == []
    assert verify_expected(fp, {}, [], "<vllm-xpu-image>:<tag>", "TODO-image-digest", is_placeholder) == []


def test_container_not_as_labeled_flags_every_row(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"])
    spec.server.env = {"K8V4_QSA": "1"}
    r = _runner(spec, tmp_path / "o", mock_server, tmp_path, require_container=True)
    r._fingerprint_fn = lambda: _fp(env=["K8V4_QSA=0"])
    man = r.run()
    assert man.launch_check["ok"] is False
    assert any("K8V4_QSA" in m for m in man.launch_check["mismatches"])
    rows = read_jsonl(tmp_path / "o" / "results.jsonl")
    assert rows and all("NOT_AS_LABELED" in row.flags for row in rows)


def test_matching_container_is_not_flagged(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"])
    r = _runner(spec, tmp_path / "o", mock_server, tmp_path, require_container=True)
    r._fingerprint_fn = lambda: _fp()
    man = r.run()
    assert man.launch_check["ok"] is True
    assert man.status == "completed"


def _ab(**variant_extra) -> RunSpec:
    return RunSpec.model_validate(
        {
            "name": "ab",
            "profile": "custom",
            "server": {"container_name": "vllm-bench-ab"},
            "variants": [
                {"name": "a", "env": {"X": "1"}, "container_name": "vllm-bench-ab-a", **variant_extra.get("a", {})},
                {"name": "b", "env": {"X": "0"}, "container_name": "vllm-bench-ab-b", **variant_extra.get("b", {})},
            ],
        }
    )


class _FakeRunner:
    calls: list[RunSpec] = []

    def __init__(self, spec, out, **kw):
        self.spec = spec

    def run(self):
        _FakeRunner.calls.append(self.spec)
        return self.spec.name


@pytest.fixture
def fake_runner(monkeypatch):
    _FakeRunner.calls = []
    monkeypatch.setattr(runner_mod, "Runner", _FakeRunner)
    return _FakeRunner


def test_shared_endpoint_variants_refused_without_choice(fake_runner, tmp_path: Path) -> None:
    with pytest.raises(runner_mod.RunRefused, match="--variant"):
        runner_mod.run_spec(_ab(), tmp_path)
    assert fake_runner.calls == []


def test_variant_flag_runs_only_that_variant(fake_runner, tmp_path: Path) -> None:
    runner_mod.run_spec(_ab(), tmp_path, variant="b")
    assert [s.name for s in fake_runner.calls] == ["ab__b"]
    assert fake_runner.calls[0].server.env == {"X": "0"}
    with pytest.raises(ValueError, match="unknown variant"):
        runner_mod.run_spec(_ab(), tmp_path, variant="zzz")


def test_pause_prompts_before_each_variant(fake_runner, tmp_path: Path) -> None:
    prompts: list[str] = []
    runner_mod.run_spec(_ab(), tmp_path, pause_between_variants=True, prompt=lambda m: prompts.append(m) or "")
    assert len(prompts) == 2 and "'a'" in prompts[0] and "X=1" in prompts[0] and "'b'" in prompts[1]
    assert [s.name for s in fake_runner.calls] == ["ab__a", "ab__b"]


def test_per_variant_endpoints_run_back_to_back(fake_runner, tmp_path: Path) -> None:
    spec = _ab(
        a={"endpoint": {"base_url": "http://127.0.0.1:8201/v1", "metrics_url": "http://127.0.0.1:8201/metrics"}},
        b={"endpoint": {"base_url": "http://127.0.0.1:8202/v1", "metrics_url": "http://127.0.0.1:8202/metrics"}},
    )
    runner_mod.run_spec(spec, tmp_path)
    assert [s.endpoint.base_url for s in fake_runner.calls] == ["http://127.0.0.1:8201/v1", "http://127.0.0.1:8202/v1"]


def test_apply_variant_endpoint_override() -> None:
    spec = _ab(a={"endpoint": {"base_url": "http://127.0.0.1:8201/v1"}})
    one = apply_variant(spec, spec.variants[0])
    assert one.endpoint.base_url == "http://127.0.0.1:8201/v1"
    # metrics/health follow the variant's own base_url, not the parent's port.
    assert one.endpoint.metrics_url == "http://127.0.0.1:8201/metrics"
    assert one.endpoint.health_url == "http://127.0.0.1:8201/health"
    assert spec.endpoint.health_url == "http://127.0.0.1:8000/health"
    assert one.variants == []
