from pathlib import Path

import pytest

from xpu_llm_bench import site
from xpu_llm_bench.import_legacy import import_row
from xpu_llm_bench.runner import safety_preflight
from xpu_llm_bench.safety import SafetyError, is_protected_container
from xpu_llm_bench.spec import RunSpec, load_spec

SPECS = Path(__file__).resolve().parents[1] / "specs"
EXAMPLE = Path(__file__).resolve().parents[1] / "bench-site.example.yaml"


def _use(monkeypatch, path: Path) -> None:
    monkeypatch.setenv(site.SITE_ENV, str(path))
    site.clear_cache()


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "site.yaml"
    p.write_text(
        "host: lab-box\n"
        "ssh_host: lab-alias\n"
        "protected_containers: [prod-stack-a]\n"
        "protected_patterns: ['gateway']\n"
        "forbidden_ports: [9100]\n"
        "hold_command: /opt/hold/bench-hold\n"
        "quiet_hold_file: /srv/ops/QUIET_HOLD\n"
    )
    return p


def _spec(base: str = "http://127.0.0.1:8102", container: str = "vllm-bench-x") -> RunSpec:
    return RunSpec.model_validate(
        {
            "name": "s",
            "endpoint": {"base_url": base + "/v1", "metrics_url": base + "/metrics", "health_url": base + "/health"},
            "server": {"container_name": container},
        }
    )


def test_no_site_config_keeps_neutral_defaults() -> None:
    assert site.load_site().protected_containers == []
    assert is_protected_container("vllm-production")
    assert not is_protected_container("prod-stack-a")
    safety_preflight(_spec())


def test_site_config_adds_protection(monkeypatch, tmp_path: Path) -> None:
    _use(monkeypatch, _write(tmp_path))
    assert is_protected_container("prod-stack-a")
    assert is_protected_container("my-Gateway-1")
    assert is_protected_container("vllm-production")  # built-ins stay
    with pytest.raises(SafetyError, match="9100"):
        safety_preflight(_spec("http://127.0.0.1:9100"))
    with pytest.raises(SafetyError, match="8001"):
        safety_preflight(_spec("http://127.0.0.1:8001"))
    with pytest.raises(SafetyError):
        safety_preflight(_spec(container="prod-stack-a"))
    row = import_row({"users": 1, "ctx": 8192, "tok/s": 10.0}, port=9100)
    assert "PROTECTED_PORT" in row.flags


def test_site_fills_unset_spec_values(monkeypatch, tmp_path: Path) -> None:
    _use(monkeypatch, _write(tmp_path))
    spec = load_spec(SPECS / "qsa-full-fix-mtp6.yaml")
    assert spec.hardware.host == "lab-box"
    assert spec.access.ssh_host == "lab-alias"
    assert spec.safety.hold_command == "/opt/hold/bench-hold"
    assert spec.safety.quiet_hold_file == "/srv/ops/QUIET_HOLD"


def test_bench_host_env(monkeypatch) -> None:
    monkeypatch.setenv("BENCH_HOST", "env-box")
    spec = load_spec(SPECS / "qsa-full-fix-mtp6.yaml")
    assert spec.hardware.host == "env-box"
    assert spec.access.ssh_host == "env-box"


def test_missing_site_file_is_an_error(monkeypatch, tmp_path: Path) -> None:
    _use(monkeypatch, tmp_path / "nope.yaml")
    with pytest.raises(FileNotFoundError):
        site.load_site()


def test_example_site_file_is_rejected_until_filled_in(monkeypatch, tmp_path: Path) -> None:
    _use(monkeypatch, EXAMPLE)
    with pytest.raises(site.SiteConfigError) as err:
        site.load_site()
    msg = str(err.value)
    assert "host" in msg and "protected_containers[0]" in msg and "forbidden_ports[0]" in msg
    # The example becomes valid once every placeholder is replaced.
    filled = EXAMPLE.read_text()
    for old, new in (("<bench-host>", "lab-box"), ("<production-vllm-container>", "prod-a"),
                     ("<protected-service-pattern>", "gateway"), ("<production-port>", "9100")):
        filled = filled.replace(old, new)
    p = tmp_path / "filled.yaml"
    p.write_text(filled)
    _use(monkeypatch, p)
    cfg = site.load_site()
    assert cfg.forbidden_ports == [9100] and cfg.protected_containers == ["prod-a"]
