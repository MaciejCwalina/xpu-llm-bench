import pytest

from xpu_llm_bench.hostexec import HostExec
from xpu_llm_bench.hostinfo import HOST_SCAN, parse_host_scan
from xpu_llm_bench.procmem import PROC_SCAN
from xpu_llm_bench.runner import safety_preflight
from xpu_llm_bench.safety import (
    SafetyError,
    assert_endpoint_allowed,
    assert_test_container_name,
    bound_host_ports,
)
from xpu_llm_bench.spec import RunSpec


def _spec(**server):
    return RunSpec.model_validate(
        {
            "name": "s",
            "endpoint": {
                "base_url": "http://127.0.0.1:8102/v1",
                "metrics_url": "http://127.0.0.1:8102/metrics",
                "health_url": "http://127.0.0.1:8102/health",
            },
            "server": {"container_name": "vllm-bench-x", **server},
        }
    )


def test_production_port_refused_everywhere() -> None:
    with pytest.raises(SafetyError, match="8001"):
        assert_endpoint_allowed(["http://127.0.0.1:8001/v1"], [8001])
    spec = _spec()
    spec.endpoint.metrics_url = "http://localhost:8001/metrics"
    with pytest.raises(SafetyError):
        safety_preflight(spec)
    safety_preflight(_spec())  # 8102 is fine


def test_production_container_refused() -> None:
    with pytest.raises(SafetyError):
        safety_preflight(_spec(container_name="vllm-production"))
    with pytest.raises(SafetyError):
        safety_preflight(_spec(container_name="gitea-runner-1"))


def test_managed_container_needs_test_prefix() -> None:
    with pytest.raises(SafetyError, match="test prefix"):
        assert_test_container_name("my-vllm", ["vllm-bench-"])
    assert assert_test_container_name("vllm-bench-qsa", ["vllm-bench-"]) == "vllm-bench-qsa"
    with pytest.raises(SafetyError):
        safety_preflight(_spec(container_name="combined-test-v3", manage_container=True))


def test_bound_host_ports() -> None:
    bridge = {"HostConfig": {"PortBindings": {"8000/tcp": [{"HostIp": "", "HostPort": "8001"}]}}}
    assert bound_host_ports(bridge) == {8001}
    host = {"HostConfig": {"NetworkMode": "host"}, "Config": {"Cmd": ["--model", "/model", "--port", "8001"]}}
    assert bound_host_ports(host) == {8001}
    host2 = {"HostConfig": {"NetworkMode": "host"}, "Args": ["--port=8102"]}
    assert bound_host_ports(host2) == {8102}


def test_no_xpu_smi_anywhere() -> None:
    ex = HostExec(runner=lambda *a, **k: None)
    with pytest.raises(SafetyError):
        ex.run(["xpu-smi", "discovery"])
    with pytest.raises(SafetyError):
        ex.sh("/usr/bin/xpu-smi dump")
    assert "xpu-smi" not in HOST_SCAN and "xpu-smi" not in PROC_SCAN


def test_ssh_wrapping() -> None:
    ex = HostExec("ssh", "bench-host")
    argv = ex.argv(["docker", "logs", "--tail", "10", "vllm-bench-x"])
    assert argv[:3] == ["ssh", "-o", "BatchMode=yes"]
    assert argv[-1] == "docker logs --tail 10 vllm-bench-x"
    with pytest.raises(ValueError):
        HostExec("ssh", None)


def test_host_scan_parse() -> None:
    text = (
        "kernel=7.0.0-31-generic\nmem_total_kib=26214400\ncpu=AMD EPYC 7532 32-Core Processor\nvcpus=32\n"
        "gpu=card1 0x8086 0xe211 driver=xe\ngpu=card2 0x8086 0xe211 driver=xe\n"
        "pkg=intel-opencl-icd 26.09.37435.12-1\n"
    )
    d = parse_host_scan(text)
    assert d["host_ram_gib"] == 25.0
    assert len(d["gpus"]) == 2 and "0xe211" in d["gpus"][0]
    assert d["packages"]["intel-opencl-icd"].startswith("26.09")
