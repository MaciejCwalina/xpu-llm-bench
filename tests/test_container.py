import pytest

from xpu_llm_bench.container import ContainerAdapter
from xpu_llm_bench.gpu_mem import collect_gpu_mem
from xpu_llm_bench.safety import SafetyError, is_protected_container


@pytest.mark.parametrize(
    "name",
    [
        "vllm-production",
        "act-runner",
        "github-actions-runner-1",
        "ci-runner",
    ],
)
def test_protected_names(name: str) -> None:
    assert is_protected_container(name)
    adapter = ContainerAdapter(allowed=["vllm-bench-test"], dry_run=True)
    with pytest.raises(SafetyError, match="protected"):
        adapter.stop(name)


def test_refuses_container_not_in_spec() -> None:
    adapter = ContainerAdapter(allowed=["vllm-bench-test"], dry_run=True)
    with pytest.raises(SafetyError, match="not named"):
        adapter.stop("vllm-bench-other")


def test_dry_run_start_allowed() -> None:
    adapter = ContainerAdapter(allowed=["vllm-bench-test"], dry_run=True)
    result = adapter.stop("vllm-bench-test")
    assert result.dry_run
    assert result.command[:2] == ["docker", "stop"]


def test_gpu_mem_rejects_xpu_smi() -> None:
    with pytest.raises(SafetyError, match="xpu-smi"):
        collect_gpu_mem(extra_command=["xpu-smi", "dump", "-d", "0"])
