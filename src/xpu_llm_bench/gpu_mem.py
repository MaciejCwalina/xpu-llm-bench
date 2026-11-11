"""GPU / container memory collectors that never call xpu-smi."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from typing import Any

from xpu_llm_bench.safety import (
    SafetyError,
    assert_container_allowed,
    assert_no_xpu_smi,
    command_mentions_xpu_smi,
)


def collect_docker_stats(container: str, runner: Callable[..., Any] | None = None) -> str | None:
    assert_container_allowed(container)
    cmd = [
        "docker",
        "stats",
        "--no-stream",
        "--format",
        "{{.MemUsage}}",
        container,
    ]
    assert_no_xpu_smi(cmd)
    run = runner or subprocess.run
    proc = run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    text = (proc.stdout or "").strip()
    return text or None


def collect_cgroup_mem(path: str = "/sys/fs/cgroup/memory.current") -> str | None:
    try:
        raw = open(path, encoding="utf-8").read().strip()
    except OSError:
        return None
    try:
        bytes_ = int(raw)
    except ValueError:
        return None
    return f"{bytes_ / (1024 ** 3):.3f} GiB cgroup"


def collect_gpu_mem(
    container: str | None = None,
    kv_cache_usage_pct: float | None = None,
    extra_command: list[str] | None = None,
    runner: Callable[..., Any] | None = None,
) -> str | None:
    """Compose a short GPU/host-device memory note from allowed sources.

    Never runs xpu-smi. An explicit extra_command is rejected if it would.
    """
    parts: list[str] = []
    if kv_cache_usage_pct is not None:
        parts.append(f"kv_cache {kv_cache_usage_pct:.1f}%")
    if container:
        try:
            stats = collect_docker_stats(container, runner=runner)
        except SafetyError:
            stats = None
        if stats:
            parts.append(f"docker {stats}")
    if extra_command:
        if command_mentions_xpu_smi(extra_command):
            raise SafetyError("Refusing GPU memory command that invokes xpu-smi")
        assert_no_xpu_smi(extra_command)
        run = runner or subprocess.run
        proc = run(extra_command, check=False, capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            parts.append(proc.stdout.strip())
    return "; ".join(parts) if parts else None


def format_optional_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True)
