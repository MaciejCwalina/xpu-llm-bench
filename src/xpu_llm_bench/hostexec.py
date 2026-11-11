"""Run read-only commands on the bench host, locally or over SSH, with the safety guards."""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from xpu_llm_bench.safety import assert_no_xpu_smi


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str


class HostExec:
    """Local or SSH command execution. Never runs xpu-smi.

    ``runner`` is injectable for tests (same signature as subprocess.run).
    """

    def __init__(
        self,
        mode: str = "local",
        ssh_host: str | None = None,
        ssh_extra_args: Sequence[str] | None = None,
        runner: Callable[..., Any] | None = None,
    ):
        if mode == "ssh" and (not ssh_host or ssh_host.startswith("TODO")):
            raise ValueError("access.mode is ssh but ssh_host is not configured")
        self.mode = mode
        self.ssh_host = ssh_host
        self.ssh_extra_args = list(ssh_extra_args or [])
        self._runner = runner or subprocess.run

    def argv(self, cmd: Sequence[str]) -> list[str]:
        cmd = list(cmd)
        assert_no_xpu_smi(cmd)
        if self.mode == "ssh":
            return ["ssh", "-o", "BatchMode=yes", *self.ssh_extra_args, str(self.ssh_host), "--", shlex.join(cmd)]
        return cmd

    def run(self, cmd: Sequence[str], timeout: float = 60.0) -> ExecResult:
        full = self.argv(cmd)
        try:
            proc = self._runner(full, check=False, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError as exc:
            return ExecResult(127, "", str(exc))
        except subprocess.TimeoutExpired:
            return ExecResult(124, "", f"timeout after {timeout}s")
        return ExecResult(proc.returncode, proc.stdout or "", proc.stderr or "")

    def sh(self, script: str, timeout: float = 60.0) -> ExecResult:
        """Run a read-only POSIX shell snippet (used for /proc scans over SSH)."""
        assert_no_xpu_smi(script)
        return self.run(["sh", "-c", script], timeout=timeout)

    def read_file(self, path: str) -> str | None:
        res = self.run(["cat", path], timeout=20.0)
        return res.stdout if res.returncode == 0 else None
