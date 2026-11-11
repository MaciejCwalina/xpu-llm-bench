"""Docker container adapter with a production-name guard and dry-run mode."""

from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from xpu_llm_bench.safety import SafetyError, assert_container_allowed, assert_no_xpu_smi


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    dry_run: bool = False


class ContainerAdapter:
    def __init__(
        self,
        allowed: Iterable[str] | None = None,
        dry_run: bool = False,
        runner: Callable[..., Any] | None = None,
    ):
        self.allowed = list(allowed or [])
        self.dry_run = dry_run
        self._runner = runner or subprocess.run

    def _run(self, args: Sequence[str], check: bool = True) -> CommandResult:
        cmd = list(args)
        assert_no_xpu_smi(cmd)
        if self.dry_run:
            return CommandResult(cmd, 0, f"[dry-run] {shlex.join(cmd)}\n", "", True)
        proc = self._runner(cmd, check=False, capture_output=True, text=True)
        result = CommandResult(cmd, proc.returncode, proc.stdout or "", proc.stderr or "", False)
        if check and result.returncode != 0:
            raise SafetyError(
                f"command failed ({result.returncode}): {shlex.join(cmd)}\n{result.stderr}"
            )
        return result

    def _guard(self, name: str) -> str:
        return assert_container_allowed(name, self.allowed or None)

    def inspect(self, name: str) -> CommandResult:
        self._guard(name)
        return self._run(["docker", "inspect", name], check=False)

    def image_digest(self, image: str) -> str | None:
        result = self._run(
            ["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", image],
            check=False,
        )
        if result.dry_run or result.returncode != 0:
            return None
        try:
            digests = json.loads(result.stdout.strip() or "[]")
        except json.JSONDecodeError:
            return None
        if not digests:
            return None
        return str(digests[0])

    def logs_since(self, name: str, since: str | None = None, tail: int = 2000) -> str:
        self._guard(name)
        args = ["docker", "logs", "--tail", str(tail)]
        if since:
            args.extend(["--since", since])
        args.append(name)
        result = self._run(args, check=False)
        return (result.stdout or "") + (result.stderr or "")

    def start(self, name: str) -> CommandResult:
        self._guard(name)
        return self._run(["docker", "start", name])

    def stop(self, name: str, timeout_s: int = 30) -> CommandResult:
        self._guard(name)
        return self._run(["docker", "stop", "-t", str(timeout_s), name])

    def run(self, name: str, image: str, args: Sequence[str], env: dict[str, str] | None = None) -> CommandResult:
        self._guard(name)
        cmd = ["docker", "run", "-d", "--name", name]
        for key, val in (env or {}).items():
            cmd.extend(["-e", f"{key}={val}"])
        cmd.append(image)
        cmd.extend(list(args))
        return self._run(cmd)
