"""Adapter for the bench host's GPU hold CLI (``bench-hold``).

The path comes from ``safety.hold_command`` or the site config. The CLI is:

    bench-hold acquire <minutes> <reason> [--by NAME]
    bench-hold release
    bench-hold status
    bench-hold wait-idle [timeout_s]

Renew is another acquire with the same --by name (the CLI has no distinct renew).
All operations are dry-run capable. The runner must never force a hold.

Behaviour this adapter is written against:

- ``acquire`` does NOT check for an existing hold. It atomically overwrites
  the hold file. So the runner calls ``status`` first and refuses if anyone
  else holds it, then re-reads ``status`` after acquiring to confirm.
- ``status`` is read-only. It prints ``active by=NAME reason=... until=...``
  and exits 0, or ``released path=... queued=N`` and exits 1.
- ``release`` unlinks the file whoever owns it, then lets the host's queued
  GPU work resume. The runner only releases after ``status`` shows its own name.
- An optional operator quiet-hold file (``safety.quiet_hold_file``) is not
  printed by ``status``; ``quiet_hold_active`` reads it (never writes it).
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xpu_llm_bench.safety import SafetyError, assert_no_xpu_smi


def _expand(command: str) -> str:
    return os.path.expanduser(os.path.expandvars(command))


@dataclass
class HoldResult:
    ok: bool
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    dry_run: bool = False


_ACTIVE_RE = re.compile(r"^active by=(?P<by>\S*) reason=(?P<reason>.*?) until=(?P<until>.*?) acquired=")
_QUIET_RE = re.compile(r"until=(?P<until>[0-9.]+)(?:\s+by=(?P<by>\S+))?")


@dataclass
class HoldState:
    active: bool
    by: str | None = None
    reason: str | None = None
    until: str | None = None
    raw: str = ""
    known: bool = True  # False when status output could not be parsed


def parse_status(result: HoldResult) -> HoldState:
    text = (result.stdout or "").strip()
    if result.dry_run:
        return HoldState(active=False, raw=text)
    first = text.splitlines()[0] if text else ""
    m = _ACTIVE_RE.match(first)
    if result.returncode == 0 and m:
        return HoldState(True, m.group("by"), m.group("reason"), m.group("until"), raw=text)
    if result.returncode == 1 and first.startswith("released"):
        return HoldState(active=False, raw=text)
    return HoldState(active=False, raw=text + (result.stderr or ""), known=False)


def _read_local(path: str) -> str | None:
    try:
        return Path(_expand(path)).read_text(encoding="utf-8")
    except OSError:
        return None


def quiet_hold_active(
    path: str | None,
    now: float | None = None,
    read_file: Callable[[str], str | None] | None = None,
) -> str | None:
    """Return a description if the owner quiet hold file is active, else None."""
    if not path:
        return None
    p = path
    text = (read_file or _read_local)(path)
    if text is None:
        return None
    m = _QUIET_RE.search(text)
    if not m:
        return f"quiet hold file {p} present but unparsed; treating as active"
    until = float(m.group("until"))
    if until > (now if now is not None else time.time()):
        return f"owner quiet hold active until epoch {until:.0f} by={m.group('by') or '?'}"
    return None


class BenchHold:
    def __init__(
        self,
        command: str = "bench-hold",
        name: str = "xpu-llm-bench",
        runner: Callable[..., Any] | None = None,
        dry_run: bool = False,
    ):
        self.command = _expand(command)
        self.name = name
        self.dry_run = dry_run
        self._runner = runner or subprocess.run

    def _run(self, args: list[str], check: bool = False) -> HoldResult:
        cmd = [self.command, *args]
        assert_no_xpu_smi(cmd)
        if self.dry_run:
            return HoldResult(True, cmd, 0, f"[dry-run] {shlex.join(cmd)}\n", "", True)
        proc = self._runner(cmd, check=False, capture_output=True, text=True)
        result = HoldResult(
            ok=proc.returncode == 0,
            command=cmd,
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            dry_run=False,
        )
        if check and not result.ok:
            raise SafetyError(
                f"bench-hold failed ({result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
            )
        return result

    def acquire(self, minutes: int | float, reason: str, by: str | None = None) -> HoldResult:
        args = ["acquire", str(minutes), reason, "--by", by or self.name]
        return self._run(args, check=not self.dry_run)

    def renew(self, minutes: int | float, reason: str, by: str | None = None) -> HoldResult:
        return self.acquire(minutes, reason, by=by)

    def release(self) -> HoldResult:
        return self._run(["release"])

    def status(self) -> HoldResult:
        return self._run(["status"])

    def state(self) -> HoldState:
        return parse_status(self.status())

    def guard_and_acquire(
        self,
        minutes: int | float,
        reason: str,
        quiet_hold_file: str | None = None,
        read_file: Callable[[str], str | None] | None = None,
    ) -> HoldState:
        """status -> refuse if someone else holds it -> acquire -> confirm."""
        quiet = quiet_hold_active(quiet_hold_file, read_file=read_file)
        if quiet and not self.dry_run:
            raise SafetyError(f"Refusing to start: {quiet}")
        before = self.state()
        if not before.known:
            raise SafetyError(f"Refusing to start: cannot parse bench-hold status: {before.raw!r}")
        if before.active and before.by != self.name:
            raise SafetyError(
                f"Refusing to start: bench-hold is held by {before.by!r} "
                f"(reason={before.reason!r}, until {before.until}). "
                "acquire would silently overwrite it."
            )
        self.acquire(minutes, reason)
        after = self.state()
        if not self.dry_run and (not after.active or after.by != self.name):
            raise SafetyError(
                f"bench-hold not ours after acquire (status: {after.raw!r}); not running"
            )
        return after

    def release_if_ours(self) -> tuple[bool, str]:
        """Release only when status shows our own name. Returns (released, note)."""
        st = self.state()
        if self.dry_run:
            self.release()
            return True, "dry-run"
        if st.active and st.by == self.name:
            self.release()
            return True, "released"
        if not st.active:
            return False, "hold already gone (expired or released by someone else)"
        return False, f"hold now belongs to {st.by!r}; not releasing it"

    def wait_idle(self, timeout_s: float = 120.0) -> HoldResult:
        return self._run(["wait-idle", str(timeout_s)])


def hold_exists_on_path(command: str = "bench-hold") -> bool:
    path = Path(_expand(command))
    return path.exists()
