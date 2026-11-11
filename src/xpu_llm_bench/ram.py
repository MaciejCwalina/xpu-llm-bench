"""Host RAM watcher. Aborts when MemAvailable falls through the floor."""

from __future__ import annotations

import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from xpu_llm_bench.safety import SafetyError, assert_no_xpu_smi, assert_not_intentional_stop

MEMINFO_RE = re.compile(r"^MemAvailable:\s+(\d+)\s+kB", re.MULTILINE)


class RamFloorError(SafetyError):
    """MemAvailable dropped to or below the configured floor."""


def parse_memavailable_kib(meminfo: str) -> int:
    m = MEMINFO_RE.search(meminfo)
    if not m:
        raise ValueError("MemAvailable not found in meminfo text")
    return int(m.group(1))


def kib_to_gib(kib: int | float) -> float:
    return float(kib) / (1024 * 1024)


@dataclass
class RamSample:
    mem_available_kib: int
    t: float


class RamSource:
    def read_kib(self) -> int:
        raise NotImplementedError


class LocalMemInfo(RamSource):
    def __init__(self, path: str = "/proc/meminfo"):
        assert_not_intentional_stop(path)
        self.path = path

    def read_kib(self) -> int:
        text = open(self.path, encoding="utf-8").read()
        return parse_memavailable_kib(text)


class SshMemInfo(RamSource):
    """Read /proc/meminfo over SSH. Host and extra args come from the spec / env.

    Never embed keys or passwords. Uses the invoking user's SSH agent / config.
    """

    def __init__(self, ssh_host: str, extra_args: list[str] | None = None, remote_path: str = "/proc/meminfo"):
        if not ssh_host or ssh_host.startswith("TODO"):
            raise ValueError("ssh_host is not configured (TODO)")
        self.ssh_host = ssh_host
        self.extra_args = list(extra_args or [])
        self.remote_path = remote_path

    def read_kib(self) -> int:
        cmd = ["ssh", *self.extra_args, self.ssh_host, "cat", self.remote_path]
        assert_no_xpu_smi(cmd)
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=15)
        if proc.returncode != 0:
            raise RuntimeError(f"ssh meminfo failed: {proc.stderr.strip()}")
        return parse_memavailable_kib(proc.stdout)


@dataclass
class RamWatcher:
    floor_gib: float
    source: RamSource
    interval_s: float = 0.5
    samples: list[RamSample] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _min_kib: int | None = None
    _error: Exception | None = None
    on_floor: Callable[[RamSample], None] | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="ram-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def min_gib(self) -> float | None:
        if self._min_kib is None:
            return None
        return kib_to_gib(self._min_kib)

    def check_now(self) -> RamSample:
        sample = RamSample(mem_available_kib=self.source.read_kib(), t=time.time())
        self.samples.append(sample)
        if self._min_kib is None or sample.mem_available_kib < self._min_kib:
            self._min_kib = sample.mem_available_kib
        if kib_to_gib(sample.mem_available_kib) <= self.floor_gib:
            err = RamFloorError(
                f"MemAvailable {kib_to_gib(sample.mem_available_kib):.3f} GiB "
                f"is at or below the {self.floor_gib:.3f} GiB floor"
            )
            self._error = err
            if self.on_floor:
                self.on_floor(sample)
            raise err
        return sample

    def raise_if_tripped(self) -> None:
        if self._error:
            raise self._error

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.check_now()
            except RamFloorError:
                return
            except Exception as exc:  # noqa: BLE001
                self._error = exc
                return
            self._stop.wait(self.interval_s)
