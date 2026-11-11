from pathlib import Path

import pytest

from xpu_llm_bench.ram import LocalMemInfo, RamFloorError, RamWatcher, parse_memavailable_kib
from xpu_llm_bench.safety import SafetyError


def test_parse_memavailable(tmp_path: Path) -> None:
    text = "MemTotal: 25000000 kB\nMemAvailable: 3145728 kB\n"
    assert parse_memavailable_kib(text) == 3145728
    path = tmp_path / "meminfo"
    path.write_text(text)
    src = LocalMemInfo(str(path))
    assert src.read_kib() == 3145728


def test_watcher_trips_floor(tmp_path: Path) -> None:
    path = tmp_path / "meminfo"
    # 1.5 GiB available, floor is 2 GiB
    path.write_text("MemAvailable: 1572864 kB\n")
    watcher = RamWatcher(floor_gib=2.0, source=LocalMemInfo(str(path)))
    with pytest.raises(RamFloorError, match="floor"):
        watcher.check_now()
    assert watcher.min_gib() is not None
    assert watcher.min_gib() < 2.0


def test_refuses_intentional_stop_path() -> None:
    from xpu_llm_bench.ram import LocalMemInfo

    with pytest.raises(SafetyError):
        LocalMemInfo("/run/vllm-intentional-stop")
