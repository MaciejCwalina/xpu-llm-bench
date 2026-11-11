import pytest

from xpu_llm_bench.hold import BenchHold
from xpu_llm_bench.safety import SafetyError


class FakeProc:
    def __init__(self, code=0, stdout="", stderr=""):
        self.returncode = code
        self.stdout = stdout
        self.stderr = stderr


def test_dry_run_does_not_invoke() -> None:
    called = []

    def runner(*args, **kwargs):
        called.append(args)
        return FakeProc()

    hold = BenchHold(command="/bin/bench-hold", name="unit-test", runner=runner, dry_run=True)
    result = hold.acquire(30, "ladder")
    assert result.dry_run
    assert result.command[:4] == ["/bin/bench-hold", "acquire", "30", "ladder"]
    assert "--by" in result.command
    assert "unit-test" in result.command
    assert called == []
    renew = hold.renew(15, "ladder")
    assert renew.command[1] == "acquire"
    rel = hold.release()
    assert rel.command == ["/bin/bench-hold", "release"]


def test_acquire_raises_on_failure() -> None:
    def runner(*args, **kwargs):
        return FakeProc(code=2, stderr="already held")

    hold = BenchHold(command="/bin/bench-hold", runner=runner, dry_run=False)
    with pytest.raises(SafetyError, match="already held"):
        hold.acquire(10, "x")
