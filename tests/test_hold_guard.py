import pytest

from xpu_llm_bench.hold import BenchHold, HoldResult, parse_status, quiet_hold_active
from xpu_llm_bench.safety import SafetyError


class P:
    def __init__(self, code=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = code, out, err


ACTIVE_OTHER = "active by=other-job reason=other-job-matrix until=Wed Nov 11 1:27:42 PM EST acquired=Wed Nov 11 9:27:42 AM EST queued=0\n"
RELEASED = "released path=/run/user/1000/bench-hold.json queued=0\n"


def scripted(responses):
    calls = []

    def runner(cmd, **kw):
        calls.append(cmd[1:])
        return responses.pop(0)

    return runner, calls


def test_parse_status_variants() -> None:
    st = parse_status(HoldResult(True, [], 0, ACTIVE_OTHER, ""))
    assert st.active and st.by == "other-job" and st.reason == "other-job-matrix"
    st = parse_status(HoldResult(False, [], 1, RELEASED, ""))
    assert not st.active and st.known
    st = parse_status(HoldResult(False, [], 2, "", "Traceback"))
    assert not st.known


def test_refuses_when_someone_else_holds_it_and_never_acquires() -> None:
    runner, calls = scripted([P(0, ACTIVE_OTHER)])
    hold = BenchHold("/x/bench-hold", name="xpu-llm-bench", runner=runner)
    with pytest.raises(SafetyError, match="other-job"):
        hold.guard_and_acquire(30, "ladder")
    assert calls == [["status"]]  # acquire would have silently overwritten it


def test_acquires_when_free_and_confirms_ownership() -> None:
    mine = ACTIVE_OTHER.replace("other-job ", "xpu-llm-bench ", 1)
    runner, calls = scripted([P(1, RELEASED), P(0, "acquired"), P(0, mine)])
    hold = BenchHold("/x/bench-hold", name="xpu-llm-bench", runner=runner)
    st = hold.guard_and_acquire(30, "ladder")
    assert st.by == "xpu-llm-bench"
    assert calls == [["status"], ["acquire", "30", "ladder", "--by", "xpu-llm-bench"], ["status"]]


def test_refuses_if_status_after_acquire_is_not_ours() -> None:
    runner, _ = scripted([P(1, RELEASED), P(0, "acquired"), P(0, ACTIVE_OTHER)])
    hold = BenchHold("/x/bench-hold", name="xpu-llm-bench", runner=runner)
    with pytest.raises(SafetyError, match="not ours"):
        hold.guard_and_acquire(30, "ladder")


def test_unparseable_status_refuses() -> None:
    runner, _ = scripted([P(2, "", "ImportError")])
    hold = BenchHold("/x/bench-hold", name="xpu-llm-bench", runner=runner)
    with pytest.raises(SafetyError, match="cannot parse"):
        hold.guard_and_acquire(30, "ladder")


def test_release_only_if_ours() -> None:
    runner, calls = scripted([P(0, ACTIVE_OTHER)])
    hold = BenchHold("/x/bench-hold", name="xpu-llm-bench", runner=runner)
    released, note = hold.release_if_ours()
    assert not released and "other-job" in note
    assert ["release"] not in calls


def test_quiet_hold_file(tmp_path) -> None:
    f = tmp_path / "QUIET_HOLD"
    f.write_text("quiet-hold until=4102444800.000 by=operator\n")
    assert "quiet hold active" in (quiet_hold_active(str(f), now=1794400000) or "")
    f.write_text("quiet-hold until=1000.000 by=operator\n")
    assert quiet_hold_active(str(f), now=1794400000) is None
    assert quiet_hold_active(str(tmp_path / "missing")) is None
    runner, calls = scripted([])
    f.write_text("quiet-hold until=4102444800.000 by=operator\n")
    with pytest.raises(SafetyError, match="quiet hold"):
        BenchHold("/x/bench-hold", runner=runner).guard_and_acquire(5, "x", quiet_hold_file=str(f))
    assert calls == []
