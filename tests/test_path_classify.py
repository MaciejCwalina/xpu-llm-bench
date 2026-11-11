from pathlib import Path

from xpu_llm_bench.path_classify import classify_logs
from xpu_llm_bench.spec import LogPatterns

FIX = Path(__file__).resolve().parent / "fixtures"


def test_mixed_when_width_and_skip() -> None:
    text = (FIX / "qsa_logs.txt").read_text()
    got = classify_logs(text, requested="qsa")
    assert got.path == "mixed"
    assert any("width=65536" in line for line in got.evidence)
    assert any("skip starts=" in line for line in got.evidence)


def test_fallback_when_only_skip_and_qsa_requested() -> None:
    logs = ["QSA_GRAPH skip starts=(14,) n=14"]
    got = classify_logs(logs, requested="qsa")
    assert got.path == "fallback"


def test_qsa_when_only_width() -> None:
    logs = ["QSA_GRAPH width=65536"]
    got = classify_logs(logs, requested="qsa")
    assert got.path == "qsa"
    assert got.active_widths == ["65536"]


def test_dense_when_requested_dense() -> None:
    got = classify_logs(["ready"], requested="dense")
    assert got.path == "dense"


def test_unknown_without_evidence() -> None:
    got = classify_logs(["hello"], requested="auto")
    assert got.path == "unknown"


def test_custom_patterns() -> None:
    pats = LogPatterns(qsa_active=r"QSA on w=(\d+)", qsa_skip=r"QSA off n=(\S+)")
    got = classify_logs(["QSA on w=9"], pats, requested="qsa")
    assert got.path == "qsa"
