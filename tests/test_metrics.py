from pathlib import Path

from xpu_llm_bench.metrics import (
    inter_token_latencies_ms,
    mtp_window,
    parse_prometheus,
    percentile,
    stream_stats,
    system_window,
)
from xpu_llm_bench.spec import MetricNameMap

FIX = Path(__file__).resolve().parent / "fixtures"
NAMES = MetricNameMap()


def test_parse_and_mtp_formulas() -> None:
    before = parse_prometheus((FIX / "metrics_before.txt").read_text())
    after = parse_prometheus((FIX / "metrics_after.txt").read_text())
    mtp = mtp_window(before, after, NAMES)
    # 2064-1062 = 1002 drafts; 12384-6372 = 6012 draft tokens; 5262-2714 = 2548 accepted
    assert mtp.drafts == 1002
    assert mtp.draft_tokens == 6012
    assert mtp.accepted_drafts == 2548
    assert mtp.accepted_drafts_per_step == pytest_approx(2548 / 1002)
    assert mtp.tokens_per_step == pytest_approx(1 + 2548 / 1002)
    assert mtp.hit_rate == pytest_approx(2548 / 6012)
    # per_pos must not have been added into accepted
    assert mtp.accepted_drafts != 2548 + (1654 - 839)


def pytest_approx(val, rel=1e-9):
    import pytest

    return pytest.approx(val, rel=rel)


def test_system_window_and_prefix_hit() -> None:
    before = parse_prometheus((FIX / "metrics_before.txt").read_text())
    after = parse_prometheus((FIX / "metrics_after.txt").read_text())
    sysm = system_window(before, after, NAMES)
    assert sysm.preemptions == 1
    # hits 18-10=8, queries 24-20=4
    assert sysm.prefix_cache_hit_rate == pytest_approx(8 / 4)
    assert sysm.prefix_hits == 8
    assert sysm.prefix_queries == 4
    assert sysm.kv_cache_usage_pct == 55.5
    assert sysm.running_reqs == 4
    assert sysm.waiting_reqs == 1
    assert sysm.queue_time_ms == pytest_approx((0.5 / 4) * 1000)


def test_percentiles_and_itl() -> None:
    assert percentile([1, 2, 3, 4], 50) == pytest_approx(2.5)
    assert percentile([10], 99) == 10
    assert percentile([], 50) is None
    itls = inter_token_latencies_ms([1.000, 1.010, 1.030])
    assert itls[0] == pytest_approx(10.0)
    assert itls[1] == pytest_approx(20.0)


def test_stream_stats_ttft_and_prefill() -> None:
    stats = stream_stats(
        t_send=10.0,
        first_token_at=10.5,
        t_end=11.5,
        token_times=[10.5, 10.6, 10.7, 11.5],
        output_tokens=4,
        prompt_tokens=1000,
        cached_tokens=200,
        new_prompt_tokens=800,
    )
    assert stats.ttft_ms == pytest_approx(500)
    assert stats.e2e_ms == pytest_approx(1500)
    assert stats.decode_tok_s == pytest_approx(3 / 1.0)
    assert stats.prefill_tok_s == pytest_approx(800 / 0.5)
    assert stats.itl_p50_ms is not None
