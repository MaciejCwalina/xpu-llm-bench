from types import SimpleNamespace

import pytest

from xpu_llm_bench import runner as runner_module
from xpu_llm_bench.runner import METRICS_FETCH_ATTEMPTS, Runner


class FlakyClient:
    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def get_text(self, url, timeout=None):
        self.calls += 1
        if self.calls <= self.failures:
            return 0, "TimeoutError('timed out')"
        return 200, "vllm:num_requests_running 0\n"


def _runner(client) -> Runner:
    runner = Runner.__new__(Runner)
    runner.fetch_metrics = None
    runner.client = client
    runner.spec = SimpleNamespace(endpoint=SimpleNamespace(metrics_url="http://127.0.0.1:1/metrics"))
    return runner


def test_metrics_fetch_recovers_from_transient_failures(monkeypatch):
    sleeps = []
    monkeypatch.setattr(runner_module.time, "sleep", sleeps.append)
    client = FlakyClient(failures=METRICS_FETCH_ATTEMPTS - 1)
    assert "num_requests_running" in _runner(client)._metrics_text()
    assert client.calls == METRICS_FETCH_ATTEMPTS
    assert sleeps == sorted(sleeps) and len(sleeps) == METRICS_FETCH_ATTEMPTS - 1


def test_metrics_fetch_gives_up_after_all_attempts(monkeypatch):
    monkeypatch.setattr(runner_module.time, "sleep", lambda seconds: None)
    client = FlakyClient(failures=METRICS_FETCH_ATTEMPTS)
    with pytest.raises(RuntimeError, match=f"after {METRICS_FETCH_ATTEMPTS} attempts"):
        _runner(client)._metrics_text()
    assert client.calls == METRICS_FETCH_ATTEMPTS
