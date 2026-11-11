from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_site_config(monkeypatch):
    """Tests never read a developer's real site config or $BENCH_HOST."""
    from xpu_llm_bench import site

    monkeypatch.setenv(site.SITE_ENV, "")
    monkeypatch.delenv("BENCH_HOST", raising=False)
    site.clear_cache()
    yield
    site.clear_cache()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class MetricsState:
    def __init__(self) -> None:
        self.drafts = 0.0
        self.draft_tokens = 0.0
        self.accepted = 0.0
        self.generation = 0.0
        self.prompt = 0.0
        self.preemptions = 0.0
        self.prefix_hits = 0.0
        self.prefix_queries = 0.0
        self.running = 0.0
        self.waiting = 0.0
        self.queue_sum = 0.0
        self.queue_count = 0.0
        self.kv_usage = 12.5
        self.lock = threading.Lock()
        # Test controls (unused by the original scaffold tests).
        self.chat_requests = 0
        self.fail_from: int | None = None  # 0-based chat request index that starts failing
        self.max_model_len: int | None = None  # reported by /v1/models when set
        self.reject_above: int | None = None  # 400 "maximum context length" above this many prompt tokens
        self.cache_blocks: tuple[int, int] | None = None  # (num_gpu_blocks, block_size) for cache_config_info

    def bump(self, output_tokens: int, prompt_tokens: int, chunks: list[int] | None = None) -> None:
        """Account one request. ``chunks`` = tokens per streamed step; the first
        chunk is the prefill step, every later chunk is one MTP-6 verify step
        carrying 1 bonus token + (n-1) accepted drafts."""
        with self.lock:
            verify = (chunks or [1] * max(output_tokens, 1))[1:]
            self.drafts += len(verify)
            self.draft_tokens += len(verify) * 6
            self.accepted += sum(n - 1 for n in verify)
            self.generation += output_tokens
            self.prompt += prompt_tokens
            self.prefix_queries += 1
            self.prefix_hits += 1
            self.queue_sum += 0.01
            self.queue_count += 1
            self.running = 1.0
            self.waiting = 0.0

    def render(self) -> str:
        with self.lock:
            model = 'model_name="mock"'
            return "\n".join(
                [
                    "# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.",
                    "# TYPE vllm:spec_decode_num_drafts_total counter",
                    f'vllm:spec_decode_num_drafts_total{{engine="0",{model}}} {self.drafts}',
                    "# HELP vllm:spec_decode_num_draft_tokens_total Number of draft tokens.",
                    f'vllm:spec_decode_num_draft_tokens_total{{engine="0",{model}}} {self.draft_tokens}',
                    "# HELP vllm:spec_decode_num_accepted_tokens_total Number of accepted tokens.",
                    f'vllm:spec_decode_num_accepted_tokens_total{{engine="0",{model}}} {self.accepted}',
                    *[
                        f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",{model},position="{p}"}} {self.accepted * w}'
                        for p, w in enumerate((0.4, 0.25, 0.15, 0.1, 0.06, 0.04))
                    ],
                    f"vllm:generation_tokens_total{{{model}}} {self.generation}",
                    f"vllm:prompt_tokens_total{{{model}}} {self.prompt}",
                    f"vllm:num_preemptions_total{{{model}}} {self.preemptions}",
                    f'vllm:prefix_cache_hits_total{{{model}}} {self.prefix_hits}',
                    f'vllm:prefix_cache_queries_total{{{model}}} {self.prefix_queries}',
                    # KV usage counts blocks held by running requests: 0 when idle.
                    f"vllm:gpu_cache_usage_perc {self.kv_usage if self.running else 0.0}",
                    *(
                        [f'vllm:cache_config_info{{block_size="{self.cache_blocks[1]}",'
                         f'num_gpu_blocks="{self.cache_blocks[0]}"}} 1.0']
                        if self.cache_blocks else []
                    ),
                    f"vllm:num_requests_running {self.running}",
                    f"vllm:num_requests_waiting {self.waiting}",
                    f"vllm:request_queue_time_seconds_sum {self.queue_sum}",
                    f"vllm:request_queue_time_seconds_count {self.queue_count}",
                    "",
                ]
            )


def _sse(token: str, usage: dict | None = None) -> bytes:
    body: dict = {"choices": [{"delta": {"content": token}}]}
    if usage is not None:
        body["usage"] = usage
    return f"data: {json.dumps(body)}\n\n".encode()


def make_handler(state: MetricsState, logs: list[str]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/metrics"):
                body = state.render().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.endswith("/v1/models"):
                item = {"id": "m", "object": "model"}
                if state.max_model_len:
                    item["max_model_len"] = state.max_model_len
                body = json.dumps({"object": "list", "data": [item]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/health"):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode())
            except json.JSONDecodeError:
                payload = {}
            if self.path.endswith("/tokenize"):
                if payload.get("messages") is not None:
                    # Same count the chat endpoint reports as prompt_tokens.
                    chars = sum(len(m.get("content") or "") for m in payload["messages"])
                    body = json.dumps({"count": max(1, chars // 4)}).encode()
                else:
                    prompt = payload.get("prompt") or ""
                    body = json.dumps({"count": max(1, len(prompt) // 4)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.endswith("/chat/completions"):
                with state.lock:
                    idx = state.chat_requests
                    state.chat_requests += 1
                if state.fail_from is not None and idx >= state.fail_from:
                    logs.append(
                        "AssertionError: Expected spec_token == num_spec_decodes * (num_speculative_tokens + 1)"
                    )
                    body = b'{"error": "EngineCore encountered an issue"}'
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                max_tokens = int(payload.get("max_tokens") or 8)
                messages = payload.get("messages") or []
                prompt_tokens = max(1, sum(len(m.get("content") or "") for m in messages) // 4)
                if state.reject_above is not None and prompt_tokens > state.reject_above:
                    msg = (
                        f"This model's maximum context length is {state.reject_above} tokens. However, you "
                        f"requested {prompt_tokens + max_tokens} tokens. Please reduce the length of the messages."
                    )
                    body = json.dumps({"object": "error", "message": msg, "code": 400}).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                # MTP-6 style stream: prefill chunk of 1 token, then verify
                # steps of 4 tokens (1 bonus + 3 accepted drafts).
                chunks: list[int] = [1]
                while sum(chunks) < max_tokens:
                    chunks.append(min(4, max_tokens - sum(chunks)))
                state.bump(max_tokens, prompt_tokens, chunks)
                logs.append(f"QSA_GRAPH width=65536 n={max_tokens}")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                done = 0
                for n in chunks:
                    done += n
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": done,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    }
                    self.wfile.write(_sse("x" * n, usage))
                    self.wfile.flush()
                    time.sleep(0.001)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                with state.lock:
                    state.running = 0.0
                return
            self.send_response(404)
            self.end_headers()

    return Handler


@pytest.fixture
def mock_server():
    state = MetricsState()
    logs: list[str] = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state, logs))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        yield {"base": base, "state": state, "logs": logs, "port": port}
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def _fast_load_probe(monkeypatch):
    """The pre-run load check waits 2 s between /metrics scrapes; not in tests."""
    monkeypatch.setattr("xpu_llm_bench.runner.DEFAULT_LOAD_PROBE_S", 0.0)
