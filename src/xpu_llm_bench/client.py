"""OpenAI-compatible streaming client (stdlib only). API keys come from the environment."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlsplit


@dataclass
class StreamResult:
    status: int
    text: str
    t_send: float
    t_end: float
    first_token_at: float | None = None
    token_times: list[float] = field(default_factory=list)
    output_tokens: int = 0
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    timed_out: bool = False
    raw_chunks: int = 0
    # Tokens carried by each token-bearing chunk, aligned with token_times.
    # One chunk per engine step, so len(chunk_tokens) - 1 = verify steps.
    chunk_tokens: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    token_ids: list[int] = field(default_factory=list)
    # False if any chunk's token count had to be guessed from text (no
    # token_ids and no continuous usage) -> per-step counts are unreliable.
    per_step_counts: bool = True

    @property
    def chunks(self) -> int:
        return len(self.chunk_tokens)

    @property
    def last_token_at(self) -> float | None:
        return self.token_times[-1] if self.token_times else None


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: float = 600.0,
        opener: urllib.request.OpenerDirector | None = None,
    ):
        self.base_url = base_url.rstrip("/") + "/"
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self._opener = opener or urllib.request.build_opener()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        key = os.environ.get(self.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return urljoin(self.base_url, path.lstrip("/"))

    def request_json(self, method: str, url: str, body: dict[str, Any] | None = None, timeout: float | None = None) -> tuple[int, str]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method, headers=self._headers())
        try:
            with self._opener.open(req, timeout=timeout or self.timeout_s) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError) as exc:
            return 0, repr(exc)

    def get_text(self, url: str, timeout: float | None = None) -> tuple[int, str]:
        return self.request_json("GET", url, None, timeout=timeout or 15.0)

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        stop_at: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> StreamResult:
        url = self._url("chat/completions")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if extra:
            payload.update(extra)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST", headers=self._headers())
        t_send = time.time()
        result = StreamResult(status=0, text="", t_send=t_send, t_end=t_send)
        try:
            with self._opener.open(req, timeout=self.timeout_s) as resp:
                result.status = getattr(resp, "status", 200)
                chunks: list[str] = []
                while True:
                    if stop_at and time.time() >= stop_at:
                        break
                    raw = resp.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload_s = line[5:].strip()
                    if payload_s == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload_s)
                    except json.JSONDecodeError:
                        continue
                    result.raw_chunks += 1
                    now = time.time()
                    usage = obj.get("usage") or {}
                    usage_ct = None
                    if usage:
                        if usage.get("prompt_tokens") is not None:
                            result.prompt_tokens = usage.get("prompt_tokens")
                        if usage.get("completion_tokens") is not None:
                            usage_ct = int(usage.get("completion_tokens"))
                            result.completion_tokens = usage_ct
                        details = usage.get("prompt_tokens_details") or {}
                        if details.get("cached_tokens") is not None:
                            result.cached_tokens = details.get("cached_tokens")
                    piece_all = ""
                    ids_here: list[int] = []
                    for choice in obj.get("choices") or []:
                        delta = choice.get("delta") or {}
                        piece_all += (delta.get("content") or "") + (
                            delta.get("reasoning_content") or delta.get("reasoning") or ""
                        )
                        if choice.get("text"):
                            piece_all += choice["text"]
                        ids_here.extend(choice.get("token_ids") or [])
                        if choice.get("finish_reason"):
                            result.finish_reason = choice.get("finish_reason")
                    # Tokens in this chunk: token_ids if returned, else the
                    # continuous-usage completion_tokens delta, else 1 per
                    # non-empty text chunk (no MTP information available).
                    if ids_here:
                        n_new = len(ids_here)
                        result.token_ids.extend(ids_here)
                    elif usage_ct is not None and usage_ct >= result.output_tokens:
                        n_new = usage_ct - result.output_tokens
                        if n_new == 0 and piece_all:
                            n_new = 1
                    else:
                        n_new = 1 if piece_all else 0
                        if piece_all:
                            result.per_step_counts = False
                    if piece_all:
                        chunks.append(piece_all)
                    if n_new > 0:
                        result.token_times.append(now)
                        result.chunk_tokens.append(n_new)
                        result.output_tokens += n_new
                        if result.first_token_at is None:
                            result.first_token_at = now
                    if result.output_tokens >= max_tokens and result.finish_reason:
                        break
                result.text = "".join(chunks)
        except TimeoutError:
            result.timed_out = True
            result.error = "timeout"
        except urllib.error.HTTPError as exc:
            result.status = exc.code
            result.error = exc.read().decode("utf-8", errors="replace")[:800]
        except (urllib.error.URLError, OSError) as exc:
            result.error = repr(exc)
        result.t_end = time.time()
        if result.completion_tokens and result.output_tokens == 0:
            result.output_tokens = int(result.completion_tokens)
        return result

    def tokenize(self, text: str, url: str | None = None, add_special_tokens: bool = False) -> list[int] | None:
        """vLLM /tokenize. Returns token ids or None if unavailable."""
        target = url or default_origin(self.base_url.rstrip("/")) + "/tokenize"
        status, body = self.request_json(
            "POST",
            target,
            {"model": self.model, "prompt": text, "add_special_tokens": add_special_tokens},
            timeout=120.0,
        )
        if status != 200:
            return None
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return None
        toks = data.get("tokens")
        return [int(t) for t in toks] if isinstance(toks, list) else None

    def prompt_logprobs(self, prompt_ids: list[int], timeout: float | None = None) -> list[float | None] | None:
        """Per-position logprob of each prompt token (None for position 0).

        Uses vLLM's ``prompt_logprobs`` extension on /v1/completions with
        max_tokens=1. Falls back to OpenAI ``echo`` + ``logprobs`` parsing.
        vLLM V1 does not read the prefix cache for prompt_logprobs requests,
        so the whole prompt is re-prefilled.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt_ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": 0,
        }
        status, body = self.request_json(
            "POST", self._url("completions"), payload, timeout=timeout or self.timeout_s
        )
        if status != 200:
            payload.pop("prompt_logprobs")
            payload.update({"echo": True, "logprobs": 0})
            status, body = self.request_json(
                "POST", self._url("completions"), payload, timeout=timeout or self.timeout_s
            )
            if status != 200:
                return None
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return None
        return parse_prompt_logprobs(data, prompt_ids)

    def complete_greedy(self, messages: list[dict[str, str]], max_tokens: int) -> StreamResult:
        return self.chat_stream(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            extra={"return_token_ids": True, "skip_special_tokens": False},
        )


def parse_prompt_logprobs(data: dict[str, Any], prompt_ids: list[int]) -> list[float | None] | None:
    choices = data.get("choices") or []
    if not choices:
        return None
    ch = choices[0]
    plp = ch.get("prompt_logprobs") or data.get("prompt_logprobs")
    if isinstance(plp, list):
        out: list[float | None] = []
        for pos, entry in enumerate(plp):
            if not entry:
                out.append(None)
                continue
            tok = prompt_ids[pos] if pos < len(prompt_ids) else None
            val = entry.get(str(tok)) if tok is not None else None
            if val is None and tok is not None:
                val = entry.get(tok)  # type: ignore[call-overload]
            if isinstance(val, dict):
                out.append(float(val.get("logprob")))
            elif isinstance(val, (int, float)):
                out.append(float(val))
            else:
                out.append(None)
        return out
    lp = ch.get("logprobs") or {}
    tl = lp.get("token_logprobs")
    if isinstance(tl, list):
        return [None if v is None else float(v) for v in tl[: len(prompt_ids)]]
    return None


def default_origin(url: str) -> str:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return url
    return f"{parts.scheme}://{parts.netloc}"
