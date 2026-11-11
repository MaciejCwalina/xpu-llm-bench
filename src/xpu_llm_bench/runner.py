"""Ladder runner: boot once, grow sessions, measure fixed windows, resume after abort.

Safety, in order, before any request:
1. Endpoint ports: refuse the protected production port(s) (spec
   ``safety.forbidden_ports`` plus the untracked site config).
2. Container: refuse protected production containers and services (built-in
   neutral names, spec, and site config) and CI runners; refuse a container
   that publishes a protected port.
3. bench-hold: ``status`` first; refuse if anyone else holds it (acquire
   overwrites silently); refuse an active operator quiet hold; acquire; confirm.
4. MemAvailable above the floor (default 2 GiB); sampled every 0.5 s.

The runner never starts, stops, or restarts any container.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xpu_llm_bench import bootlog, hostinfo, procmem
from xpu_llm_bench.client import OpenAICompatClient, StreamResult
from xpu_llm_bench.container import ContainerAdapter
from xpu_llm_bench.fingerprint import Fingerprint, collect_fingerprint
from xpu_llm_bench.gpu_mem import collect_gpu_mem
from xpu_llm_bench.hold import BenchHold
from xpu_llm_bench.hostexec import HostExec
from xpu_llm_bench.metrics import (
    ChunkStats,
    histogram_mean_ms,
    kv_usage_pct,
    metric_value,
    mtp_window,
    parse_prometheus,
    per_pos_sum,
    percentile,
    spec_counter_check,
    stream_stats,
    system_window,
    true_hit_rate,
)
from xpu_llm_bench.path_classify import classify_logs
from xpu_llm_bench.planner import make_plan
from xpu_llm_bench.prompts import corpus, load_corpus_dir, needle_for, session_seed
from xpu_llm_bench.quality import (
    code_task_for,
    cut_at_eos,
    extract_code,
    first_divergence,
    format_check,
    load_keyed_jsonl,
    logprob_probes,
    mean_answer_logprob,
    needle_pass,
    run_code_test,
)
from xpu_llm_bench.ram import LocalMemInfo, RamFloorError, RamSource, RamWatcher, SshMemInfo
from xpu_llm_bench.results import (
    AttentionSnapshot,
    DecodeMethod,
    GraphSnapshot,
    HardwareSnapshot,
    QualityResult,
    RungResult,
    RunManifest,
    SpecDecodeSnapshot,
    stamp_pair,
    write_jsonl_row,
)
from xpu_llm_bench.safety import (
    SafetyError,
    assert_container_allowed,
    assert_endpoint_allowed,
    assert_test_container_name,
    is_protected_container,
    site_forbidden_ports,
)
from xpu_llm_bench.schedule import Rung, build_schedule
from xpu_llm_bench.spec import RunSpec, apply_variant

ABORTING_EXITS = {"RAM_FLOOR", "OOM", "SPEC_ASSERT", "ENGINE_ERROR"}


# ---------------------------------------------------------------------------
# Sessions: one growing conversation each, persisted for --resume
# ---------------------------------------------------------------------------


class Session:
    """A conversation that only grows.

    Growth appends a new user turn (fresh corpus text + one planted needle +
    the ask). After each measured window the model's own reply (cut at EOS)
    is appended as the assistant turn. Earlier turns are never edited, so
    the rendered prefix is stable and the prefix cache is reused.
    """

    def __init__(
        self,
        sid: int,
        prompt_type: str,
        seed: int,
        chars_per_token: float,
        style: str = "realistic",
        file_text: str | None = None,
        plant_needles: bool = True,
    ):
        self.sid = sid
        self.prompt_type = prompt_type
        self.seed = seed
        self.cpt = chars_per_token
        self.plant_needles = plant_needles
        self.corpus = corpus(prompt_type, seed=seed, chars_per_token=chars_per_token, style=style, file_text=file_text)  # type: ignore[arg-type]
        self.messages: list[dict[str, str]] = []
        self.segments = 0
        self.needles: list[dict[str, Any]] = []
        self.ctx_tokens = 0  # last server-reported prompt_tokens
        # back-compat with the scaffold API
        self.prefix = ""

    # -- size bookkeeping ---------------------------------------------------
    def chars(self) -> int:
        return sum(len(m["content"]) for m in self.messages)

    def est_tokens(self) -> int:
        if self.ctx_tokens:
            return self.ctx_tokens
        return int(self.chars() / max(self.cpt, 0.5))

    def calibrate(self, prompt_tokens: int | None) -> None:
        if prompt_tokens:
            self.ctx_tokens = int(prompt_tokens)
            self.cpt = max(self.chars() / max(prompt_tokens, 1), 0.5)

    # -- growth -------------------------------------------------------------
    def grow_to(self, target_tokens: int) -> list[dict[str, str]]:
        need = target_tokens - self.est_tokens() - 64  # template overhead per turn
        if need <= 0 and self.messages and self.messages[-1]["role"] == "user":
            return self.messages
        n_chars = max(int(max(need, 32) * self.cpt), 64)
        needle_text = None
        if self.plant_needles:
            fact, question, answer = needle_for(self.seed, self.segments)
            needle_text = fact
            self.needles.append({"segment": self.segments, "question": question, "answer": answer})
        body = self.corpus.segment(n_chars, self.segments, needle=needle_text)
        opener = self.corpus.opener() if not self.messages else ""
        self.messages.append({"role": "user", "content": opener + body + self.corpus.ask()})
        self.segments += 1
        return self.messages

    def grow_message(self, target_tokens: int) -> list[dict[str, str]]:  # scaffold name
        return self.grow_to(target_tokens)

    def append_reply(self, text: str) -> None:
        reply, _ = cut_at_eos(text)
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages.append({"role": "assistant", "content": reply.strip() or "(no reply)"})

    def to_dict(self) -> dict[str, Any]:
        return {
            "sid": self.sid,
            "prompt_type": self.prompt_type,
            "seed": self.seed,
            "cpt": self.cpt,
            "messages": self.messages,
            "segments": self.segments,
            "needles": self.needles,
            "ctx_tokens": self.ctx_tokens,
        }

    def load_state(self, d: dict[str, Any]) -> None:
        self.cpt = float(d.get("cpt") or self.cpt)
        self.messages = list(d.get("messages") or [])
        self.segments = int(d.get("segments") or 0)
        self.needles = list(d.get("needles") or [])
        self.ctx_tokens = int(d.get("ctx_tokens") or 0)


# ---------------------------------------------------------------------------
# Window sampler: /metrics and /proc during a measured window
# ---------------------------------------------------------------------------


@dataclass
class WindowSamples:
    running: list[float] = field(default_factory=list)
    waiting: list[float] = field(default_factory=list)
    kv_pct: list[float] = field(default_factory=list)
    rss: dict[str, int] = field(default_factory=dict)
    other_stacks: set[str] = field(default_factory=set)


class WindowSampler:
    def __init__(self, fetch: Callable[[], str], names, interval_s: float, proc_fn: Callable[[], procmem.ProcSnapshot] | None):
        self.fetch = fetch
        self.names = names
        self.interval_s = max(interval_s, 0.05)
        self.proc_fn = proc_fn
        self.samples = WindowSamples()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _proc(self) -> None:
        if not self.proc_fn:
            return
        try:
            snap = self.proc_fn()
        except Exception:  # noqa: BLE001
            return
        self.samples.rss = procmem.peak_merge(self.samples.rss, snap.ours)
        self.samples.other_stacks.update(snap.other_stacks)

    def _loop(self) -> None:
        n = 0
        while not self._stop.is_set():
            try:
                snap = parse_prometheus(self.fetch())
                for key, dst in ((self.names.running, self.samples.running), (self.names.waiting, self.samples.waiting)):
                    v = metric_value(snap, key)
                    if v is not None:
                        dst.append(v)
                kv = kv_usage_pct(snap, self.names)
                if kv is not None:
                    self.samples.kv_pct.append(kv)
            except Exception:  # noqa: BLE001
                pass
            if n % 5 == 0:
                self._proc()
            n += 1
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._proc()
        self._thread = threading.Thread(target=self._loop, name="window-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> WindowSamples:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._proc()
        return self.samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_id(spec_name: str) -> str:
    utc, _ = stamp_pair()
    short = uuid.uuid4().hex[:8]
    stamp = utc.replace(":", "").replace("-", "").split(".")[0]
    return f"{spec_name}-{stamp}-{short}"


def _mean(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def _sum(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) if vals else None


def _host_exec(spec: RunSpec) -> HostExec:
    return HostExec(spec.access.mode, spec.access.ssh_host, spec.access.ssh_extra_args)


def _ram_source(spec: RunSpec) -> RamSource:
    if spec.access.mode == "ssh":
        if not spec.access.ssh_host:
            raise SafetyError("access.mode is ssh but ssh_host is not set (TODO)")
        return SshMemInfo(spec.access.ssh_host, spec.access.ssh_extra_args)
    return LocalMemInfo()


def safety_preflight(spec: RunSpec) -> None:
    """Checks that need no network: ports and container names. Runs in dry-run too."""
    ep = spec.endpoint
    assert_endpoint_allowed(
        [ep.base_url, ep.metrics_url, ep.health_url, ep.tokenize_url],
        site_forbidden_ports(spec.safety.forbidden_ports),
    )
    name = spec.server.container_name
    if name:
        assert_container_allowed(name)
        if is_protected_container(name, spec.safety.protected_containers):
            raise SafetyError(f"Refusing protected container {name!r}")
        if spec.server.manage_container:
            assert_test_container_name(name, spec.safety.test_container_prefixes)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class Runner:
    def __init__(
        self,
        spec: RunSpec,
        output_dir: str | Path,
        dry_run: bool = False,
        resume: bool = False,
        client: OpenAICompatClient | None = None,
        hold: BenchHold | None = None,
        containers: ContainerAdapter | None = None,
        ram_source: RamSource | None = None,
        sleep: Callable[[float], None] = time.sleep,
        fetch_metrics: Callable[[], str] | None = None,
        fetch_logs: Callable[[], str] | None = None,
        collect_mem: Callable[..., str | None] | None = None,
        host_exec: HostExec | None = None,
        fingerprint_fn: Callable[[], Fingerprint | None] | None = None,
        proc_fn: Callable[[], procmem.ProcSnapshot] | None = None,
        host_fn: Callable[[], dict] | None = None,
        ram_wait_min: float = 0.0,
    ):
        self.spec = spec
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.resume = resume
        self.client = client or OpenAICompatClient(
            spec.endpoint.base_url,
            spec.endpoint.model,
            api_key_env=spec.endpoint.api_key_env,
            timeout_s=spec.endpoint.timeout_s,
        )
        self.exec = host_exec or _host_exec(spec)
        if hold is None:
            cmd = spec.safety.hold_command
            if spec.access.mode == "ssh" and cmd.startswith("~/"):
                cmd = cmd[2:]  # ssh runs in $HOME; avoid expanding ~ on this machine
            hold = BenchHold(
                cmd,
                name=spec.safety.hold_name,
                dry_run=dry_run,
                runner=(lambda c, **kw: __import__("subprocess").run(self.exec.argv(c), **kw))
                if spec.access.mode == "ssh"
                else None,
            )
        self.hold = hold
        self.containers = containers or ContainerAdapter(allowed=spec.safety.allowed_containers, dry_run=dry_run)
        self.ram_source = ram_source or _ram_source(spec)
        self.sleep = sleep
        self.fetch_metrics = fetch_metrics
        self.fetch_logs = fetch_logs
        self.collect_mem = collect_mem or collect_gpu_mem
        self._fingerprint_fn = fingerprint_fn
        self._proc_fn = proc_fn
        self._host_fn = host_fn
        self.ram_wait_min = ram_wait_min
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.k_declared = spec.server.speculative.k
        self.fp: Fingerprint | None = None
        self.boot: bootlog.BootEvidence | None = None
        self.greedy_ref = load_keyed_jsonl(spec.quality.greedy_reference)
        self.logprob_base = load_keyed_jsonl(spec.quality.logprob_baseline)
        self.diverged: dict[tuple[str, str, int], bool] = {}
        self.corpus_text: dict[str, str | None] = {}
        self.corpus_manifest: dict[str, list[dict]] = {}

    # -- I/O ------------------------------------------------------------------
    def _metrics_text(self) -> str:
        if self.fetch_metrics:
            return self.fetch_metrics()
        status, body = self.client.get_text(self.spec.endpoint.metrics_url)
        if status != 200:
            raise RuntimeError(f"/metrics returned {status}")
        return body

    def _logs_text(self, tail: int = 4000) -> str:
        if self.fetch_logs:
            return self.fetch_logs()
        name = self.spec.server.container_name
        if not name:
            return ""
        assert_container_allowed(name)
        res = self.exec.run(["docker", "logs", "--timestamps", "--tail", str(tail), name], timeout=60)
        return res.stdout + res.stderr

    def _fingerprint(self) -> Fingerprint | None:
        if self._fingerprint_fn:
            return self._fingerprint_fn()
        name = self.spec.server.container_name
        if not name:
            return None
        try:
            return collect_fingerprint(self.exec, name)
        except Exception:  # noqa: BLE001
            return None

    def _procs(self) -> procmem.ProcSnapshot:
        if self._proc_fn:
            return self._proc_fn()
        cid = self.fp.container_id if self.fp else None
        return procmem.collect(self.exec, cid)

    def _host(self) -> dict:
        if self._host_fn:
            return self._host_fn()
        try:
            return hostinfo.collect(self.exec)
        except Exception as exc:  # noqa: BLE001
            return {"error": repr(exc)}

    def _write_manifest(self, manifest: RunManifest) -> None:
        manifest.save(self.output_dir / "manifest.json")

    def _save_sessions(self, sessions: dict[tuple[str, str, int], Session]) -> None:
        data = {"|".join(map(str, k)): s.to_dict() for k, s in sessions.items()}
        tmp = self.output_dir / "sessions.json.tmp"
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.output_dir / "sessions.json")

    def _load_sessions(self) -> dict[str, dict]:
        p = self.output_dir / "sessions.json"
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    def _append_jsonl(self, name: str, obj: dict) -> None:
        with (self.output_dir / name).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj) + "\n")

    # -- manifest -------------------------------------------------------------
    def new_manifest(self, run_id: str, rungs: list[Rung]) -> RunManifest:
        utc, et = stamp_pair()
        names = self.spec.metrics
        method = DecodeMethod(
            window="fixed_output_tokens",
            output_tokens=self.spec.ladder.output_tokens_per_rung,
            counters=[
                names.drafts,
                names.draft_tokens,
                names.accepted_tokens,
                names.accepted_per_pos,
                names.preemptions,
                names.prefix_hits,
                names.prefix_queries,
                names.generation_tokens,
                names.gpu_cache_usage,
                names.running,
                names.waiting,
                "vllm:request_queue_time_seconds",
            ],
        )
        return RunManifest(
            run_id=run_id,
            spec_name=self.spec.name,
            profile=self.spec.profile,
            started_utc=utc,
            started_et=et,
            image=self.spec.server.image,
            image_digest=self.spec.server.image_digest,
            server_source_commit=self.spec.server.source_commit,
            hardware=HardwareSnapshot(
                host=self.spec.hardware.host,
                gpu_model=self.spec.hardware.gpu_model,
                gpu_count=self.spec.hardware.gpu_count,
                tensor_parallel=self.spec.hardware.tensor_parallel,
                host_ram_gib=self.spec.hardware.host_ram_gib,
                mem_available_floor_gib=self.spec.hardware.mem_available_floor_gib,
            ),
            config={
                "env": self.spec.server.env,
                "launch_args": self.spec.server.launch_args,
                "attention_requested": self.spec.server.attention_requested,
                "max_num_seqs": self.spec.server.max_num_seqs,
                "max_num_batched_tokens": self.spec.server.max_num_batched_tokens,
                "gpu_memory_utilization": self.spec.server.gpu_memory_utilization,
                "boot_once": self.spec.ladder.boot_once,
                "grow_monotonically": self.spec.ladder.grow_monotonically,
                "shape": self.spec.ladder.shape,
                "temperature": self.spec.ladder.temperature,
                "ignore_eos": self.spec.ladder.ignore_eos,
                "prompt_style": self.spec.prompts.style,
            },
            attention=AttentionSnapshot(requested=self.spec.server.attention_requested),
            kv_format=self.spec.server.kv_format,
            graph=GraphSnapshot(
                mode=self.spec.server.graph.mode,
                per_gpu=self.spec.server.graph.per_gpu,
                capture_sizes=list(self.spec.server.graph.capture_sizes),
                graphs_on=self.spec.server.graph.mode != "EAGER",
                eager_fallback=False,
            ),
            speculative=SpecDecodeSnapshot(
                method=self.spec.server.speculative.method,
                num_speculative_tokens=self.spec.server.speculative.num_speculative_tokens,
                dflash_k=self.spec.server.speculative.dflash_k,
            ),
            decode_method=method,
            output_jsonl=str(self.output_dir / "results.jsonl"),
            notes=list(self.spec.notes),
            prompt_seed=self.spec.prompts.seed,
            k=self.k_declared,
        )

    # -- per-rung pieces ------------------------------------------------------
    def _stream(self, messages: list[dict[str, str]], max_tokens: int, measured: bool) -> StreamResult:
        extra: dict[str, Any] = {"skip_special_tokens": False, "return_token_ids": True}
        if measured and self.spec.ladder.ignore_eos:
            extra["ignore_eos"] = True
        return self.client.chat_stream(
            messages, max_tokens=max_tokens, temperature=self.spec.ladder.temperature, extra=extra
        )

    def _grow(self, rung: Rung, sessions: dict[int, Session], watcher: RamWatcher, metrics_ok: bool) -> tuple[list, list[str]]:
        """Grow sessions one at a time (joiners join one at a time). Returns (prefill stats, errors)."""
        stats = []
        errors: list[str] = []
        for sid in rung.grow_sessions:
            watcher.raise_if_tripped()
            sess = sessions[sid]
            messages = sess.grow_to(rung.target_for(sid))
            before = parse_prometheus(self._metrics_text()) if metrics_ok else {}
            res = self._stream(messages, max_tokens=max(self.spec.ladder.warmup.tokens, 1), measured=False)
            after = parse_prometheus(self._metrics_text()) if metrics_ok else {}
            if res.error or res.timed_out:
                errors.append(f"grow s{sid}: {res.error or 'timeout'}")
                break
            sess.calibrate(res.prompt_tokens)
            new_tokens = None
            if res.prompt_tokens is not None and res.cached_tokens is not None:
                new_tokens = max(int(res.prompt_tokens) - int(res.cached_tokens), 0)
            elif before and after:
                q = metric_value(after, self.spec.metrics.prefix_queries)
                h = metric_value(after, self.spec.metrics.prefix_hits)
                q0 = metric_value(before, self.spec.metrics.prefix_queries)
                h0 = metric_value(before, self.spec.metrics.prefix_hits)
                if None not in (q, h, q0, h0):
                    new_tokens = int((q - q0) - (h - h0))  # type: ignore[operator]
            stats.append(
                stream_stats(
                    res.t_send,
                    res.first_token_at,
                    res.t_end,
                    res.token_times,
                    res.output_tokens,
                    prompt_tokens=res.prompt_tokens,
                    cached_tokens=res.cached_tokens,
                    new_prompt_tokens=new_tokens,
                )
            )
        return stats, errors

    def _quality(
        self,
        rung: Rung,
        sessions: dict[int, Session],
        results: dict[int, StreamResult],
        run_key: str,
    ) -> QualityResult:
        q = QualityResult()
        qs = self.spec.quality
        if not qs.enabled:
            return q
        # 3a) format / repetition on the measured outputs (writing + chat)
        if qs.format_checks and rung.prompt_type in ("long_form", "multi_turn_chat"):
            problems: list[str] = []
            reps: list[float] = []
            for sid, res in results.items():
                fc = format_check(res.text, rung.prompt_type, qs.repetition_ngram, qs.repetition_max_ratio)
                reps.append(fc.repetition)
                problems += [f"s{sid}: {p}" for p in fc.problems]
            q.format_ok = not problems
            q.format_problems = problems[:10]
            q.repetition_max = max(reps) if reps else None
        # 2) greedy divergence vs stored reference (MTP-off / dense boot)
        if self.spec.ladder.temperature == 0.0:
            statuses = []
            for sid, res in results.items():
                probe = f"{rung.rung_id}:s{sid}"
                self._append_jsonl("greedy_outputs.jsonl", {"probe_id": probe, "token_ids": res.token_ids, "text": res.text})
                dkey = (rung.family, rung.load_mode, sid)
                ref = self.greedy_ref.get(probe)
                if self.diverged.get(dkey):
                    statuses.append("context_diverged")
                    continue
                if not ref:
                    statuses.append("no_reference")
                    continue
                a = res.token_ids or list(res.text)
                b = ref.get("token_ids") or list(ref.get("text") or "")
                idx = first_divergence(a, b)
                if idx is None:
                    statuses.append("match")
                else:
                    statuses.append("diverged")
                    self.diverged[dkey] = True
                    if q.greedy_first_divergence is None:
                        q.greedy_first_divergence = idx
            if statuses:
                order = ["diverged", "context_diverged", "no_reference", "match"]
                q.greedy_ref = next(s for s in order if s in statuses)
        s1 = sessions.get(rung.decoding_sessions[0])
        # 3b) needle at this context (deepest + newest planted fact)
        if qs.needle and s1 and s1.needles:
            picks = [s1.needles[0]] + ([s1.needles[-1]] if len(s1.needles) > 1 else [])
            outcomes = []
            for nd in picks:
                msgs = s1.messages + [{"role": "user", "content": nd["question"]}]
                r = self.client.chat_stream(msgs, max_tokens=24, temperature=0.0, extra={"skip_special_tokens": False})
                ok = (not r.error) and needle_pass(r.text, nd["answer"])
                outcomes.append(ok)
                if not ok and q.needle_detail is None:
                    q.needle_detail = f"segment {nd['segment']}: expected {nd['answer']!r}, got {cut_at_eos(r.text)[0][:60]!r}"
            q.needle_pass = all(outcomes)
        # 3c) code edit checked by a unit test, on the live code-edit context
        if qs.code_edit and s1 and rung.prompt_type == "code_edits" and rung.phase in ("S", "meas"):
            task = code_task_for(s1.seed + s1.segments)
            msgs = s1.messages + [{"role": "user", "content": task["prompt"]}]
            r = self.client.chat_stream(msgs, max_tokens=256, temperature=0.0, extra={"skip_special_tokens": False})
            code = extract_code(r.text) if not r.error else None
            if code is None:
                q.code_edit_pass, q.code_edit_detail = False, f"{task['id']}: no code in reply"
            else:
                ok, detail = run_code_test(code, task["test"], qs.code_edit_timeout_s)
                q.code_edit_pass, q.code_edit_detail = ok, f"{task['id']}: {detail}"
        # 1) logprob scoring vs dense baseline
        if qs.logprob_mode != "off":
            scores: dict[str, float] = {}
            probes = logprob_probes(self.spec.prompts.seed, int(qs.probe_ctx_tokens * (s1.cpt if s1 else 4.0)))
            if qs.logprob_mode == "ladder_ctx" and s1:
                ctx_text = "\n\n".join(m["content"] for m in s1.messages)
                probes = [type(p)(f"{p.probe_id}@{rung.rung_id}", ctx_text + "\n\n" + p.context, p.answer) for p in probes]
            for p in probes:
                ctx_ids = self.client.tokenize(p.context, url=self.spec.endpoint.tokenize_url, add_special_tokens=True)
                ans_ids = self.client.tokenize(p.answer, url=self.spec.endpoint.tokenize_url, add_special_tokens=False)
                if not ctx_ids or not ans_ids:
                    continue
                lps = self.client.prompt_logprobs(ctx_ids + ans_ids)
                if not lps:
                    continue
                m = mean_answer_logprob(lps, len(ctx_ids))
                if m is not None:
                    scores[p.probe_id] = m
                    self._append_jsonl("logprob_scores.jsonl", {"probe_id": p.probe_id, "mean_logprob": m, "rung_id": rung.rung_id, "run": run_key})
            q.logprob_scores = scores
            deltas = [scores[k] - float(self.logprob_base[k]["mean_logprob"]) for k in scores if k in self.logprob_base]
            if deltas:
                q.logprob_worst_delta = min(deltas)
                q.logprob_ok = q.logprob_worst_delta >= -qs.logprob_tolerance_nats
        return q

    def _label_flags(self, attention_runtime: str, n_decoding: int, k: int | None) -> tuple[list[str], int | None, list[int]]:
        flags: list[str] = []
        boot = self.boot
        req = self.spec.server.attention_requested
        sizes = list((boot.capture_sizes_logged if boot and boot.capture_sizes_logged else self.spec.server.graph.capture_sizes) or [])
        batch = n_decoding * (k + 1) if k else None
        if boot:
            if req == "qsa" and boot.attention_boot == "dense":
                flags.append("NOT_AS_LABELED")
            if req == "dense" and boot.attention_boot == "qsa":
                flags.append("NOT_AS_LABELED")
            if self.k_declared and boot.spec_k and boot.spec_k != self.k_declared:
                flags.append("NOT_AS_LABELED")
            if boot.kv_dtype and self.spec.server.kv_format and boot.kv_dtype != self.spec.server.kv_format:
                flags.append("NOT_AS_LABELED")
            if self.spec.server.graph.mode != "EAGER" and boot.graphs_on is False:
                flags.append("NOT_AS_LABELED")
            if boot.graphs_on is False:
                flags.append("GRAPH_MISS")
        if batch is not None and sizes and batch > max(sizes):
            flags.append("GRAPH_MISS")
        if attention_runtime == "fallback":
            flags.append("QSA_FALLBACK")
        if attention_runtime == "mixed":
            flags.append("MIXED_PATH")
        return sorted(set(flags)), batch, sizes

    def _measure_rung(
        self,
        run_id: str,
        rung: Rung,
        sessions: dict[int, Session],
        watcher: RamWatcher,
        extra_flags: list[str] | None = None,
    ) -> RungResult:
        utc0, et0 = stamp_pair()
        seeds = [sessions[s].seed for s in rung.decoding_sessions]
        base = dict(
            run_id=run_id,
            rung_id=rung.rung_id,
            users=rung.users,
            ctx_per_user=rung.ctx_per_user,
            prompt_type=rung.prompt_type,
            load_mode=rung.load_mode,
            phase=rung.phase,
            prompt_seed=self.spec.prompts.seed,
            session_seeds=seeds,
            temperature=self.spec.ladder.temperature,
            k=self.k_declared,
            kv_format=self.spec.server.kv_format,
            build_fingerprint=self.fp.build_fingerprint if self.fp else None,
            launch_fingerprint=self.fp.launch_fingerprint if self.fp else None,
            image_id=self.fp.image_id if self.fp else None,
            boot_started_at=self.fp.started_at if self.fp else None,
            attention_boot=self.boot.attention_boot if self.boot else None,
            started_utc=utc0,
            started_et=et0,
        )
        if self.dry_run:
            for sid in rung.grow_sessions:
                sessions[sid].grow_to(rung.target_for(sid))
                sessions[sid].ctx_tokens = rung.target_for(sid)
            utc1, et1 = stamp_pair()
            return RungResult(
                **base,
                ctx_tokens=[rung.target_for(s) for s in rung.decoding_sessions],
                valid="valid",
                reason="dry-run",
                flags=["DRY_RUN"],
                finished_utc=utc1,
                finished_et=et1,
            )

        # Prefill / growth (joiners one at a time). Outside the measured window.
        grow_stats, grow_errors = self._grow(rung, sessions, watcher, metrics_ok=True)
        watcher.raise_if_tripped()
        if grow_errors:
            return self._failed_row(base, rung, sessions, grow_errors, timed_out=False, since=time.time() - 600)

        # Measured window.
        before = parse_prometheus(self._metrics_text())
        t_window = time.time()
        sampler = WindowSampler(self._metrics_text, self.spec.metrics, self.spec.ladder.metrics_poll_s, self._procs)
        sampler.start()
        delay = self.spec.ladder.staggered.delay_s if rung.load_mode == "staggered" else 0.0
        collected: dict[int, StreamResult] = {}

        def _one(sid: int, start_delay: float) -> None:
            if start_delay:
                self.sleep(start_delay)
            collected[sid] = self._stream(sessions[sid].messages, rung.output_tokens, measured=True)

        threads = []
        for i, sid in enumerate(rung.decoding_sessions):
            start_delay = i * delay
            if rung.load_mode == "open_loop":
                start_delay = i / max(self.spec.ladder.open_loop.requests_per_second, 1e-6)
            th = threading.Thread(target=_one, args=(sid, start_delay), daemon=True)
            threads.append(th)
            th.start()
        for th in threads:
            th.join()
        samples = sampler.stop()
        after = parse_prometheus(self._metrics_text())
        log_lines = bootlog.lines_since(self._logs_text(), t_window - 1.0)

        errors: list[str] = []
        timeouts = 0
        for sid in rung.decoding_sessions:
            r = collected.get(sid)
            if r is None:
                errors.append(f"session {sid}: no result")
            elif r.error:
                errors.append(f"session {sid}: {r.error}")
            if r is not None and r.timed_out:
                timeouts += 1
        watcher.raise_if_tripped()
        if errors:
            return self._failed_row(base, rung, sessions, errors, timed_out=timeouts > 0, since=t_window, log_lines=log_lines)

        # Per-session stream accounting.
        mtp = mtp_window(before, after, self.spec.metrics)
        sysm = system_window(before, after, self.spec.metrics)
        flags: list[str] = list(extra_flags or [])
        k = self.k_declared
        if k is None and mtp.drafts and mtp.drafts > 0 and mtp.draft_tokens:
            k = round(mtp.draft_tokens / mtp.drafts)
            flags.append("K_UNDECLARED")
        chunk_stats: list[ChunkStats] = []
        sstats = []
        ctxs: list[int] = []
        per_step_ok = True
        eos_seen = False
        for sid in rung.decoding_sessions:
            r = collected[sid]
            per_step_ok = per_step_ok and r.per_step_counts
            cs = ChunkStats(tokens=r.output_tokens, chunks=r.chunks, t_first=r.first_token_at, t_last=r.last_token_at)
            chunk_stats.append(cs)
            sstats.append(
                stream_stats(r.t_send, r.first_token_at, r.t_end, r.token_times, r.output_tokens,
                             prompt_tokens=r.prompt_tokens, cached_tokens=r.cached_tokens)
            )
            if cut_at_eos(r.text)[1]:
                eos_seen = True
            sessions[sid].calibrate(r.prompt_tokens)
            ctxs.append(sessions[sid].ctx_tokens or rung.target_for(sid))

        E = sum(c.decode_tokens for c in chunk_stats)
        S_stream = sum(c.verify_steps for c in chunk_stats)
        S_server = int(mtp.drafts) if mtp.drafts is not None else None
        if per_step_ok and S_stream > 0:
            S = S_stream
            steps_user_each = [c.steps_s for c in chunk_stats]
        else:
            S = S_server
            flags.append("DERIVED_STEPS")
            share = (S_server or 0) / max(len(chunk_stats), 1)
            steps_user_each = [(share / c.window_s) if c.window_s else None for c in chunk_stats]
        if S_server and S_stream and per_step_ok and abs(S_stream - S_server) / S_server > 0.05:
            flags.append("STEP_COUNT_MISMATCH")
        tok_user_each = [c.decode_tok_s for c in chunk_stats]
        hit = true_hit_rate(E, S, k) if k else None
        check = spec_counter_check(
            mtp, k, E, S_stream, len(chunk_stats),
            per_pos_sum(before, after, self.spec.metrics.accepted_per_pos),
        )
        if k and mtp.drafts and not check.ok:
            flags.append("SPEC_COUNTER_BROKEN")
        if any(c.verify_steps < self.spec.ladder.min_window_steps for c in chunk_stats):
            flags.append("SHORT_WINDOW")
        if eos_seen:
            flags.append("EOS_IN_WINDOW")
        if sysm.preemptions:
            flags.append("PREEMPTED")
        running_max = max(samples.running) if samples.running else None
        waiting_max = max(samples.waiting) if samples.waiting else None
        r0 = metric_value(before, self.spec.metrics.running) or 0.0
        if r0 > 0 or (running_max or 0) + (waiting_max or 0) > len(rung.decoding_sessions):
            flags.append("CONCURRENT_CLIENT")
        if samples.other_stacks:
            flags.append("FOREIGN_STACK")

        # Attention path in this window (runtime lines) on top of boot evidence.
        runtime = classify_logs(log_lines, patterns=self.spec.log_patterns, requested=self.spec.server.attention_requested)
        has_runtime = bool(runtime.active_widths or runtime.skip_widths)
        attention = runtime.path if has_runtime else (self.boot.attention_boot if self.boot and self.boot.attention_boot != "unknown" else runtime.path)
        lf, batch, sizes = self._label_flags(runtime.path if has_runtime else "", len(rung.decoding_sessions), k)
        flags += lf
        # Spec assert can be logged without failing the request.
        fail = bootlog.classify_failure(log_lines, [], patterns=self.spec.log_patterns)
        exit_reason = None
        if fail and fail.flag == "SPEC_ASSERT":
            flags.append("SPEC_ASSERT")
            exit_reason = "SPEC_ASSERT"

        # Rebooted mid-run?
        fp_now = self._fingerprint()
        if self.fp and fp_now and fp_now.started_at and self.fp.started_at and fp_now.started_at != self.fp.started_at:
            flags.append("REBOOTED")

        # Assistant replies are appended (conversation grows; never edited).
        for sid in rung.decoding_sessions:
            sessions[sid].append_reply(collected[sid].text)

        quality = self._quality(rung, sessions, {s: collected[s] for s in rung.decoding_sessions}, run_id)
        if quality.needle_pass is False or quality.code_edit_pass is False or quality.format_ok is False or quality.logprob_ok is False or quality.greedy_ref == "diverged":
            flags.append("QUALITY_FAIL")

        prefill_new = [s.prefill_tok_s for s in grow_stats]
        ttft_grow = [s.ttft_ms for s in grow_stats if s.ttft_ms]
        prefill_tokens = None
        prefill_rate = None
        if grow_stats:
            toks = [
                (s.prompt_tokens or 0) - (s.cached_tokens or 0) if s.cached_tokens is not None else None
                for s in grow_stats
            ]
            if all(t is not None for t in toks) and ttft_grow:
                prefill_tokens = int(sum(toks))  # type: ignore[arg-type]
                prefill_rate = prefill_tokens / (sum(ttft_grow) / 1000.0) if sum(ttft_grow) > 0 else None
            else:
                prefill_rate = _mean(prefill_new)
        itls = []
        for sid in rung.decoding_sessions:
            tt = collected[sid].token_times
            itls += [(b - a) * 1000.0 for a, b in zip(tt, tt[1:], strict=False)]
        windows = [c.window_s for c in chunk_stats if c.window_s]
        tpots = [((s.e2e_ms or 0) - (s.ttft_ms or 0)) / max(c.tokens - 1, 1) for s, c in zip(sstats, chunk_stats, strict=False) if s.ttft_ms is not None and c.tokens > 1]
        gpu = None
        try:
            gpu = self.collect_mem(container=None, kv_cache_usage_pct=sysm.kv_cache_usage_pct)
        except SafetyError:
            gpu = None
        utc1, et1 = stamp_pair()
        valid = "valid"
        reason = None
        hard = {"NOT_AS_LABELED", "SPEC_COUNTER_BROKEN", "SPEC_ASSERT", "REBOOTED", "CONCURRENT_CLIENT", "QSA_FALLBACK", "MIXED_PATH"}
        if set(flags) & hard:
            valid = "warning"
            reason = "flags: " + " ".join(sorted(set(flags) & hard))
        if attention in ("fallback", "mixed") and valid == "valid":
            valid, reason = "warning", f"attention path={attention}"
        acc_step = (E / S - 1.0) if S else None
        return RungResult(
            **base,
            ctx_tokens=ctxs,
            capture_sizes=sizes,
            decode_batch_tokens=batch,
            prefill_tok_s=prefill_rate,
            prefill_tokens=prefill_tokens,
            ttft_ms=_mean([s.ttft_ms for s in sstats]),
            decode_tok_s_per_user=_mean(tok_user_each),
            decode_tok_s_total=_sum(tok_user_each),
            steps_s_per_user=_mean(steps_user_each),
            steps_s_total=_sum(steps_user_each),
            tpot_ms=_mean(tpots),
            decode_window_s=_mean(windows),
            accepted_drafts_per_step=acc_step,
            hit_rate=hit,
            acceptance_rate_vllm=mtp.hit_rate,
            tokens_per_step=(E / S) if S else None,
            tokens_delivered=E,
            verify_steps=S,
            verify_steps_stream=S_stream if per_step_ok else None,
            verify_steps_server=S_server,
            draft_tokens=int(mtp.draft_tokens) if mtp.draft_tokens is not None else None,
            accepted_drafts=int(mtp.accepted_drafts) if mtp.accepted_drafts is not None else None,
            drafts=S_server,
            itl_p50_ms=percentile(itls, 50),
            itl_p95_ms=percentile(itls, 95),
            itl_p99_ms=percentile(itls, 99),
            e2e_ms=_mean([s.e2e_ms for s in sstats]),
            queue_time_ms=histogram_mean_ms(before, after, "vllm:request_queue_time_seconds") or sysm.queue_time_ms,
            running_reqs=_mean(samples.running) if samples.running else sysm.running_reqs,
            waiting_reqs=_mean(samples.waiting) if samples.waiting else sysm.waiting_reqs,
            running_max=running_max,
            waiting_max=waiting_max,
            prefix_cache_hit_rate=sysm.prefix_cache_hit_rate,
            kv_cache_usage_pct=sysm.kv_cache_usage_pct,
            kv_cache_usage_max_pct=max(samples.kv_pct) if samples.kv_pct else sysm.kv_cache_usage_pct,
            preemptions=int(sysm.preemptions) if sysm.preemptions is not None else None,
            mem_available_min_gib=watcher.min_gib(),
            gpu_mem=gpu,
            rss_kib=samples.rss,
            other_vllm_stacks=sorted(samples.other_stacks),
            attention_path=attention,  # type: ignore[arg-type]
            attention_evidence=(runtime.evidence or (self.boot.evidence if self.boot else []))[:12],
            graphs_on=self.boot.graphs_on if self.boot and self.boot.graphs_on is not None else self.spec.server.graph.mode != "EAGER",
            eager_fallback=bool(runtime.skip_widths and self.spec.server.graph.eager_fallback_when_unmatched),
            exit_reason=exit_reason,
            flags=sorted(set(flags)),
            quality=quality,
            valid=valid,  # type: ignore[arg-type]
            reason=reason,
            errors=[],
            timeouts=0,
            finished_utc=utc1,
            finished_et=et1,
        )

    def _failed_row(
        self,
        base: dict,
        rung: Rung,
        sessions: dict[int, Session],
        errors: list[str],
        timed_out: bool,
        since: float,
        log_lines: list[str] | None = None,
        ram_floor: bool = False,
        mem_min: float | None = None,
    ) -> RungResult:
        if log_lines is None:
            try:
                log_lines = bootlog.lines_since(self._logs_text(), since - 1.0)
            except Exception:  # noqa: BLE001
                log_lines = []
        fp_now = None if ram_floor else self._fingerprint()
        fail = bootlog.classify_failure(
            log_lines, errors, timed_out=timed_out, ram_floor_hit=ram_floor,
            oom_killed=bool(fp_now and fp_now.oom_killed), patterns=self.spec.log_patterns,
        )
        flag = fail.flag if fail else "CLIENT_ERROR"
        utc1, et1 = stamp_pair()
        detail = (fail.detail if fail else "") + (" | " + " / ".join(fail.lines[:2]) if fail and fail.lines else "")
        return RungResult(
            **base,
            ctx_tokens=[sessions[s].ctx_tokens or rung.target_for(s) for s in rung.decoding_sessions],
            mem_available_min_gib=mem_min,
            valid="invalid",
            reason=detail[:500] or "; ".join(errors)[:500],
            errors=errors[:10],
            timeouts=1 if timed_out else 0,
            exit_reason=flag,
            flags=[flag],
            finished_utc=utc1,
            finished_et=et1,
        )

    # -- preflight ------------------------------------------------------------
    def _preflight(self, manifest: RunManifest) -> None:
        safety_preflight(self.spec)
        if self.dry_run:
            manifest.hold = {"dry_run": True, "would_run": [
                f"{self.hold.command} status",
                f"{self.hold.command} acquire {self.spec.safety.hold_minutes} {self.spec.safety.hold_reason!r} --by {self.hold.name}",
                f"{self.hold.command} status",
                "...",
                f"{self.hold.command} release  (only if status still shows by={self.hold.name})",
            ]}
            return
        self.fp = self._fingerprint()
        if self.fp:
            bad = set(self.fp.host_ports) & site_forbidden_ports(self.spec.safety.forbidden_ports)
            if bad:
                raise SafetyError(f"Refusing: container {self.fp.container} is bound to production port(s) {sorted(bad)}")
            manifest.fingerprint = self.fp.to_dict()
        try:
            self.boot = bootlog.parse_boot(self._logs_text(tail=200000), self.spec.log_patterns)
        except Exception:  # noqa: BLE001
            self.boot = None
        if self.boot:
            manifest.boot = self.boot.summary()
            manifest.attention.actual = self.boot.attention_boot  # type: ignore[assignment]
            manifest.attention.evidence_lines = self.boot.evidence[:20]
        manifest.host = self._host()

    # -- main loop -------------------------------------------------------------
    def run(self) -> RunManifest:
        rungs = build_schedule(self.spec)
        completed: set[str] = set()
        manifest: RunManifest | None = None
        jsonl = self.output_dir / "results.jsonl"
        prior_boot = None
        if self.resume and (self.output_dir / "manifest.json").exists():
            manifest = RunManifest.load(self.output_dir / "manifest.json")
            completed = set(manifest.completed_rung_ids)
            prior_boot = (manifest.fingerprint or {}).get("started_at")
        if manifest is None:
            manifest = self.new_manifest(_run_id(self.spec.name), rungs)
            if jsonl.exists():
                jsonl.unlink()
            for extra in ("sessions.json", "greedy_outputs.jsonl", "logprob_scores.jsonl"):
                (self.output_dir / extra).unlink(missing_ok=True)
        run_id = manifest.run_id

        # Safety and identity before anything touches the server.
        self._preflight(manifest)
        self._write_manifest(manifest)

        watcher = RamWatcher(floor_gib=self.spec.hardware.mem_available_floor_gib, source=self.ram_source, interval_s=0.5)
        try:
            watcher.check_now()
        except RamFloorError as exc:
            manifest.status = "aborted"
            manifest.abort_reason = f"RAM_FLOOR before start: {exc}"
            manifest.finished_utc, manifest.finished_et = stamp_pair()
            self._write_manifest(manifest)
            raise

        if not self.dry_run:
            st = self.hold.guard_and_acquire(
                self.spec.safety.hold_minutes,
                self.spec.safety.hold_reason,
                quiet_hold_file=self.spec.safety.quiet_hold_file,
                read_file=self._read_remote if self.spec.access.mode == "ssh" else None,
            )
            manifest.hold = {"by": st.by, "until": st.until, "acquired_by_runner": True}
            self._write_manifest(manifest)
            watcher.start()

        # Corpora (real files if configured).
        if self.spec.prompts.corpus_dir:
            for ptype in self.spec.prompts.types:
                text, files = load_corpus_dir(self.spec.prompts.corpus_dir, ptype)
                self.corpus_text[ptype] = text
                self.corpus_manifest[ptype] = files
            manifest.corpus_files = self.corpus_manifest

        sessions: dict[tuple[str, str, int], Session] = {}
        saved = self._load_sessions() if self.resume else {}
        refill_pending = bool(saved) and not self.dry_run
        same_boot = bool(prior_boot and self.fp and self.fp.started_at == prior_boot)
        abort_reason = None
        family_index = {p: i for i, p in enumerate(self.spec.prompts.types)}
        last_renew = time.time()
        try:
            for rung in rungs:
                if rung.rung_id in completed:
                    continue
                local: dict[int, Session] = {}
                for sid in rung.live_sessions:
                    key = (rung.family, rung.load_mode, sid)
                    if key not in sessions:
                        sess = Session(
                            sid,
                            rung.prompt_type,
                            seed=session_seed(self.spec.prompts.seed, family_index[rung.prompt_type], sid),
                            chars_per_token=self.spec.prompts.chars_per_token_guess,
                            style=self.spec.prompts.style,
                            file_text=self.corpus_text.get(rung.prompt_type),
                            plant_needles=self.spec.prompts.plant_needles,
                        )
                        skey = "|".join(map(str, key))
                        if skey in saved:
                            sess.load_state(saved[skey])
                        sessions[key] = sess
                    local[sid] = sessions[key]
                extra_flags: list[str] = []
                if refill_pending:
                    # Re-prefill every live session to its high-water context
                    # outside the window; never shrink, never rebuild smaller.
                    for sess in local.values():
                        if sess.messages:
                            self._stream(sess.messages, 1, measured=False)
                    if not same_boot:
                        extra_flags.append("RESUMED_REFILL")
                    refill_pending = False
                if not self.dry_run and time.time() - last_renew > max(self.spec.safety.hold_renew_below_min, 1) * 60:
                    st = self.hold.state()
                    if st.active and st.by == self.hold.name:
                        self.hold.renew(self.spec.safety.hold_minutes, self.spec.safety.hold_reason)
                        last_renew = time.time()
                    else:
                        abort_reason = f"bench-hold no longer ours ({st.raw.strip()[:120]}); stopping"
                        break
                try:
                    watcher.raise_if_tripped()
                    row = self._measure_rung(run_id, rung, local, watcher, extra_flags)
                except RamFloorError as exc:
                    base_seeds = [local[s].seed for s in rung.decoding_sessions]
                    row = self._failed_row(
                        dict(run_id=run_id, rung_id=rung.rung_id, users=rung.users, ctx_per_user=rung.ctx_per_user,
                             prompt_type=rung.prompt_type, load_mode=rung.load_mode, phase=rung.phase,
                             prompt_seed=self.spec.prompts.seed, session_seeds=base_seeds,
                             temperature=self.spec.ladder.temperature, k=self.k_declared,
                             build_fingerprint=self.fp.build_fingerprint if self.fp else None,
                             started_utc=stamp_pair()[0], started_et=stamp_pair()[1]),
                        rung, local, [str(exc)], timed_out=False, since=time.time(), log_lines=[],
                        ram_floor=True, mem_min=watcher.min_gib(),
                    )
                write_jsonl_row(jsonl, row)
                self._save_sessions(sessions)
                if row.exit_reason in ABORTING_EXITS and row.valid == "invalid":
                    if row.exit_reason == "RAM_FLOOR" and self.ram_wait_min > 0 and self._wait_for_ram(watcher):
                        watcher = self._restart_watcher(watcher)
                        continue  # rung not marked complete; next --resume or loop picks it up
                    abort_reason = f"{row.exit_reason}: {row.reason}"
                    break
                manifest.completed_rung_ids.append(rung.rung_id)
                if manifest.attention.actual == "unknown":
                    manifest.attention.actual = row.attention_path
                    manifest.attention.evidence_lines = list(row.attention_evidence)
                if row.quality.greedy_ref == "diverged" and not manifest.first_greedy_divergence:
                    manifest.first_greedy_divergence = {"rung_id": row.rung_id, "token_index": row.quality.greedy_first_divergence}
                self._write_manifest(manifest)
        finally:
            watcher.stop()
            if not self.dry_run:
                released, note = self.hold.release_if_ours()
                manifest.hold = {**(manifest.hold or {}), "released": released, "release_note": note}

        manifest.finished_utc, manifest.finished_et = stamp_pair()
        if abort_reason:
            manifest.status = "aborted"
            manifest.abort_reason = abort_reason
        else:
            manifest.status = "completed"
        self._write_manifest(manifest)
        return manifest

    def _read_remote(self, path: str) -> str | None:
        p = path[2:] if path.startswith("~/") else path
        return self.exec.read_file(p)

    def _wait_for_ram(self, watcher: RamWatcher) -> bool:
        """After a floor abort: wait (bounded) for MemAvailable >= floor + 1 GiB."""
        deadline = time.time() + self.ram_wait_min * 60
        target = self.spec.hardware.mem_available_floor_gib + 1.0
        while time.time() < deadline:
            try:
                if self.ram_source.read_kib() / (1024 * 1024) >= target:
                    return True
            except Exception:  # noqa: BLE001
                return False
            self.sleep(5.0)
        return False

    def _restart_watcher(self, old: RamWatcher) -> RamWatcher:
        old.stop()
        w = RamWatcher(floor_gib=old.floor_gib, source=old.source, interval_s=old.interval_s)
        w.check_now()
        if not self.dry_run:
            w.start()
        return w


def run_spec(
    spec: RunSpec,
    output_root: str | Path,
    dry_run: bool = False,
    resume: bool = False,
    **kwargs: Any,
) -> list[RunManifest]:
    """Run a spec, expanding comparison variants into sequential one-boot runs."""
    manifests: list[RunManifest] = []
    variants = spec.variants or [None]
    for var in variants:
        one = apply_variant(spec, var) if var is not None else spec
        out = Path(output_root) / one.name
        runner = Runner(one, out, dry_run=dry_run, resume=resume, **kwargs)
        manifests.append(runner.run())
    return manifests


def plan_and_explain(spec: RunSpec) -> str:
    from xpu_llm_bench.planner import format_plan

    return format_plan(make_plan(spec))
