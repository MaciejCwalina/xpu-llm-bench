"""`bench plan`: rung order, time estimate, and KV/RAM feasibility."""

from __future__ import annotations

from dataclasses import dataclass, field

from xpu_llm_bench.schedule import (
    Rung,
    build_schedule,
    ceiling_key,
    peak_live_tokens,
    validate_progressive,
)
from xpu_llm_bench.spec import RunSpec, resolve_contexts, with_contexts

GiB = 1024 ** 3


FeasStatus = str  # "ok" | "tight" | "fail"


@dataclass
class CeilingEstimate:
    """Estimated context ceiling of one (family, load mode, users) line."""

    key: str
    family: str
    load_mode: str
    users: int
    top_ok: int | None  # highest rung estimated to fit (None: not even the first)
    stops_at: int | None  # first rung estimated not to fit (None: reaches the top)
    reason: str  # "fits" or "kv_capacity"
    limit_tokens_per_user: int | None = None  # KV capacity / users - output window


def rung_kv_tokens(rung: Rung) -> int:
    """KV tokens a rung needs at once: every live context plus each decoder's output window."""
    return sum(rung.target_for(s) for s in rung.live_sessions) + len(rung.decoding_sessions) * rung.output_tokens


def kv_capacity_estimate(spec: RunSpec) -> int | None:
    """KV token capacity from the planner's GPU assumptions (None if GPU size is unknown).

    A live run uses the server's own figure (boot log "KV cache size: N tokens"
    or vllm:cache_config_info) instead.
    """
    a = spec.planner
    hw = spec.hardware
    if not hw.gpu_count or not hw.gpu_mem_gib or a.assumed_kv_bytes_per_token <= 0:
        return None
    free = hw.gpu_count * hw.gpu_mem_gib * a.gpu_fail_frac - a.assumed_weight_gib - a.assumed_fixed_overhead_gib
    return max(0, int(free * GiB / a.assumed_kv_bytes_per_token))


def ceiling_estimates(
    spec: RunSpec,
    rungs: list[Rung],
    kv_capacity_tokens: int | None = None,
    mem_available_gib: float | None = None,
) -> list[CeilingEstimate]:
    """Per (family, load mode, users): the highest rung that fits.

    A rung fits when its live KV tokens (``rung_kv_tokens``) fit the KV
    capacity and its host-RAM growth (live tokens x planner.host_bytes_per_token
    plus any host KV share) fits MemAvailable minus the floor. Unknown limits
    are not checked. Lines climb rung by rung, so the first rung that does not
    fit is the estimated ceiling.
    """
    a = spec.planner
    host_budget = None
    if mem_available_gib is not None:
        host_budget = mem_available_gib - spec.hardware.mem_available_floor_gib
    by_key: dict[str, list[Rung]] = {}
    for r in rungs:
        by_key.setdefault(ceiling_key(r), []).append(r)
    out: list[CeilingEstimate] = []
    for key, rs in by_key.items():
        rs = sorted(rs, key=lambda r: r.ctx_per_user)
        top_ok = None
        stops_at = None
        reason = "fits"
        for r in rs:
            kv_tokens = rung_kv_tokens(r)
            live = sum(r.target_for(s) for s in r.live_sessions)
            host_need = (live * a.host_bytes_per_token + kv_tokens * a.assumed_kv_bytes_per_token * a.host_kv_fraction) / GiB
            if kv_capacity_tokens is not None and kv_tokens > kv_capacity_tokens:
                stops_at, reason = r.ctx_per_user, "kv_capacity"
                break
            if host_budget is not None and host_need > host_budget:
                stops_at, reason = r.ctx_per_user, "host_ram"
                break
            top_ok = r.ctx_per_user
        users = rs[0].users
        limit = None
        if kv_capacity_tokens is not None:
            limit = kv_capacity_tokens // users - rs[0].output_tokens
        out.append(CeilingEstimate(key, rs[0].family, rs[0].load_mode, users, top_ok, stops_at, reason, limit))
    return out


@dataclass
class Feasibility:
    peak_live_tokens: int
    estimated_kv_gib: float
    estimated_total_gib: float  # GPU side: weights + KV + device overhead
    host_ram_gib: float
    mem_floor_gib: float
    host_ram_ok: bool
    gpu_total_gib: float = 0.0
    gpu_status: FeasStatus = "ok"
    host_need_gib: float = 0.0
    host_budget_gib: float = 0.0
    host_status: FeasStatus = "ok"
    host_basis: str = "estimate"  # "estimate" or "measured"
    notes: list[str] = field(default_factory=list)
    ceilings: list[CeilingEstimate] = field(default_factory=list)
    kv_capacity_tokens: int | None = None
    kv_capacity_basis: str = ""

    @property
    def ok(self) -> bool:
        """Whether the ESTIMATES fit. Informational: neither GPU nor host
        estimates block a run; live ceilings stop a user count instead."""
        return self.gpu_status != "fail"

    @property
    def failures(self) -> list[str]:
        # Estimates never block a run (pre-run checks only look at live load).
        return []

@dataclass
class Plan:
    spec_name: str
    profile: str
    rungs: list[Rung]
    estimated_seconds: float
    target_minutes: float | None
    fits_target: bool | None
    feasibility: Feasibility
    repeats: int = 1
    issues: list[str] = field(default_factory=list)
    contexts: list[int] = field(default_factory=list)
    context_note: str = ""
    max_model_len: int | None = None

    @property
    def estimated_minutes(self) -> float:
        return self.estimated_seconds / 60.0


def estimate_seconds(spec: RunSpec, rungs: list[Rung]) -> float:
    """Wall-time estimate from planning assumptions, not from measurements."""
    a = spec.planner
    total = 0.0
    # Session context already reached, per (family, mode, sid)
    reached: dict[tuple[str, str, int], int] = {}
    for rung in rungs:
        prefill_tokens = 0
        for sid in rung.grow_sessions:
            key = (rung.family, rung.load_mode, sid)
            prev = reached.get(key, 0)
            target = rung.target_for(sid)
            prefill_tokens += max(target - prev, 0)
            reached[key] = max(prev, target)
        prefill_s = prefill_tokens / max(a.assumed_prefill_tok_s, 1e-6)
        # Closed-all and staggered: decode window ≈ output tokens / per-user rate
        # (users decode in parallel). Open-loop uses the same window as a lower bound.
        decode_s = rung.output_tokens / max(a.assumed_decode_tok_s_per_user, 1e-6)
        # ladder.repeats re-measures the window on the same context; the prompt
        # is not regrown (the server's prefix cache serves the re-send).
        decode_s *= max(spec.ladder.repeats, 1)
        warmup_s = 0.0
        if spec.ladder.warmup.requests and spec.ladder.warmup.tokens:
            warmup_s = (spec.ladder.warmup.tokens / max(a.assumed_decode_tok_s_per_user, 1e-6)) * spec.ladder.warmup.requests
        quality_s = a.quality_s_per_rung if spec.quality.enabled else 0.0
        total += prefill_s + decode_s + warmup_s + quality_s + a.overhead_s_per_rung
    return total


def feasibility(
    spec: RunSpec,
    rungs: list[Rung],
    mem_available_gib: float | None = None,
    basis: str = "live",
    kv_capacity_tokens: int | None = None,
    kv_capacity_basis: str = "",
) -> Feasibility:
    """GPU and host-RAM feasibility of the planned peak.

    GPU: weights + KV(peak live tokens) + device overhead against
    gpu_count x gpu_mem_gib (skipped, with a warning, if either is unknown).

    Host: the run's host-RAM growth (peak live tokens x
    planner.host_bytes_per_token, plus any host-resident KV share) against
    MemAvailable *now* minus the floor. MemAvailable already accounts for the
    booted server and every other resident process. ``mem_available_gib``
    comes from a live reading (plan with access.mode local, or preflight) or
    a measured baseline from an earlier run's results. Without a reading the
    host check is UNVERIFIED (a warning), never OK.
    """
    a = spec.planner
    hw = spec.hardware
    ceilings: list[CeilingEstimate] = []
    ceiling_notes: list[str] = []
    if kv_capacity_tokens is None:
        kv_capacity_tokens = kv_capacity_estimate(spec)
        if kv_capacity_tokens is not None:
            kv_capacity_basis = "planner estimate (hardware.gpu_* and planner.assumed_*)"
    if spec.ladder.stop_at_ceiling and spec.ladder.shape != "grid":
        # Each user count climbs until it stops fitting, then the run records
        # CTX_CEILING and moves on. Check fit on the rungs estimated to fit;
        # the live run finds the real ceilings.
        ceilings = ceiling_estimates(spec, rungs, kv_capacity_tokens)
        tops = {c.key: c.top_ok for c in ceilings}
        fitting = [r for r in rungs if tops.get(ceiling_key(r)) is not None and r.ctx_per_user <= tops[ceiling_key(r)]]
        stopped = [c for c in ceilings if c.stops_at is not None]
        if stopped:
            ceiling_notes.append(
                f"{len(stopped)} user-count line(s) are estimated to stop below the top rung (see the ceiling "
                "table). The run climbs each line until it stops fitting, records CTX_CEILING and continues "
                "with the next user count; the GPU and host checks below cover the rungs estimated to fit."
            )
        rungs = fitting or rungs[:1]
    peak_tokens = peak_live_tokens(rungs)
    kv = (peak_tokens * a.assumed_kv_bytes_per_token) / GiB
    gpu_need = kv + a.assumed_weight_gib + a.assumed_fixed_overhead_gib
    notes = [
        "KV, GPU and per-token host-RAM figures are planner ESTIMATES (planner.*), not measurements.",
        "Peak live tokens assumes sessions stay resident when quiesced (prefix cache retained).",
    ] + ceiling_notes
    gpu_total = 0.0
    if not hw.gpu_count or not hw.gpu_mem_gib:
        gpu_status = "unknown"
        notes.append(
            "WARN GPU: hardware.gpu_count / gpu_mem_gib not set (site config `hardware:`); "
            f"GPU fit not checked (estimated need {gpu_need:.1f} GiB)."
        )
    else:
        gpu_total = hw.gpu_count * hw.gpu_mem_gib
        if gpu_need > gpu_total * a.gpu_fail_frac:
            gpu_status = "fail"
            notes.append(
                f"WARN GPU (estimate, does not block): estimated weights+KV+overhead {gpu_need:.1f} GiB exceeds "
                f"{a.gpu_fail_frac:.0%} of {gpu_total:.0f} GiB ({hw.gpu_count} x {hw.gpu_mem_gib:.0f} GiB). "
                "Lower the top context rung or concurrency."
            )
        elif gpu_need > gpu_total * a.gpu_warn_frac:
            gpu_status = "tight"
            notes.append(
                f"WARN GPU: estimated {gpu_need:.1f} GiB is over {a.gpu_warn_frac:.0%} of {gpu_total:.0f} GiB; "
                "expect preemption or KV pressure at the peak."
            )
        else:
            gpu_status = "ok"

    host_need = (peak_tokens * a.host_bytes_per_token) / GiB + kv * a.host_kv_fraction
    floor = hw.mem_available_floor_gib
    # Host RAM: an ESTIMATE shown for information only. It never blocks a run,
    # a user count or a bigger ladder; the live RAM floor (runtime) stops a
    # user count instead and records CTX_CEILING.
    if mem_available_gib is None:
        host_status, host_basis, host_budget = "unverified", "none", 0.0
    else:
        host_basis = basis
        host_budget = mem_available_gib - floor
        if host_need > host_budget:
            host_status = "over"
        elif host_budget - host_need < a.host_tight_gib:
            host_status = "tight"
        else:
            host_status = "ok"
    return Feasibility(
        peak_live_tokens=peak_tokens,
        estimated_kv_gib=kv,
        estimated_total_gib=gpu_need,
        host_ram_gib=hw.host_ram_gib or 0.0,
        mem_floor_gib=floor,
        host_ram_ok=host_status != "over",
        gpu_total_gib=gpu_total,
        gpu_status=gpu_status,
        host_need_gib=host_need,
        host_budget_gib=host_budget,
        host_status=host_status,
        host_basis=host_basis,
        notes=notes,
        ceilings=ceilings,
        kv_capacity_tokens=kv_capacity_tokens,
        kv_capacity_basis=kv_capacity_basis,
    )


def baseline_mem_available_from_rows(rows: list) -> float | None:
    """MemAvailable (GiB) measured on the earliest row of an earlier run on the
    same host: the server was booted and the context was smallest there."""
    for row in rows:
        val = getattr(row, "mem_available_min_gib", None)
        if val:
            return float(val)
    return None


def local_mem_available_gib(path: str = "/proc/meminfo") -> float | None:
    try:
        text = open(path, encoding="utf-8").read()  # noqa: SIM115
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    return None


GRID_WARNING = (
    "!!! GRID LADDER: every user starts at once at the first rung and the run then steps "
    "down. This is NOT the progressive ladder (session 1 grows, then users join). "
    "Only use it if you explicitly need the old grid order. !!!"
)


def make_plan(
    spec: RunSpec,
    mem_available_gib: float | None = None,
    basis: str = "live",
    max_model_len: int | None = None,
    kv_capacity_tokens: int | None = None,
) -> Plan:
    ctxs, ctx_note = resolve_contexts(spec.ladder, max_model_len)
    spec = with_contexts(spec, ctxs)
    rungs = build_schedule(spec)
    issues = validate_progressive(rungs)
    if spec.ladder.shape == "grid":
        issues.insert(0, GRID_WARNING)
    seconds = estimate_seconds(spec, rungs)
    target = spec.ladder.target_minutes
    fits = None if target is None else (seconds / 60.0) <= target
    if fits is False:
        issues.append(
            f"Estimated {seconds / 60.0:.1f} min exceeds the {target:.1f} min target. "
            "Shorten output_tokens_per_rung, drop a load mode, or use --profile quick."
            + (f" repeats={spec.ladder.repeats} multiplies every decode window; fewer --repeats is shorter."
               if spec.ladder.repeats > 1 else "")
        )
    return Plan(
        spec_name=spec.name,
        profile=spec.profile,
        rungs=rungs,
        estimated_seconds=seconds,
        target_minutes=target,
        repeats=spec.ladder.repeats,
        fits_target=fits,
        feasibility=(feas := feasibility(
            spec, rungs, mem_available_gib, basis, kv_capacity_tokens,
            "given" if kv_capacity_tokens is not None else "",
        )),
        issues=issues + feas.failures,
        contexts=ctxs,
        context_note=ctx_note,
        max_model_len=max_model_len,
    )


def format_plan(plan: Plan) -> str:
    lines: list[str] = []
    lines.append(f"# Plan for {plan.spec_name}  (profile={plan.profile})")
    lines.append("")
    if any(i == GRID_WARNING for i in plan.issues):
        lines.append(GRID_WARNING)
        lines.append("")
    lines.append(f"Rungs: {len(plan.rungs)}  phases={sorted({r.phase for r in plan.rungs})}")
    if plan.contexts:
        lines.append(
            "Context rungs: " + ", ".join(ctx_k(c) for c in plan.contexts)
            + (f"  ({plan.context_note})" if plan.context_note else "")
        )
    lines.append(
        f"Estimated wall time: {plan.estimated_minutes:.1f} min "
        f"({plan.estimated_seconds:.0f} s)"
        + (f"  target={plan.target_minutes:.1f} min" if plan.target_minutes is not None else "")
        + (f"  repeats={plan.repeats} per rung" if plan.repeats > 1 else "")
        + ("  OK" if plan.fits_target else "" if plan.fits_target is None else "  OVER TARGET")
    )
    feas = plan.feasibility
    lines.append(
        "Estimate uses planner.assumed_* rates (not measurements); "
        "a --resume onto a fresh boot adds a full re-prefill of every live session."
    )
    lines.append(
        f"Peak live tokens: {feas.peak_live_tokens:,}  "
        f"est. KV: {feas.estimated_kv_gib:.2f} GiB  "
        f"est. GPU weights+KV+overhead: {feas.estimated_total_gib:.2f} / "
        f"{(f'{feas.gpu_total_gib:.0f}' if feas.gpu_total_gib else '?')} GiB ({'OVER, estimate only' if feas.gpu_status == 'fail' else feas.gpu_status.upper()})"
    )
    if feas.host_status == "unverified":
        lines.append(
            f"Host RAM (info only, never blocks): est. growth {feas.host_need_gib:.1f} GiB; no MemAvailable reading"
        )
    else:
        lines.append(
            f"Host RAM (info only, never blocks): est. growth {feas.host_need_gib:.1f} GiB vs "
            f"{feas.host_budget_gib:.1f} GiB above the {feas.mem_floor_gib:.1f} GiB floor "
            f"(MemAvailable {feas.host_budget_gib + feas.mem_floor_gib:.1f} GiB {feas.host_basis}) "
            f"({feas.host_status.upper()}). A live run stops a user count at the RAM floor (CTX_CEILING)."
        )
    if feas.ceilings:
        lines.append("")
        cap = (f"{feas.kv_capacity_tokens:,} tokens ({feas.kv_capacity_basis})"
               if feas.kv_capacity_tokens is not None else "unknown (set hardware.gpu_* or pass --kv-capacity-tokens)")
        lines.append(f"Context ceiling per user count (ESTIMATE; KV capacity {cap}):")
        for c in feas.ceilings:
            top = ctx_k(c.top_ok) if c.top_ok else "none"
            if c.stops_at is None:
                verdict = f"reaches the top rung ({top})"
            else:
                verdict = f"up to {top}, stops at {ctx_k(c.stops_at)} ({c.reason})"
            per_user = f"; ~{c.limit_tokens_per_user:,} tokens/user fit" if c.limit_tokens_per_user is not None else ""
            lines.append(f"  {c.family} {c.load_mode} {c.users}u: {verdict}{per_user}")
    lines.append("")
    lines.append(
        "Order (progressive; context never shrinks within a prompt-type family):"
        if not any(i == GRID_WARNING for i in plan.issues)
        else "Order (GRID):"
    )
    lines.append("")
    lines.append(
        f"{'seq':>4}  {'rung_id':<48}  {'grow':>8}  {'decode':>16}  {'live':>4}  prefill  contexts"
    )
    for i, rung in enumerate(plan.rungs, 1):
        grow = ",".join(str(s) for s in rung.grow_sessions) or "-"
        dec = ",".join(str(s) for s in rung.decoding_sessions)
        ctxs = "/".join(str(rung.target_for(s)) for s in rung.live_sessions)
        lines.append(
            f"{i:4d}  {rung.rung_id:<48}  {grow:>8}  {dec:>16}  {len(rung.live_sessions):4d}  "
            f"{'yes' if rung.measures_prefill else 'no ':<7}  {ctxs}"
        )
    if plan.issues:
        lines.append("")
        lines.append("Issues:")
        for item in plan.issues:
            lines.append(f"- {item}")
    lines.append("")
    lines.append("Planner notes:")
    for note in feas.notes:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)



def ctx_k(n: int) -> str:
    """Context label, same convention as the HTML report: 8K, 64K, 131K, 196K, 262K."""
    from xpu_llm_bench.html_report import ctx_label

    return ctx_label(n)
