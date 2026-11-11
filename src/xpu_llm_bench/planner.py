"""`bench plan`: rung order, time estimate, and KV/RAM feasibility."""

from __future__ import annotations

from dataclasses import dataclass, field

from xpu_llm_bench.schedule import Rung, build_schedule, peak_live_tokens, validate_progressive
from xpu_llm_bench.spec import RunSpec

GiB = 1024 ** 3


FeasStatus = str  # "ok" | "tight" | "fail"


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

    @property
    def ok(self) -> bool:
        return self.gpu_status != "fail" and self.host_status != "fail"

    @property
    def failures(self) -> list[str]:
        return [n for n in self.notes if n.startswith("FAIL")]

@dataclass
class Plan:
    spec_name: str
    profile: str
    rungs: list[Rung]
    estimated_seconds: float
    target_minutes: float | None
    fits_target: bool | None
    feasibility: Feasibility
    issues: list[str] = field(default_factory=list)

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
        warmup_s = 0.0
        if spec.ladder.warmup.requests and spec.ladder.warmup.tokens:
            warmup_s = (spec.ladder.warmup.tokens / max(a.assumed_decode_tok_s_per_user, 1e-6)) * spec.ladder.warmup.requests
        quality_s = a.quality_s_per_rung if spec.quality.enabled else 0.0
        total += prefill_s + decode_s + warmup_s + quality_s + a.overhead_s_per_rung
    return total


def feasibility(spec: RunSpec, rungs: list[Rung], measured_host_gib: float | None = None) -> Feasibility:
    """GPU and host-RAM feasibility of the planned peak.

    GPU: weights + KV(peak live tokens) + device overhead against
    gpu_count x gpu_mem_gib. Host: server-process RAM (measured RSS when
    given, else the planner ESTIMATE) plus any host-resident KV share,
    against host_ram_gib minus the MemAvailable floor.
    """
    a = spec.planner
    hw = spec.hardware
    peak_tokens = peak_live_tokens(rungs)
    kv = (peak_tokens * a.assumed_kv_bytes_per_token) / GiB
    gpu_need = kv + a.assumed_weight_gib + a.assumed_fixed_overhead_gib
    gpu_total = hw.gpu_count * hw.gpu_mem_gib
    notes = [
        "KV and GPU figures are planner ESTIMATES (assumed bytes/token and weight size), not measurements.",
        "Host RAM floor is MemAvailable, not a cgroup max. A run aborts if MemAvailable hits the floor.",
        "Peak live tokens assumes sessions stay resident when quiesced (prefix cache retained).",
    ]
    if gpu_need > gpu_total * a.gpu_fail_frac:
        gpu_status = "fail"
        notes.append(
            f"FAIL GPU: estimated weights+KV+overhead {gpu_need:.1f} GiB exceeds "
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
    if measured_host_gib is not None and measured_host_gib > 0:
        proc_gib, basis = measured_host_gib, "measured"
    else:
        proc_gib, basis = a.assumed_host_process_gib, "estimate"
    host_need = proc_gib + kv * a.host_kv_fraction
    host_budget = hw.host_ram_gib - hw.mem_available_floor_gib
    if host_need > host_budget:
        host_status = "fail"
        notes.append(
            f"FAIL host RAM: server processes {proc_gib:.1f} GiB ({basis}) + host KV "
            f"{kv * a.host_kv_fraction:.1f} GiB = {host_need:.1f} GiB exceeds the "
            f"{host_budget:.1f} GiB left above the {hw.mem_available_floor_gib:.1f} GiB MemAvailable floor."
        )
    elif host_budget - host_need < a.host_tight_gib:
        host_status = "tight"
        notes.append(
            f"WARN host RAM: {host_need:.1f} GiB ({basis}) leaves only "
            f"{host_budget - host_need:.1f} GiB above the MemAvailable floor."
        )
    else:
        host_status = "ok"
    if basis == "estimate":
        notes.append(
            f"Host process RAM is the planner ESTIMATE ({proc_gib:.1f} GiB). Pass "
            "`bench plan --measured <results.jsonl>` to use measured RSS; a live run re-checks "
            "with the server's measured RSS at preflight."
        )
    return Feasibility(
        peak_live_tokens=peak_tokens,
        estimated_kv_gib=kv,
        estimated_total_gib=gpu_need,
        host_ram_gib=hw.host_ram_gib,
        mem_floor_gib=hw.mem_available_floor_gib,
        host_ram_ok=host_status != "fail",
        gpu_total_gib=gpu_total,
        gpu_status=gpu_status,
        host_need_gib=host_need,
        host_budget_gib=host_budget,
        host_status=host_status,
        host_basis=basis,
        notes=notes,
    )


def measured_host_gib_from_rows(rows: list) -> float | None:
    """Peak summed server RSS (GiB) across result rows that recorded rss_kib."""
    best = 0.0
    for row in rows:
        rss = getattr(row, "rss_kib", None) or {}
        total = sum(int(v) for v in rss.values() if v)
        best = max(best, total / (1024 * 1024))
    return best or None


GRID_WARNING = (
    "!!! GRID LADDER: every user starts at once at the first rung and the run then steps "
    "down. This is NOT the progressive ladder (session 1 grows, then users join). "
    "Only use it if you explicitly need the old grid order. !!!"
)


def make_plan(spec: RunSpec, measured_host_gib: float | None = None) -> Plan:
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
        )
    return Plan(
        spec_name=spec.name,
        profile=spec.profile,
        rungs=rungs,
        estimated_seconds=seconds,
        target_minutes=target,
        fits_target=fits,
        feasibility=(feas := feasibility(spec, rungs, measured_host_gib)),
        issues=issues + feas.failures,
    )


def format_plan(plan: Plan) -> str:
    lines: list[str] = []
    lines.append(f"# Plan for {plan.spec_name}  (profile={plan.profile})")
    lines.append("")
    if any(i == GRID_WARNING for i in plan.issues):
        lines.append(GRID_WARNING)
        lines.append("")
    lines.append(f"Rungs: {len(plan.rungs)}  phases={sorted({r.phase for r in plan.rungs})}")
    lines.append(
        f"Estimated wall time: {plan.estimated_minutes:.1f} min "
        f"({plan.estimated_seconds:.0f} s)"
        + (f"  target={plan.target_minutes:.1f} min" if plan.target_minutes is not None else "")
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
        f"est. GPU weights+KV+overhead: {feas.estimated_total_gib:.2f} / {feas.gpu_total_gib:.0f} GiB "
        f"({feas.gpu_status.upper()})"
    )
    lines.append(
        f"Host RAM: need {feas.host_need_gib:.1f} GiB ({feas.host_basis}) of "
        f"{feas.host_budget_gib:.1f} GiB usable ({feas.host_ram_gib:.1f} GiB minus a "
        f"{feas.mem_floor_gib:.1f} GiB MemAvailable floor) ({feas.host_status.upper()})"
    )
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
