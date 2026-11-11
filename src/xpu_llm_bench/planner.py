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


def feasibility(
    spec: RunSpec,
    rungs: list[Rung],
    mem_available_gib: float | None = None,
    basis: str = "live",
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
    peak_tokens = peak_live_tokens(rungs)
    kv = (peak_tokens * a.assumed_kv_bytes_per_token) / GiB
    gpu_need = kv + a.assumed_weight_gib + a.assumed_fixed_overhead_gib
    notes = [
        "KV, GPU and per-token host-RAM figures are planner ESTIMATES (planner.*), not measurements.",
        "Host RAM floor is MemAvailable, not a cgroup max. A run aborts if MemAvailable hits the floor.",
        "Peak live tokens assumes sessions stay resident when quiesced (prefix cache retained).",
    ]
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

    host_need = (peak_tokens * a.host_bytes_per_token) / GiB + kv * a.host_kv_fraction
    floor = hw.mem_available_floor_gib
    if mem_available_gib is None:
        host_status, host_basis, host_budget = "unverified", "none", 0.0
        notes.append(
            f"WARN host RAM: no MemAvailable reading, so the run's estimated host growth "
            f"({host_need:.1f} GiB) is NOT checked. Plan on the bench host (access.mode local), pass "
            "`--mem-available GIB` or `--measured <results.jsonl>`; a live run re-checks at preflight."
        )
    else:
        host_basis = basis
        host_budget = mem_available_gib - floor
        if host_need > host_budget:
            host_status = "fail"
            notes.append(
                f"FAIL host RAM: estimated growth {host_need:.1f} GiB (peak {peak_tokens:,} live tokens x "
                f"{a.host_bytes_per_token / 1024:.0f} KiB) exceeds MemAvailable {mem_available_gib:.1f} GiB "
                f"({basis}) minus the {floor:.1f} GiB floor = {host_budget:.1f} GiB."
            )
        elif host_budget - host_need < a.host_tight_gib:
            host_status = "tight"
            notes.append(
                f"WARN host RAM: estimated growth {host_need:.1f} GiB leaves only "
                f"{host_budget - host_need:.1f} GiB above the floor (MemAvailable {mem_available_gib:.1f} GiB, {basis})."
            )
        else:
            host_status = "ok"
    return Feasibility(
        peak_live_tokens=peak_tokens,
        estimated_kv_gib=kv,
        estimated_total_gib=gpu_need,
        host_ram_gib=hw.host_ram_gib or 0.0,
        mem_floor_gib=floor,
        host_ram_ok=host_status != "fail",
        gpu_total_gib=gpu_total,
        gpu_status=gpu_status,
        host_need_gib=host_need,
        host_budget_gib=host_budget,
        host_status=host_status,
        host_basis=host_basis,
        notes=notes,
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


def make_plan(spec: RunSpec, mem_available_gib: float | None = None, basis: str = "live") -> Plan:
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
        feasibility=(feas := feasibility(spec, rungs, mem_available_gib, basis)),
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
        f"est. GPU weights+KV+overhead: {feas.estimated_total_gib:.2f} / "
        f"{(f'{feas.gpu_total_gib:.0f}' if feas.gpu_total_gib else '?')} GiB ({feas.gpu_status.upper()})"
    )
    if feas.host_status == "unverified":
        lines.append(
            f"Host RAM: est. growth {feas.host_need_gib:.1f} GiB vs MemAvailable: no reading (UNVERIFIED)"
        )
    else:
        lines.append(
            f"Host RAM: est. growth {feas.host_need_gib:.1f} GiB of {feas.host_budget_gib:.1f} GiB "
            f"(MemAvailable {feas.host_budget_gib + feas.mem_floor_gib:.1f} GiB {feas.host_basis} minus a "
            f"{feas.mem_floor_gib:.1f} GiB floor) ({feas.host_status.upper()})"
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
