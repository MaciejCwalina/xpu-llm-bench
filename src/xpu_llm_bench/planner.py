"""`bench plan`: rung order, time estimate, and KV/RAM feasibility."""

from __future__ import annotations

from dataclasses import dataclass, field

from xpu_llm_bench.schedule import Rung, build_schedule, peak_live_tokens, validate_progressive
from xpu_llm_bench.spec import RunSpec

GiB = 1024 ** 3


@dataclass
class Feasibility:
    peak_live_tokens: int
    estimated_kv_gib: float
    estimated_total_gib: float
    host_ram_gib: float
    mem_floor_gib: float
    host_ram_ok: bool
    notes: list[str] = field(default_factory=list)


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


def feasibility(spec: RunSpec, rungs: list[Rung]) -> Feasibility:
    peak_tokens = peak_live_tokens(rungs)
    kv = (peak_tokens * spec.planner.assumed_kv_bytes_per_token) / GiB
    total = kv + spec.planner.assumed_weight_gib + spec.planner.assumed_fixed_overhead_gib
    notes = [
        "KV and RAM figures are planner ESTIMATES (assumed bytes/token and weight size), not measurements.",
        "Host RAM floor is MemAvailable, not a cgroup max. A run aborts if MemAvailable hits the floor.",
        "Peak live tokens assumes sessions stay resident when quiesced (prefix cache retained).",
    ]
    # Host RAM is 25 GiB with a 2 GiB MemAvailable floor — KV mostly lives in GPU
    # memory, but host can still be pressured by the server process. Flag if the
    # *host* budget looks tight; GPU KV is reported separately.
    host_budget = spec.hardware.host_ram_gib - spec.hardware.mem_available_floor_gib
    host_ok = spec.planner.assumed_fixed_overhead_gib < host_budget
    if not host_ok:
        notes.append(
            f"Planner warning: assumed host overhead "
            f"{spec.planner.assumed_fixed_overhead_gib:.1f} GiB may not leave "
            f"a {spec.hardware.mem_available_floor_gib:.1f} GiB MemAvailable floor "
            f"on a {spec.hardware.host_ram_gib:.1f} GiB host."
        )
    if kv > 40:  # 2x 24 GB B60 is 48 GB; leave headroom
        notes.append(
            f"Planner warning: estimated KV {kv:.1f} GiB may not fit two 24 GB GPUs "
            f"with {spec.planner.assumed_weight_gib:.1f} GiB weights."
        )
    return Feasibility(
        peak_live_tokens=peak_tokens,
        estimated_kv_gib=kv,
        estimated_total_gib=total,
        host_ram_gib=spec.hardware.host_ram_gib,
        mem_floor_gib=spec.hardware.mem_available_floor_gib,
        host_ram_ok=host_ok,
        notes=notes,
    )


def make_plan(spec: RunSpec) -> Plan:
    rungs = build_schedule(spec)
    issues = validate_progressive(rungs)
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
        feasibility=feasibility(spec, rungs),
        issues=issues,
    )


def format_plan(plan: Plan) -> str:
    lines: list[str] = []
    lines.append(f"# Plan for {plan.spec_name}  (profile={plan.profile})")
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
        f"est. weights+KV+overhead: {feas.estimated_total_gib:.2f} GiB"
    )
    lines.append(
        f"Host RAM: {feas.host_ram_gib:.1f} GiB with a {feas.mem_floor_gib:.1f} GiB "
        f"MemAvailable floor ({'ok' if feas.host_ram_ok else 'TIGHT'})"
    )
    lines.append("")
    lines.append("Order (progressive; context never shrinks within a prompt-type family):")
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
