"""Single-file HTML report with inline SVG charts drawn here.

No JavaScript chart library, no CDN, no fonts or images from the network: the
file opens offline. Each headline metric gets one chart per prompt type and
load mode: X = context length per user (ladder order), Y = the metric, one
line per concurrency (users). Flagged rows are drawn hollow; every point has
an SVG <title> tooltip with the exact value and its flags.
"""

from __future__ import annotations

import html
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from xpu_llm_bench.results import FLAGS, RungResult, RunManifest

# (field, title, unit, scale, needs_spec_decode)
METRICS: list[tuple[str, str, str, float, bool]] = [
    ("decode_tok_s_per_user", "Decode speed", "tok/s per user", 1.0, False),
    ("prefill_tok_s", "Prefill speed", "tok/s", 1.0, False),
    ("ttft_ms", "Time to first token", "ms", 1.0, False),
    ("steps_s_per_user", "Verify steps", "steps/s per user", 1.0, False),
    ("accepted_drafts_per_step", "MTP accepted drafts per step", "drafts/step", 1.0, True),
    ("hit_rate", "True hit rate", "%", 100.0, True),
]
# Flags that make a point hollow (the number is suspect or not what it says).
HOLLOW_FLAGS = {"NOT_AS_LABELED", "QSA_FALLBACK", "MIXED_PATH", "GRAPH_MISS", "HIT_RATE_UNKNOWN",
                "SPEC_COUNTER_BROKEN", "DRY_RUN"}
TABLE_COLUMNS = [
    "rung_id", "phase", "users", "ctx_per_user", "prompt_type", "load_mode", "attention_path",
    "decode_tok_s_per_user", "steps_s_per_user", "decode_tok_s_total", "tokens_per_step",
    "accepted_drafts_per_step", "hit_rate", "prefill_tok_s", "ttft_ms", "tpot_ms", "itl_p50_ms", "itl_p99_ms",
    "kv_cache_usage_max_pct", "mem_available_min_gib", "exit_reason", "flags", "valid",
]
# Series colours by users (works on light and dark backgrounds).
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]
W, H = 520, 300
ML, MR, MT, MB = 64, 16, 16, 46

CSS = """
:root{color-scheme:light dark;--bg:#fff;--fg:#1d1d1f;--muted:#6b6b70;--grid:#e3e3e8;--card:#f6f6f8;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececf1;--muted:#a0a0aa;--grid:#33333a;--card:#1d1d21;--bad:#ff8a80}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;margin:24px;max-width:1700px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 8px}h3{font-size:15px;margin:16px 0 6px}
.muted{color:var(--muted)}.bad{color:var(--bad)}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid var(--grid);padding:3px 6px;text-align:left;vertical-align:top}
th{background:var(--card);position:sticky;top:0}
.hdr td:first-child{font-weight:600;white-space:nowrap}
.grid{display:flex;flex-wrap:wrap;gap:14px}.card{background:var(--card);border-radius:8px;padding:8px 10px}
.card h3{margin:2px 0 4px}
svg text{fill:var(--fg);font-size:11px}svg .axis{stroke:var(--muted)}svg .gl{stroke:var(--grid)}
pre{white-space:pre-wrap;word-break:break-all;background:var(--card);padding:8px;border-radius:6px;font-size:12px}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin:4px 0 8px;font-size:12px}
.legend span{display:inline-flex;align-items:center;gap:4px}
.note{background:var(--card);border-left:3px solid var(--muted);padding:6px 10px;margin:6px 0}
.wide{overflow-x:auto}
"""


def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def ctx_label(n: int | None) -> str:
    if n is None:
        return "?"
    if n >= 1024 and n % 1024 == 0:
        k = n // 1024
        # Project convention: 131072 reads "131K" (8192 -> 8K, 65536 -> 64K).
        return f"{round(n / 1000)}K" if k >= 100 else f"{k}K"
    return f"{n / 1024:.1f}K" if n >= 1024 else str(n)


def _fmt(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _tick(v: float, step: float) -> str:
    decimals = max(0, -math.floor(math.log10(step))) if step < 1 else 0
    if step < 1 and round(step * 10 ** decimals) % 10 and step * 10 ** decimals % 1:
        decimals += 1
    return f"{v:,.{decimals}f}"


def _nice_max(v: float) -> tuple[float, float]:
    """Axis top and tick step for data maximum v (>0)."""
    if v <= 0:
        return 1.0, 0.25
    raw = v / 5
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    return step * math.ceil(v / step * 1.0001), step


@dataclass
class Point:
    ctx: int
    value: float
    flags: list[str]
    rung_ids: list[str]
    valid: str


def _series(rows: list[RungResult], field: str, scale: float) -> dict[int, list[Point]]:
    """users -> points ordered by context. Duplicate (users, ctx) rows are averaged."""
    acc: dict[tuple[int, int], list[RungResult]] = defaultdict(list)
    for r in rows:
        if r.ctx_per_user is None:
            continue
        acc[(r.users, r.ctx_per_user)].append(r)
    out: dict[int, list[Point]] = defaultdict(list)
    for (users, ctx), group in sorted(acc.items()):
        vals = [getattr(r, field) for r in group]
        nums = [float(v) * scale for v in vals if isinstance(v, (int, float))]
        if not nums:
            continue
        flags = sorted({f for r in group for f in r.flags})
        valid = "valid" if all(r.valid == "valid" for r in group) else "invalid"
        out[users].append(Point(ctx, sum(nums) / len(nums), flags, [r.rung_id for r in group], valid))
    return out


def _color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def svg_chart(
    runs: list[tuple[str, list[RungResult]]],
    field: str,
    title: str,
    unit: str,
    scale: float,
    users_order: list[int],
) -> str:
    """One chart. ``runs`` holds (label, rows); run 0 is solid, others dashed."""
    data = [(label, _series(rows, field, scale)) for label, rows in runs]
    ctxs = sorted({p.ctx for _, s in data for pts in s.values() for p in pts})
    vals = [p.value for _, s in data for pts in s.values() for p in pts]
    head = f'<div class="card"><h3>{_e(title)} <span class="muted">({_e(unit)})</span></h3>'
    if not ctxs:
        return head + '<p class="muted">No data for this metric.</p></div>'
    top, step = _nice_max(max(vals))
    pw, ph = W - ML - MR, H - MT - MB

    def x(ctx: int) -> float:
        if len(ctxs) == 1:
            return ML + pw / 2
        return ML + pw * ctxs.index(ctx) / (len(ctxs) - 1)

    def y(v: float) -> float:
        return MT + ph * (1 - v / top)

    parts = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
             f'aria-label="{_e(title)} by context length, one line per user count">']
    t = 0.0
    while t <= top * 1.0001:
        parts.append(f'<line class="gl" x1="{ML}" x2="{W - MR}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        parts.append(f'<text x="{ML - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{_e(_tick(t, step))}</text>')
        t += step
    parts.append(f'<line class="axis" x1="{ML}" x2="{ML}" y1="{MT}" y2="{MT + ph}"/>')
    parts.append(f'<line class="axis" x1="{ML}" x2="{W - MR}" y1="{MT + ph}" y2="{MT + ph}"/>')
    for c in ctxs:
        parts.append(f'<text x="{x(c):.1f}" y="{MT + ph + 16}" text-anchor="middle">{ctx_label(c)}</text>')
    parts.append(f'<text x="{ML + pw / 2:.1f}" y="{H - 8}" text-anchor="middle">context per user (tokens)</text>')
    parts.append(f'<text transform="translate(14 {MT + ph / 2:.1f}) rotate(-90)" text-anchor="middle">'
                 f'{_e(title)} ({_e(unit)})</text>')
    for ri, (label, series) in enumerate(data):
        dash = "" if ri == 0 else ' stroke-dasharray="6 4"'
        for users in users_order:
            pts = sorted(series.get(users, []), key=lambda p: ctxs.index(p.ctx))
            if not pts:
                continue
            col = _color(users_order.index(users))
            name = f"{users} user{'s' if users != 1 else ''}" + (f" [{label}]" if len(data) > 1 else "")
            cls = f"s u{users} r{ri}"
            if len(pts) > 1:
                d = " ".join(f"{'M' if i == 0 else 'L'}{x(p.ctx):.1f},{y(p.value):.1f}" for i, p in enumerate(pts))
                parts.append(f'<path class="{cls}" data-users="{users}" d="{d}" fill="none" stroke="{col}" '
                             f'stroke-width="2"{dash}><title>{_e(name)}</title></path>')
            for p in pts:
                hollow = bool(HOLLOW_FLAGS & set(p.flags)) or p.valid != "valid"
                fill = "none" if hollow else col
                tip = f"{name}, {ctx_label(p.ctx)} context: {_fmt(p.value)} {unit}"
                if len(p.rung_ids) > 1:
                    tip += f" (mean of {len(p.rung_ids)} rows)"
                if p.flags:
                    tip += "\nflags: " + ", ".join(p.flags)
                if p.valid != "valid":
                    tip += "\nrow not valid"
                tip += "\nrung: " + ", ".join(p.rung_ids)
                shape = (f'<rect x="{x(p.ctx) - 4:.1f}" y="{y(p.value) - 4:.1f}" width="8" height="8"'
                         if ri else f'<circle cx="{x(p.ctx):.1f}" cy="{y(p.value):.1f}" r="4"')
                parts.append(f'{shape} class="pt{" hollow" if hollow else ""}" data-users="{users}" fill="{fill}" '
                             f'stroke="{col}" stroke-width="2"><title>{_e(tip)}</title>'
                             f'{"</rect>" if ri else "</circle>"}')
    parts.append("</svg>")
    return head + "".join(parts) + "</div>"


def _legend(users_order: list[int], labels: list[str]) -> str:
    items = []
    for i, u in enumerate(users_order):
        items.append(f'<span><svg width="22" height="10"><line x1="0" x2="22" y1="5" y2="5" stroke="{_color(i)}" '
                     f'stroke-width="3"/></svg>{u} user{"s" if u != 1 else ""}</span>')
    if len(labels) > 1:
        for ri, lab in enumerate(labels):
            dash = "" if ri == 0 else ' stroke-dasharray="6 4"'
            items.append(f'<span><svg width="22" height="10"><line x1="0" x2="22" y1="5" y2="5" stroke="currentColor" '
                         f'stroke-width="2"{dash}/></svg>{_e(lab)} ({"solid, circles" if ri == 0 else "dashed, squares"})'
                         "</span>")
    items.append('<span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="none" stroke="currentColor" '
                 'stroke-width="2"/></svg>hollow = flagged or invalid row (hover for flags)</span>')
    return '<div class="legend">' + "".join(items) + "</div>"


def has_spec_decode(rows: list[RungResult], manifest: RunManifest | None) -> bool:
    if manifest is not None:
        sp = manifest.speculative
        if manifest.k or (sp is not None and getattr(sp, "method", None)):
            return True
    return any(r.k or r.accepted_drafts_per_step is not None or r.hit_rate is not None for r in rows)


def _cell(r: RungResult, col: str) -> str:
    v = getattr(r, col, None)
    if col == "hit_rate" and isinstance(v, (int, float)):
        return f"{v * 100:.1f}%"
    if isinstance(v, float):
        return _fmt(v)
    if isinstance(v, list):
        return ", ".join(map(str, v))
    return "" if v is None else str(v)


def numbers_table(rows: list[RungResult]) -> str:
    head = "".join(f"<th>{_e(c)}</th>" for c in TABLE_COLUMNS)
    body = []
    for r in rows:
        cls = ' class="bad"' if (HOLLOW_FLAGS & set(r.flags)) or r.valid != "valid" else ""
        body.append(f"<tr{cls}>" + "".join(f"<td>{_e(_cell(r, c))}</td>" for c in TABLE_COLUMNS) + "</tr>")
    return f'<div class="wide"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _kv_table(items: list[tuple[str, Any]]) -> str:
    return '<table class="hdr">' + "".join(
        f"<tr><td>{_e(k)}</td><td>{v if isinstance(v, Markup) else _e(v)}</td></tr>" for k, v in items
    ) + "</table>"


class Markup(str):
    """Already-escaped HTML."""


def _header(rows: list[RungResult], manifest: RunManifest | None) -> str:
    out: list[str] = []
    m = manifest
    r0 = rows[0] if rows else None
    if m is None:
        out.append('<p class="note">No manifest.json next to the results: run fingerprint unknown '
                   "(imported or copied rows).</p>")
    fp = (m.fingerprint if m else {}) or {}
    boot = (m.boot if m else {}) or {}
    launch = (m.launch if m else {}) or {}
    att = m.attention if m else None
    graph = m.graph if m else None
    sp = m.speculative if m else None
    hw = m.hardware if m else None
    caps = (graph.capture_sizes if graph and getattr(graph, "capture_sizes", None) else None) or (
        r0.capture_sizes if r0 else None)
    att_txt = "unknown"
    if att is not None:
        att_txt = f"{att.actual} (requested {att.requested}; from boot/runtime log lines, {len(att.evidence_lines)} quoted)"
    elif r0 is not None:
        att_txt = f"{r0.attention_path} (from rows)"
    graphs_txt = "unknown"
    if graph is not None:
        graphs_txt = (f"on={graph.graphs_on} (basis {graph.graphs_basis}; declared {graph.graphs_declared}), "
                      f"mode={graph.mode}, capture sizes={caps}")
    items: list[tuple[str, Any]] = []
    if m:
        items += [
            ("run", f"{m.run_id} — spec {m.spec_name} / profile {m.profile} — status {m.status}"
             + (f" ({m.abort_reason})" if m.abort_reason else "")),
            ("started", f"{m.started_utc} UTC / {m.started_local} ({m.display_timezone or 'local'})"),
            ("finished", f"{m.finished_utc or 'n/a'} UTC / {m.finished_local or 'n/a'}"),
        ]
    items += [
        ("image", f"{fp.get('image_tag') or (m.image if m else None) or 'n/a'}"
         f"{' — digest ' + ', '.join(fp.get('repo_digests') or []) if fp.get('repo_digests') else ''}"
         f"{' — id ' + str(fp.get('image_id'))[:19] if fp.get('image_id') else ''}"),
        ("build fingerprint", fp.get("build_fingerprint") or (r0.build_fingerprint if r0 else None) or "n/a"),
        ("attention path", att_txt),
        ("KV format", f"{boot.get('kv_dtype') or (m.kv_format if m else None) or (r0.kv_format if r0 else None) or 'unknown'}"
         + (f" ({boot.get('kv_tokens')} KV tokens)" if boot.get("kv_tokens") else "")),
        ("graphs", graphs_txt),
        ("speculative decoding", (f"{getattr(sp, 'method', None)} k={m.k if m else None}" if sp else "none declared")
         + (f"; boot log: {boot.get('spec_method')} k={boot.get('spec_k')}" if boot.get("spec_method") else "")),
    ]
    if hw is not None:
        items.append(("hardware", f"host {hw.host}, GPU {hw.gpu_model} x{hw.gpu_count or '?'}, TP {hw.tensor_parallel or '?'}, "
                                  f"host RAM {hw.host_ram_gib or '?'} GiB, MemAvailable floor {hw.mem_available_floor_gib} GiB"))
    out.append(_kv_table(items))
    # Launch: how the server was started (captured read-only at preflight).
    out.append("<h3>Server launch</h3>")
    if launch:
        out.append(f'<p class="muted">Captured {_e(launch.get("captured_utc"))} UTC / '
                   f'{_e(launch.get("captured_local"))} from {_e(launch.get("source"))} (read-only; secrets redacted).</p>')
        out.append(f"<pre>{_e(launch.get('command_line') or '(start command not captured)')}</pre>")
        env = launch.get("env") or {}
        if env:
            out.append("<details><summary>environment</summary><pre>"
                       + _e("\n".join(f"{k}={v}" for k, v in sorted(env.items()))) + "</pre></details>")
        extra = [(k, ", ".join(map(str, launch[k])) if isinstance(launch.get(k), list) else launch.get(k))
                 for k in ("container", "devices", "group_add", "ipc_mode", "shm_size", "port_bindings",
                           "container_started_at") if launch.get(k)]
        if extra:
            out.append(_kv_table(extra))
        if launch.get("vllm_boot_config"):
            out.append("<details><summary>vLLM boot config (log)</summary><pre>"
                       + _e("\n".join(launch["vllm_boot_config"])) + "</pre></details>")
    else:
        out.append('<p class="muted">No launch record (dry run, imported rows, or a run from before launch '
                   "capture).</p>")
    # Caveats
    cav: list[str] = []
    if m and m.status != "completed":
        cav.append(f"run status is {m.status}")
    lc = (m.launch_check if m else {}) or {}
    for mm in lc.get("mismatches", []) or []:
        cav.append(f"NOT_AS_LABELED: {mm}")
    for w in lc.get("warnings", []) or []:
        cav.append(str(w))
    flagged = sorted({f for r in rows for f in r.flags})
    for f in flagged:
        n = sum(f in r.flags for r in rows)
        cav.append(f"{f} on {n} row(s): {FLAGS.get(f, '')}")
    invalid = sum(r.valid != "valid" for r in rows)
    if invalid:
        cav.append(f"{invalid} row(s) not valid")
    if cav:
        out.append("<h3>Caveats</h3><ul>" + "".join(f"<li>{_e(c)}</li>" for c in cav) + "</ul>")
    return "".join(out)


def render_html(
    rows: list[RungResult],
    manifest: RunManifest | None = None,
    title: str | None = None,
    compare: list[tuple[str, list[RungResult]]] | None = None,
) -> str:
    """Whole report. ``compare`` adds more runs, drawn dashed over the first."""
    label0 = (manifest.run_id if manifest else None) or (rows[0].run_id if rows else "run")
    runs = [(label0, rows)] + list(compare or [])
    all_rows = [r for _, rs in runs for r in rs]
    users_order = sorted({r.users for r in all_rows})
    groups = sorted({(r.prompt_type, r.load_mode) for r in all_rows})
    spec_on = any(has_spec_decode(rs, manifest if i == 0 else None) for i, (_, rs) in enumerate(runs))
    ttl = title or f"xpu-llm-bench report — {label0}"
    out = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{_e(ttl)}</title><style>{CSS}</style></head><body>",
        f"<h1>{_e(ttl)}</h1>",
        f'<p class="muted">{len(all_rows)} rows. Charts: X = context per user, one line per concurrency. '
        "Hover a point for the exact value. Self-contained file: no scripts, no network assets.</p>",
        "<h2>Run</h2>",
        _header(rows, manifest),
    ]
    if len(runs) > 1:
        out.append('<p class="note">Compare view: ' + ", ".join(
            f"<b>{_e(lab)}</b> ({'solid' if i == 0 else 'dashed'})" for i, (lab, _) in enumerate(runs)) + ".</p>")
    out.append("<h2>Charts</h2>")
    out.append(_legend(users_order, [lab for lab, _ in runs]))
    if not spec_on:
        out.append('<p class="note">No speculative decoding in this run: the accepted-drafts and hit-rate charts '
                   "are skipped.</p>")
    if not all_rows:
        out.append('<p class="muted">No rows.</p>')
    for pt, lm in groups:
        out.append(f"<h3>{_e(pt)} · {_e(lm)}</h3><div class=\"grid\">")
        for field, mtitle, unit, scale, needs_spec in METRICS:
            if needs_spec and not spec_on:
                continue
            sub = [(lab, [r for r in rs if (r.prompt_type, r.load_mode) == (pt, lm)]) for lab, rs in runs]
            out.append(svg_chart(sub, field, mtitle, unit, scale, users_order))
        out.append("</div>")
    out.append("<h2>All numbers</h2>")
    for lab, rs in runs:
        if len(runs) > 1:
            out.append(f"<h3>{_e(lab)}</h3>")
        out.append(numbers_table(rs))
    flagged = sorted({f for r in all_rows for f in r.flags})
    if flagged:
        out.append("<h2>Flags</h2><ul>" + "".join(
            f"<li><code>{_e(f)}</code>: {_e(FLAGS.get(f, ''))}</li>" for f in flagged) + "</ul>")
    out.append("</body></html>\n")
    return "\n".join(out)
