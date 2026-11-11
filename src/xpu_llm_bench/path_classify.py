"""Classify the attention path that actually ran from server log lines."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from xpu_llm_bench.results import AttentionPath
from xpu_llm_bench.spec import LogPatterns


@dataclass
class PathClassification:
    path: AttentionPath
    evidence: list[str] = field(default_factory=list)
    skip_widths: list[str] = field(default_factory=list)
    active_widths: list[str] = field(default_factory=list)


def classify_logs(
    lines: list[str] | str,
    patterns: LogPatterns | None = None,
    requested: str | None = None,
) -> PathClassification:
    """Classify qsa | dense | fallback | mixed | unknown from log evidence.

    Rules (see docs/METHODOLOGY.md):
    - `QSA_GRAPH width=...` means a QSA graph path ran.
    - `QSA_GRAPH skip starts=(...) n=..` means that batch skipped QSA (dense / eager).
    - Batches with n > qsa_graph_max_tokens have no QSA graph path and run dense.
    - If QSA was requested and only skip lines appear, that is a silent fallback.
    - Both active and skip lines in the same window → mixed.
    """
    pats = patterns or LogPatterns()
    active_re = re.compile(pats.qsa_active)
    skip_re = re.compile(pats.qsa_skip)
    if isinstance(lines, str):
        raw_lines = lines.splitlines()
    else:
        raw_lines = list(lines)

    evidence: list[str] = []
    active_widths: list[str] = []
    skip_widths: list[str] = []
    dense_mentions = 0

    for line in raw_lines:
        text = line.rstrip()
        if not text:
            continue
        am = active_re.search(text)
        if am:
            active_widths.append(am.group(1))
            if len(evidence) < 24:
                evidence.append(text)
            continue
        sm = skip_re.search(text)
        if sm:
            skip_widths.append(sm.group(2))
            if len(evidence) < 24:
                evidence.append(text)
            continue
        low = text.lower()
        if "attention" in low and "dense" in low and len(evidence) < 24:
            dense_mentions += 1
            evidence.append(text)

    has_qsa = bool(active_widths)
    has_skip = bool(skip_widths)
    requested_qsa = (requested or "").lower() == "qsa"

    if has_qsa and has_skip:
        path: AttentionPath = "mixed"
    elif has_qsa and not has_skip:
        path = "qsa"
    elif has_skip and not has_qsa:
        path = "fallback" if requested_qsa else "dense"
    elif dense_mentions and requested_qsa:
        path = "fallback"
    elif dense_mentions:
        path = "dense"
    elif (requested or "").lower() == "dense":
        path = "dense"
    else:
        path = "unknown"

    return PathClassification(
        path=path,
        evidence=evidence,
        skip_widths=skip_widths,
        active_widths=active_widths,
    )
