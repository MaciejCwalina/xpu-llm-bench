"""Lightweight quality checks that run in the same pass as the speed ladder.

None of these run inside a measured decode window.

1. Logprob scoring of fixed reference answers (teacher-forced, prompt_logprobs)
   compared to a stored dense K8/V4 baseline within a tolerance.
2. Greedy (temperature 0) output with MTP on vs a stored MTP-off reference:
   the first token where they diverge.
3. Pass/fail tasks: a planted fact (needle) at each context length, a small
   code edit checked by a unit test, and format / repetition-loop checks.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SPECIAL_LEAKS = ("<|im_start|>", "<|endoftext|>", "<|im_end|>", "<think>", "</think>")
EOS_MARKERS = ("<|im_end|>", "<|endoftext|>")


# ---------------------------------------------------------------------------
# Text hygiene
# ---------------------------------------------------------------------------


def cut_at_eos(text: str) -> tuple[str, bool]:
    """Text before the first EOS marker. With ignore_eos the server keeps
    decoding past EOS; anything after it is not part of the answer."""
    cut = len(text)
    for m in EOS_MARKERS:
        i = text.find(m)
        if i != -1:
            cut = min(cut, i)
    return text[:cut], cut < len(text)


def repetition_ratio(text: str, n: int = 6) -> float:
    words = re.findall(r"\S+", text)
    if len(words) < n * 2:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def repeated_lines(text: str, run: int = 3) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    streak = 1
    for a, b in zip(lines, lines[1:], strict=False):
        streak = streak + 1 if a == b else 1
        if streak >= run:
            return True
    return False


@dataclass
class FormatCheck:
    ok: bool
    problems: list[str] = field(default_factory=list)
    repetition: float = 0.0
    eos_seen: bool = False


def format_check(raw_text: str, prompt_type: str, ngram: int = 6, max_ratio: float = 0.35) -> FormatCheck:
    text, eos = cut_at_eos(raw_text)
    problems: list[str] = []
    if not text.strip():
        problems.append("empty answer")
    for leak in SPECIAL_LEAKS:
        if leak in text:
            problems.append(f"leaked {leak}")
    if text:
        printable = sum(ch.isprintable() or ch in "\n\t" for ch in text) / len(text)
        if printable < 0.95:
            problems.append(f"non-printable ratio {1 - printable:.2f}")
    if prompt_type == "multi_turn_chat" and re.search(r"^\s*User:", text, re.MULTILINE):
        problems.append("answer continues as the user (role bleed)")
    rep = repetition_ratio(text, ngram)
    if rep > max_ratio:
        problems.append(f"repetition loop ({ngram}-gram repeat ratio {rep:.2f})")
    if repeated_lines(text):
        problems.append("same line repeated 3+ times")
    return FormatCheck(ok=not problems, problems=problems, repetition=rep, eos_seen=eos)


# ---------------------------------------------------------------------------
# Needle
# ---------------------------------------------------------------------------


def needle_pass(answer: str, expected: str) -> bool:
    text, _ = cut_at_eos(answer)
    return expected.lower() in text.lower()


# ---------------------------------------------------------------------------
# Code edit checked by a unit test
# ---------------------------------------------------------------------------

CODE_TASKS: list[dict[str, str]] = [
    {
        "id": "median",
        "prompt": (
            "This function should return the median of a non-empty list of numbers, "
            "but it is wrong for even-length lists:\n\n```python\n"
            "def median(xs):\n    s = sorted(xs)\n    return s[len(s) // 2]\n```\n\n"
            "Fix it. Reply with only the corrected function in one ```python block."
        ),
        "test": (
            "assert median([3, 1, 2]) == 2\n"
            "assert median([4, 1, 3, 2]) == 2.5\n"
            "assert median([5]) == 5\n"
            "assert median([1.0, 2.0]) == 1.5\n"
        ),
    },
    {
        "id": "chunks",
        "prompt": (
            "This function should split a list into chunks of size n (the last chunk may be "
            "shorter), but it drops the remainder:\n\n```python\n"
            "def chunks(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs) - n + 1, n)]\n```\n\n"
            "Fix it. Reply with only the corrected function in one ```python block."
        ),
        "test": (
            "assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\n"
            "assert chunks([], 3) == []\n"
            "assert chunks([1, 2, 3], 3) == [[1, 2, 3]]\n"
        ),
    },
    {
        "id": "slugify",
        "prompt": (
            "This function should lower-case a title and join words with single hyphens, "
            "dropping punctuation, but it leaves double hyphens and punctuation:\n\n```python\n"
            "def slugify(title):\n    return title.lower().replace(' ', '-')\n```\n\n"
            "Fix it. Reply with only the corrected function in one ```python block."
        ),
        "test": (
            "assert slugify('Hello, World') == 'hello-world'\n"
            "assert slugify('  Many   spaces here ') == 'many-spaces-here'\n"
            "assert slugify('A/B test: v2!') == 'a-b-test-v2'\n"
        ),
    },
]


def code_task_for(seed: int) -> dict[str, str]:
    return CODE_TASKS[seed % len(CODE_TASKS)]


def extract_code(answer: str) -> str | None:
    text, _ = cut_at_eos(answer)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[0]
    if "def " in text:
        return text[text.index("def "):]
    return None


def _limits() -> None:  # pragma: no cover - runs in the child
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    except Exception:
        pass


def run_code_test(code: str, test: str, timeout_s: float = 10.0) -> tuple[bool, str]:
    """Run model code + asserts in an isolated child interpreter (-I, temp dir,
    CPU/memory rlimits, no inherited env). Returns (passed, detail)."""
    with tempfile.TemporaryDirectory(prefix="xpu-llm-bench-code-") as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text("import re, math, string\n" + code + "\n\n" + test, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                preexec_fn=_limits if os.name == "posix" else None,  # noqa: PLW1509
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if proc.returncode == 0:
        return True, "pass"
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["failed"]
    return False, tail[0][:200]


# ---------------------------------------------------------------------------
# Greedy divergence vs a stored reference
# ---------------------------------------------------------------------------


def first_divergence(a: list[Any], b: list[Any]) -> int | None:
    """Index of the first differing element, len(shorter) if one is a prefix, None if equal."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def load_keyed_jsonl(path: str | None, key: str = "probe_id") -> dict[str, dict]:
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            obj = json.loads(line)
            out[str(obj[key])] = obj
    return out


# ---------------------------------------------------------------------------
# Logprob scoring
# ---------------------------------------------------------------------------


@dataclass
class LogprobProbe:
    probe_id: str
    context: str
    answer: str


def logprob_probes(seed: int, ctx_chars: int) -> list[LogprobProbe]:
    """Fixed (context, reference answer) pairs. Deterministic in seed."""
    from xpu_llm_bench.prompts import corpus, needle_for

    probes: list[LogprobProbe] = []
    for i, ptype in enumerate(("long_form", "code_edits", "multi_turn_chat")):
        fact, question, code = needle_for(seed, 900 + i)
        body = corpus(ptype, seed=seed + i).segment(ctx_chars, 900 + i, needle=fact)  # type: ignore[arg-type]
        probes.append(
            LogprobProbe(f"lp-{ptype}-needle", body + "\nQuestion: " + question + "\nAnswer:", " " + code)
        )
    task = CODE_TASKS[0]
    probes.append(
        LogprobProbe(
            "lp-code-median",
            task["prompt"] + "\n",
            "```python\ndef median(xs):\n    s = sorted(xs)\n    n = len(s)\n    mid = n // 2\n"
            "    if n % 2:\n        return s[mid]\n    return (s[mid - 1] + s[mid]) / 2\n```",
        )
    )
    return probes


def mean_answer_logprob(per_pos: list[float | None], n_ctx: int) -> float | None:
    vals = [v for v in per_pos[n_ctx:] if v is not None]
    return sum(vals) / len(vals) if vals else None
