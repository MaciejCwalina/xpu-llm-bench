"""Fixed, seeded prompt corpora for code edits, long-form writing, and multi-turn chat."""

from __future__ import annotations

import random
from dataclasses import dataclass

from xpu_llm_bench.spec import PromptType

SYLLABLES = [
    "ka", "lo", "mi", "ra", "ve", "tu", "sa", "no", "pe", "di",
    "gor", "lan", "mir", "tes", "bar", "fon", "qui", "zel", "han", "dor",
]


def _word(rng: random.Random, n: tuple[int, int] = (2, 3)) -> str:
    parts = [rng.choice(SYLLABLES) for _ in range(rng.randint(*n))]
    return "".join(parts)


def _code_block(rng: random.Random, idx: int) -> str:
    fn = _word(rng)
    typ = rng.choice(["int", "str", "float", "bool", "bytes"])
    return (
        f"# edit {idx:05d}: refactor {_word(rng)}::{fn}\n"
        f"def {fn}_{idx}(x: {typ}, y: {typ}) -> {typ}:\n"
        f"    \"\"\"Handle { _word(rng) } for ticket {rng.randint(1000, 9999)}.\"\"\"\n"
        f"    acc = x\n"
        f"    for i in range({rng.randint(2, 8)}):\n"
        f"        acc = acc + y  # TODO: { _word(rng) } overflow?\n"
        f"    if acc == y:\n"
        f"        return acc\n"
        f"    return acc\n\n"
    )


def _prose_block(rng: random.Random, idx: int) -> str:
    return (
        f"Paragraph {idx:05d}. In the {_word(rng)} valley, operator "
        f"{_word(rng).capitalize()} recorded a {rng.randint(1, 999)}.{rng.randint(0, 99):02d} "
        f"reading at depth {rng.randint(1, 9999)} m while the {_word(rng)} instrument "
        f"logged {_word(rng)} {_word(rng)} {_word(rng)}. "
        f"The shift notes mention {_word(rng)}, {_word(rng)}, and a delayed "
        f"{_word(rng)} hand-off.\n\n"
    )


def _chat_turn(rng: random.Random, idx: int) -> str:
    return (
        f"Turn {idx:05d}. User: Can you check the {_word(rng)} setting on unit "
        f"{_word(rng)}-{_word(rng)} and summarize whether the last "
        f"{rng.randint(2, 12)} readings stayed inside range?\n"
        f"Assistant: The last readings on that unit were within the "
        f"{_word(rng)} band, with one brief spike to {rng.randint(10, 99)} "
        f"that cleared after a {_word(rng)} reset.\n"
    )


_GENERATORS = {
    "code_edits": _code_block,
    "long_form": _prose_block,
    "multi_turn_chat": _chat_turn,
}

# ---------------------------------------------------------------------------
# Realistic generators (default). Fixed vocabularies, real English and real
# Python, combined by a seeded RNG. Not repeated padding: block-level text
# varies, which keeps MTP acceptance closer to real traffic.
# ---------------------------------------------------------------------------

_NOUNS = [
    "request", "config", "cache", "session", "token", "record", "invoice", "report",
    "schedule", "payload", "account", "order", "message", "upload", "metric", "batch",
    "profile", "channel", "ticket", "snapshot", "window", "queue", "worker", "index",
]
_VERBS = [
    "parse", "load", "save", "merge", "validate", "normalize", "render", "retry",
    "collect", "resolve", "flush", "update", "compute", "fetch", "filter", "encode",
]
_FIELDS = ["name", "status", "created_at", "owner", "size", "region", "priority", "retries", "timeout", "tags"]


def _ident(rng: random.Random) -> str:
    return f"{rng.choice(_VERBS)}_{rng.choice(_NOUNS)}"


def _code_real(rng: random.Random, idx: int) -> str:
    fn = _ident(rng)
    noun = rng.choice(_NOUNS)
    a, b = rng.sample(_FIELDS, 2)
    n = rng.randint(2, 9)
    kind = rng.randrange(5)
    if kind == 0:
        body = (
            f"def {fn}(items: list[dict], limit: int = {n * 10}) -> list[dict]:\n"
            f'    """Return {noun}s whose {a} is set, newest first, at most `limit`."""\n'
            f"    kept = [it for it in items if it.get(\"{a}\")]\n"
            f"    kept.sort(key=lambda it: it.get(\"{b}\", 0), reverse=True)\n"
            f"    return kept[:limit]\n"
        )
    elif kind == 1:
        body = (
            f"def {fn}(path: str, retries: int = {n}) -> bytes:\n"
            f'    """Read a {noun} file, retrying transient OS errors."""\n'
            f"    last_error = None\n"
            f"    for attempt in range(retries):\n"
            f"        try:\n"
            f"            with open(path, \"rb\") as fh:\n"
            f"                return fh.read()\n"
            f"        except OSError as exc:\n"
            f"            last_error = exc\n"
            f"            time.sleep(0.{n} * (attempt + 1))\n"
            f"    raise RuntimeError(f\"could not read {{path}}\") from last_error\n"
        )
    elif kind == 2:
        cls = "".join(w.capitalize() for w in fn.split("_"))
        body = (
            f"@dataclass\n"
            f"class {cls}:\n"
            f"    {a}: str\n"
            f"    {b}: int = {n}\n"
            f"    tags: list[str] = field(default_factory=list)\n\n"
            f"    def is_stale(self, now: float, max_age_s: float = {n * 60}.0) -> bool:\n"
            f"        return now - float(self.{b}) > max_age_s\n"
        )
    elif kind == 3:
        body = (
            f"def {fn}(counts: dict[str, int]) -> dict[str, float]:\n"
            f'    """Turn raw {noun} counts into shares that sum to 1.0."""\n'
            f"    total = sum(counts.values())\n"
            f"    if total == 0:\n"
            f"        return {{k: 0.0 for k in counts}}\n"
            f"    return {{k: v / total for k, v in sorted(counts.items())}}\n"
        )
    else:
        body = (
            f"async def {fn}(client, ids: list[int], concurrency: int = {n}) -> dict[int, dict]:\n"
            f'    """Fetch {noun}s in parallel with a bounded semaphore."""\n'
            f"    sem = asyncio.Semaphore(concurrency)\n"
            f"    results: dict[int, dict] = {{}}\n\n"
            f"    async def one(i: int) -> None:\n"
            f"        async with sem:\n"
            f"            results[i] = await client.get(f\"/{noun}s/{{i}}\")\n\n"
            f"    await asyncio.gather(*(one(i) for i in ids))\n"
            f"    return results\n"
        )
    return f"# file: src/{noun}_{idx % 17}.py  (block {idx})\n{body}\n"


_SUBJECTS = [
    "The maintenance team", "Our quarterly review", "The city council", "A small group of volunteers",
    "The hospital's night shift", "The research vessel's crew", "The regional rail operator",
    "The school board", "The orchestra", "The warehouse manager", "The field geologist",
]
_ACTIONS = [
    "spent most of the week reorganizing", "argued for hours about", "finally published",
    "quietly postponed", "made real progress on", "underestimated the cost of",
    "documented every step of", "asked for more time to finish",
]
_OBJECTS = [
    "the flood-response plan", "the new scheduling system", "the library renovation",
    "the winter supply route", "a backlog of unanswered letters", "the bridge inspection",
    "the volunteer training program", "the archive of old survey maps", "the budget for next spring",
]
_DETAILS = [
    "Nobody expected the weather to matter as much as it did.",
    "Several people noted that the original estimate had been written in a hurry.",
    "The minutes record a long pause before anyone answered.",
    "In hindsight, the simplest option would have been to wait another month.",
    "A follow-up meeting was scheduled, though no one agreed on the agenda.",
    "The final numbers were better than the draft, but only slightly.",
    "One reviewer asked, reasonably, who would maintain it after the grant ended.",
    "The photographs from that afternoon are still pinned to the corridor wall.",
]


def _prose_real(rng: random.Random, idx: int) -> str:
    sents = []
    for _ in range(rng.randint(3, 5)):
        sents.append(
            f"{rng.choice(_SUBJECTS)} {rng.choice(_ACTIONS)} {rng.choice(_OBJECTS)}"
            f" in {rng.choice(['March', 'June', 'September', 'November'])} {rng.randint(1990, 2025)}."
        )
        sents.append(rng.choice(_DETAILS))
    return f"Paragraph {idx}. " + " ".join(sents) + "\n\n"


_USER_ASKS = [
    "my {n} keeps timing out when I export the {n2} — is that a known issue?",
    "can you explain why the {n} total on my dashboard doesn't match the invoice?",
    "I changed the {n} setting yesterday and now the {n2} page loads slowly.",
    "how do I move a {n} from one account to another without losing history?",
    "the mobile app shows an old {n} even after I refresh. What should I try?",
]
_AGENT_REPLIES = [
    "Thanks for the detail. That usually happens when the {n} is still syncing; it can take up to ten minutes.",
    "I checked your account and the {n} was updated, but the {n2} cache had not expired yet. I've cleared it.",
    "That is expected behavior for large {n}s. Splitting the export by month avoids the timeout.",
    "Good question. The totals differ because the {n2} includes tax while the {n} summary does not.",
    "Please sign out and back in once; the {n} list is rebuilt at login.",
]


def _chat_real(rng: random.Random, idx: int) -> str:
    n, n2 = rng.sample(_NOUNS, 2)
    return (
        f"User: {rng.choice(_USER_ASKS).format(n=n, n2=n2)}\n"
        f"Assistant: {rng.choice(_AGENT_REPLIES).format(n=n, n2=n2)}\n\n"
    )


_REAL_GENERATORS = {
    "code_edits": _code_real,
    "long_form": _prose_real,
    "multi_turn_chat": _chat_real,
}


def needle_for(seed: int, segment: int) -> tuple[str, str, str]:
    """(fact sentence, question, expected answer) for one growth segment."""
    rng = random.Random(seed * 7919 + segment * 104729)
    locker = rng.randint(100, 999)
    code = f"{rng.choice(['amber', 'cobalt', 'violet', 'russet', 'teal'])}-{rng.randint(1000, 9999)}"
    fact = f"(Note for later: the access code for storage locker {locker} is {code}.)"
    question = (
        f"What is the access code for storage locker {locker}, mentioned earlier in this "
        "conversation? Reply with the code only."
    )
    return fact, question, code

_OPENERS = {
    "code_edits": (
        "You are reviewing a large code patch. The following is background source "
        "you will be asked to edit at the end. Keep the existing style.\n\n"
    ),
    "long_form": (
        "The following is a long instrument log and field journal. Read it as "
        "background; you will be asked to write a continuation at the end.\n\n"
    ),
    "multi_turn_chat": (
        "This is a multi-turn operations chat transcript. Use it as conversation "
        "history; a new user question follows at the end.\n\n"
    ),
}

_ASKS = {
    "code_edits": (
        "\n\nTask: Propose a focused refactor of the last function. Write real code, "
        "keep names consistent, and explain the change in a short comment."
    ),
    "long_form": (
        "\n\nTask: Write a detailed continuation of the journal (several paragraphs) "
        "that stays consistent with the units, operators, and readings above."
    ),
    "multi_turn_chat": (
        "\n\nUser: Please recap the open issues from this thread and recommend the "
        "next diagnostic step in a concise, actionable reply."
    ),
}


@dataclass
class PromptCorpus:
    prompt_type: PromptType
    seed: int
    chars_per_token: float = 4.0
    style: str = "realistic"
    file_text: str | None = None  # optional real corpus (from prompts.corpus_dir)

    def block(self, rng: random.Random, idx: int) -> str:
        if self.file_text:
            # Deterministic window into the real corpus.
            span = 1200
            start = rng.randrange(max(len(self.file_text) - span, 1))
            return self.file_text[start:start + span] + "\n\n"
        if self.style == "synthetic" or (self.style == "mixed" and idx % 2):
            return _GENERATORS[self.prompt_type](rng, idx)
        return _REAL_GENERATORS[self.prompt_type](rng, idx)

    def segment(self, n_chars: int, segment_idx: int, needle: str | None = None) -> str:
        """Fresh text for one growth segment (appended as a new user turn)."""
        rng = random.Random(self.seed * 1_000_003 + segment_idx * 7_919)
        parts: list[str] = []
        size = 0
        idx = segment_idx * 100_000
        planted = needle is None
        while size < n_chars or not planted:
            if not planted and size >= n_chars // 2:
                parts.append(needle + "\n\n")  # type: ignore[operator]
                size += len(needle) + 2  # type: ignore[arg-type]
                planted = True
                continue
            blk = self.block(rng, idx)
            parts.append(blk)
            size += len(blk)
            idx += 1
            if idx - segment_idx * 100_000 > 1_000_000:
                raise RuntimeError("segment filler failed to reach target length")
        return "".join(parts)

    def opener(self) -> str:
        return _OPENERS[self.prompt_type]

    def ask(self) -> str:
        return _ASKS[self.prompt_type]

    def fill(self, target_tokens: int, already: str = "") -> str:
        """Append deterministic blocks until the text is about target_tokens long.

        `already` is the current conversation prefix (without the closing ask).
        """
        rng = random.Random(self.seed + 17 * target_tokens + 3 * len(already))
        text = already if already else self.opener()
        want_chars = max(int(target_tokens * self.chars_per_token) - len(self.ask()), 0)
        idx = 0
        # Skip blocks already "accounted for" by using the existing length.
        while len(text) < want_chars:
            text += self.block(rng, idx)
            idx += 1
            if idx > 1_000_000:
                raise RuntimeError("prompt filler failed to reach target length")
        return text

    def user_message(self, target_tokens: int, already: str = "") -> str:
        return self.fill(target_tokens, already=already) + self.ask()


def corpus(
    prompt_type: PromptType,
    seed: int,
    chars_per_token: float = 4.0,
    style: str = "realistic",
    file_text: str | None = None,
) -> PromptCorpus:
    if prompt_type not in _GENERATORS:
        raise ValueError(f"Unknown prompt type {prompt_type!r}")
    return PromptCorpus(
        prompt_type, seed=seed, chars_per_token=chars_per_token, style=style, file_text=file_text
    )


def load_corpus_dir(corpus_dir: str, prompt_type: str) -> tuple[str | None, list[dict]]:
    """Concatenate <corpus_dir>/<prompt_type>/* in sorted order; return (text, manifest)."""
    import hashlib
    from pathlib import Path

    root = Path(corpus_dir).expanduser() / prompt_type
    if not root.is_dir():
        return None, []
    texts: list[str] = []
    manifest: list[dict] = []
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        raw = f.read_bytes()
        manifest.append({"path": str(f.relative_to(root)), "sha256": hashlib.sha256(raw).hexdigest()})
        texts.append(raw.decode("utf-8", errors="replace"))
    return ("\n\n".join(texts) or None), manifest


def session_seed(base_seed: int, family_index: int, sid: int) -> int:
    """Per-session seed: fixed function of the spec seed. Never time-based."""
    return base_seed * 1000 + family_index * 100 + sid

