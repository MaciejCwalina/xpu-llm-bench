"""Site configuration: the protected production details of one bench host.

Nothing host-specific lives in the repo. The production container names,
protected service patterns, production ports, the bench-hold command and the
operator's quiet-hold file come from an untracked YAML file:

1. ``$XPU_BENCH_SITE`` (a path; an empty value disables site config), else
2. ``<repo root>/bench-site.local.yaml`` (gitignored; the checkout this
   package was installed from, never the current directory), else
3. ``~/.config/xpu-llm-bench/site.yaml``.

See ``bench-site.example.yaml`` and the README's "Site config" section.
Placeholder values (``<...>``, ``TODO``, ``example``, ``changeme``) are
rejected, and a live ``bench run`` refuses to start without a real site file
unless ``--no-site-config`` acknowledges that only the built-in neutral
protection applies. Site values only ever ADD protection: the
protected names, patterns and ports are unioned with the spec and the built-in
neutral defaults; they are never used to remove a guard.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

SITE_ENV = "XPU_BENCH_SITE"
LOCAL_SITE_FILE = "bench-site.local.yaml"
USER_SITE_FILE = "~/.config/xpu-llm-bench/site.yaml"


class SiteConfigError(ValueError):
    """A site config exists but is unusable (placeholders, missing fields)."""


_PLACEHOLDER = re.compile(r"<[^<>]+>|\bTODO\b|\bchangeme\b|\bexample\b|\bxxx\b", re.IGNORECASE)
REQUIRED_FIELDS = ("host", "protected_containers", "forbidden_ports")


class SiteConfig(BaseModel):
    # Display name of the bench host (also the default for hardware.host).
    host: str | None = None
    # ssh target when access.mode is ssh (e.g. an ~/.ssh/config alias).
    ssh_host: str | None = None
    # Exact container names that must never be touched (production stacks).
    protected_containers: list[str] = Field(default_factory=list)
    # Case-insensitive regexes for protected services (gateways, CI, git hosting).
    protected_patterns: list[str] = Field(default_factory=list)
    # Host ports of production endpoints; endpoints/containers on them are refused.
    forbidden_ports: list[int] = Field(default_factory=list)
    # Path of the bench-hold CLI on the bench host.
    hold_command: str | None = None
    # Operator quiet-hold file (read only). An active quiet hold blocks runs.
    quiet_hold_file: str | None = None
    source: str | None = None


def repo_root() -> Path | None:
    """The source checkout this package lives in (``src/xpu_llm_bench`` -> repo)."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").is_file() else None


def candidate_paths() -> list[Path]:
    out: list[Path] = []
    root = repo_root()
    if root is not None:
        out.append(root / LOCAL_SITE_FILE)
    out.append(Path(USER_SITE_FILE).expanduser())
    return out


def site_path() -> Path | None:
    env = os.environ.get(SITE_ENV)
    if env is not None:
        return Path(env).expanduser() if env.strip() else None
    for cand in candidate_paths():
        if cand.is_file():
            return cand
    return None


def placeholder_problems(data: dict) -> list[str]:
    """Field paths whose values are still template placeholders, plus missing required fields."""
    problems: list[str] = []

    def walk(obj: object, path: str) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                walk(val, f"{path}.{key}" if path else str(key))
        elif isinstance(obj, list):
            for i, val in enumerate(obj):
                walk(val, f"{path}[{i}]")
        elif isinstance(obj, str) and _PLACEHOLDER.search(obj):
            problems.append(f"{path} = {obj!r} is a placeholder")

    walk(data, "")
    for key in REQUIRED_FIELDS:
        if not data.get(key):
            problems.append(f"{key} is missing or empty")
    return problems


@lru_cache(maxsize=8)
def _load(path: str | None) -> SiteConfig:
    if not path:
        return SiteConfig()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"site config {p} (from ${SITE_ENV}) does not exist")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SiteConfigError(f"site config must be a mapping: {p}")
    problems = placeholder_problems(data)
    if problems:
        raise SiteConfigError(
            f"site config {p} is not filled in:\n  - "
            + "\n  - ".join(problems)
            + "\nSee the README section 'Site config' for what each field must hold."
        )
    cfg = SiteConfig.model_validate(data)
    cfg.source = str(p)
    return cfg


def load_site() -> SiteConfig:
    p = site_path()
    return _load(str(p) if p else None)


def describe_site() -> str:
    """One line naming the site file in effect (for plan / run output)."""
    env = os.environ.get(SITE_ENV)
    cfg = load_site()
    if cfg.source:
        return f"site config: {cfg.source}"
    if env is not None and not env.strip():
        return f"site config: disabled (${SITE_ENV} is empty); built-in neutral protection only"
    searched = ", ".join(str(c) for c in candidate_paths())
    return f"site config: NONE FOUND (searched ${SITE_ENV}, {searched}); built-in neutral protection only"


def clear_cache() -> None:
    _load.cache_clear()
