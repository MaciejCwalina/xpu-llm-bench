"""Site configuration: the protected production details of one bench host.

Nothing host-specific lives in the repo. The production container names,
protected service patterns, production ports, the bench-hold command and the
operator's quiet-hold file come from an untracked YAML file:

1. ``$XPU_BENCH_SITE`` (a path; an empty value disables site config), else
2. ``./bench-site.local.yaml`` (gitignored), else
3. ``~/.config/xpu-llm-bench/site.yaml``.

See ``bench-site.example.yaml``. Site values only ever ADD protection: the
protected names, patterns and ports are unioned with the spec and the built-in
neutral defaults; they are never used to remove a guard.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

SITE_ENV = "XPU_BENCH_SITE"
LOCAL_SITE_FILE = "bench-site.local.yaml"
USER_SITE_FILE = "~/.config/xpu-llm-bench/site.yaml"


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


def site_path() -> Path | None:
    env = os.environ.get(SITE_ENV)
    if env is not None:
        return Path(env).expanduser() if env.strip() else None
    for cand in (Path(LOCAL_SITE_FILE), Path(USER_SITE_FILE).expanduser()):
        if cand.is_file():
            return cand
    return None


@lru_cache(maxsize=8)
def _load(path: str | None) -> SiteConfig:
    if not path:
        return SiteConfig()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"site config {p} (from ${SITE_ENV}) does not exist")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"site config must be a mapping: {p}")
    cfg = SiteConfig.model_validate(data)
    cfg.source = str(p)
    return cfg


def load_site() -> SiteConfig:
    p = site_path()
    return _load(str(p) if p else None)


def clear_cache() -> None:
    _load.cache_clear()
