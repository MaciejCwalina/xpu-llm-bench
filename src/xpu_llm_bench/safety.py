"""Hard safety rules for anything that touches containers, GPUs, or host tools."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from urllib.parse import urlsplit

# Built-in, neutral protected container names / patterns. The real production
# names of a bench host come from the untracked site config (see site.py) and
# are added to these; they never replace them.
PROTECTED_CONTAINER_EXACT = frozenset(
    {
        "vllm-production",
    }
)

# Case-insensitive regex guards for services that must never be stopped,
# restarted, or blocked: CI runners. Site config adds gateways etc.
PROTECTED_CONTAINER_PATTERNS = (
    re.compile(r"(^|[-_./])(actions[-_]?runner|github[-_]?runner|gitea[-_]?runner|act[-_]?runner)($|[-_.])", re.IGNORECASE),
    re.compile(r"ci[-_]?runner", re.IGNORECASE),
)

INTENTIONAL_STOP_PATH = "/run/vllm-intentional-stop"
XPU_SMI = "xpu-smi"


class SafetyError(RuntimeError):
    """Raised when an action would violate a standing safety rule."""


def site_protected() -> tuple[frozenset[str], tuple[re.Pattern[str], ...]]:
    from .site import load_site

    site = load_site()
    names = PROTECTED_CONTAINER_EXACT | frozenset(n.strip() for n in site.protected_containers if n.strip())
    pats = PROTECTED_CONTAINER_PATTERNS + tuple(re.compile(p, re.IGNORECASE) for p in site.protected_patterns if p)
    return names, pats


def site_forbidden_ports(extra: Iterable[int] = ()) -> set[int]:
    from .site import load_site

    return {DEFAULT_PRODUCTION_PORT, *load_site().forbidden_ports, *extra}


def is_protected_container(name: str, extra: Iterable[str] = ()) -> bool:
    lowered = name.strip()
    if not lowered:
        return True
    names, pats = site_protected()
    if lowered in names or lowered in {e.strip() for e in extra}:
        return True
    return any(p.search(lowered) for p in pats)


def assert_container_allowed(name: str, allowed: Iterable[str] | None = None) -> str:
    """Refuse protected names, and optionally names not listed in the run spec."""
    name = name.strip()
    if is_protected_container(name):
        raise SafetyError(
            f"Refusing to touch protected container {name!r}. "
            "Production stacks, protected services, and CI runners are off-limits."
        )
    if allowed is not None:
        allowed_set = {n.strip() for n in allowed if n and n.strip()}
        if name not in allowed_set:
            raise SafetyError(
                f"Container {name!r} is not named in the run spec allowed list {sorted(allowed_set)}."
            )
    return name


def command_mentions_xpu_smi(command: Sequence[str] | str) -> bool:
    if isinstance(command, str):
        tokens = command.split()
    else:
        tokens = list(command)
    for tok in tokens:
        base = tok.rsplit("/", 1)[-1]
        if base == XPU_SMI or tok.endswith("/" + XPU_SMI):
            return True
    return False


def assert_no_xpu_smi(command: Sequence[str] | str) -> None:
    if command_mentions_xpu_smi(command):
        raise SafetyError(
            "xpu-smi is forbidden. Collect GPU memory from vLLM metrics, "
            "docker/cgroup stats, or another non-xpu-smi source."
        )


def assert_not_intentional_stop(path: str) -> None:
    if path.rstrip("/") == INTENTIONAL_STOP_PATH:
        raise SafetyError(f"Refusing to touch {INTENTIONAL_STOP_PATH}.")


# Neutral built-in default. The real production port(s) come from site config.
DEFAULT_PRODUCTION_PORT = 8001


def url_port(url: str) -> int | None:
    parts = urlsplit(url)
    if parts.port is not None:
        return parts.port
    if parts.scheme == "http":
        return 80
    if parts.scheme == "https":
        return 443
    return None


def assert_endpoint_allowed(urls: Iterable[str | None], forbidden_ports: Iterable[int]) -> None:
    """Refuse any endpoint on a forbidden (production) port."""
    bad = set(forbidden_ports)
    for url in urls:
        if not url:
            continue
        port = url_port(url)
        if port in bad:
            raise SafetyError(
                f"Refusing endpoint {url!r}: port {port} is a protected production port "
                "(safety.forbidden_ports / site config). Use a test-container port."
            )


def assert_test_container_name(name: str, prefixes: Iterable[str]) -> str:
    """Containers the runner may start or stop must look like test containers."""
    assert_container_allowed(name)
    pre = [p for p in prefixes if p]
    if pre and not any(name.startswith(p) for p in pre):
        raise SafetyError(
            f"Container {name!r} does not carry a test prefix {pre}; refusing to manage it."
        )
    return name


def bound_host_ports(inspect: dict) -> set[int]:
    """Host ports a container publishes or (with host networking) serves on."""
    ports: set[int] = set()
    host_cfg = inspect.get("HostConfig") or {}
    for bindings in (host_cfg.get("PortBindings") or {}).values():
        for b in bindings or []:
            try:
                ports.add(int(b.get("HostPort")))
            except (TypeError, ValueError):
                continue
    if host_cfg.get("NetworkMode") == "host":
        args = list((inspect.get("Config") or {}).get("Cmd") or []) + list(inspect.get("Args") or [])
        for i, a in enumerate(args):
            if a == "--port" and i + 1 < len(args):
                try:
                    ports.add(int(args[i + 1]))
                except ValueError:
                    pass
            elif isinstance(a, str) and a.startswith("--port="):
                try:
                    ports.add(int(a.split("=", 1)[1]))
                except ValueError:
                    pass
    return ports
