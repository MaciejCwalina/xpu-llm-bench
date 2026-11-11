"""Read-only capture of how the server under test was launched.

At preflight the runner records the server's start command and settings so
every result directory describes itself: ``launch.json``, a header at the top
of ``run.log``, ``manifest.launch`` and the HTML report header.

Sources, all read-only (nothing is started, stopped or restarted):
- ``docker inspect`` of the test container: entrypoint, cmd, args, env,
  image + digest, devices, ports, IPC / shm, restart policy;
- the server process command line (``ps``) when there is no container;
- vLLM's own boot-log lines ("non-default args: ..." and the engine
  "with config: ..." line), when present.

Secret-looking values (keys, tokens, passwords, ...) are redacted in env,
command arguments and log lines before anything is written.
"""

from __future__ import annotations

import re
import shlex
from typing import Any

from xpu_llm_bench.results import local_tz_name, stamp_pair

REDACTED = "<redacted>"
# A name is secret when one of its "_"/"-"/"." separated parts is one of these
# (HF_TOKEN, OPENAI_API_KEY, --api-key, db.password, ...). Plural/compound
# parts such as num_speculative_tokens, max_tokens or tokenizer are not.
SECRET_NAME_PARTS = {
    "TOKEN", "KEY", "APIKEY", "SECRET", "SECRETS", "PASSWORD", "PASSWD", "PASS", "PWD", "AUTH", "CREDENTIAL",
    "CREDENTIALS", "COOKIE", "BEARER", "PRIVATE", "SIGNATURE",
}
# Token-shaped values redacted wherever they appear.
_TOKEN_VALUE_RE = re.compile(
    r"\b(?:hf_[A-Za-z0-9]{16,}|sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,})"
)
# key=value / key: value / 'key': 'value' pairs in log lines whose key looks secret.
_KV_SECRET_RE = re.compile(
    r"""(?P<key>['"]?[A-Za-z0-9_.-]*(?:token|key|secret|password|passwd|auth|credential|cookie|bearer)[A-Za-z0-9_.-]*['"]?)"""
    r"""(?P<sep>\s*[=:]\s*)(?P<val>'[^']*'|"[^"]*"|[^\s,})\]]+)""",
    re.IGNORECASE,
)
BOOT_LINE_PATTERNS = (
    re.compile(r"non-default args", re.IGNORECASE),
    re.compile(r"Initializing a V\d+ LLM engine .* with config:", re.IGNORECASE),
    re.compile(r"\bvLLM API server version\b", re.IGNORECASE),
)
SERVER_PROC_RE = re.compile(r"(vllm\s+serve\b|vllm\.entrypoints|api_server\.py)")


def is_secret_name(name: str) -> bool:
    parts = re.split(r"[_\-.\s]+", name.strip("'\"-").upper())
    return any(part in SECRET_NAME_PARTS for part in parts if part)


def redact_text(text: str) -> str:
    """Redact token-shaped values and secret-named key/value pairs in free text."""

    def _kv(m: re.Match[str]) -> str:
        key = m.group("key")
        if not is_secret_name(key):
            return m.group(0)
        val = m.group("val")
        if val in ("None", "null", "''", '""'):
            return m.group(0)
        q = val[0] if val[:1] in "'\"" else ""
        return f"{key}{m.group('sep')}{q}{REDACTED}{q}"

    return _TOKEN_VALUE_RE.sub(REDACTED, _KV_SECRET_RE.sub(_kv, text))


def redact_env(env: list[str] | dict[str, str]) -> dict[str, str]:
    items = env.items() if isinstance(env, dict) else (e.partition("=")[::2] for e in env)
    out: dict[str, str] = {}
    for k, v in items:
        out[k] = REDACTED if is_secret_name(k) and v else redact_text(v)
    return out


def redact_argv(argv: list[str]) -> list[str]:
    """Redact values of secret-looking flags (``--api-key X`` and ``--api-key=X``)."""
    out: list[str] = []
    hide_next = False
    for tok in argv:
        if hide_next:
            out.append(REDACTED)
            hide_next = False
            continue
        if tok.startswith("-") and "=" in tok:
            flag, _, val = tok.partition("=")
            out.append(f"{flag}={REDACTED}" if is_secret_name(flag) and val else redact_text(tok))
            continue
        if tok.startswith("--") and is_secret_name(tok):
            hide_next = True
            out.append(tok)
            continue
        out.append(redact_text(tok))
    return out


def boot_config_lines(log_text: str | None, limit: int = 6, max_len: int = 6000) -> list[str]:
    """vLLM's own record of its settings, from the boot log (redacted)."""
    if not log_text:
        return []
    found: list[str] = []
    for line in log_text.splitlines():
        if any(p.search(line) for p in BOOT_LINE_PATTERNS):
            line = redact_text(line.strip())
            found.append(line if len(line) <= max_len else line[:max_len] + " ...[truncated]")
    # Keep the latest boot's lines (a container log can hold several boots).
    return found[-limit:]


def _devices(info: dict[str, Any]) -> list[str]:
    host = info.get("HostConfig") or {}
    devs = [f"{d.get('PathOnHost')}:{d.get('PathInContainer')}" for d in host.get("Devices") or []]
    for req in host.get("DeviceRequests") or []:
        devs.append(f"request driver={req.get('Driver') or ''} count={req.get('Count')} caps={req.get('Capabilities')}")
    return devs


def launch_from_inspect(info: dict[str, Any], container: str | None = None) -> dict[str, Any]:
    """Launch settings from one ``docker inspect`` object (already parsed)."""
    cfg = info.get("Config") or {}
    host = info.get("HostConfig") or {}
    entry = list(cfg.get("Entrypoint") or [])
    cmd = list(cfg.get("Cmd") or [])
    argv = redact_argv(entry + cmd)
    ports = []
    for cport, binds in sorted((host.get("PortBindings") or {}).items()):
        for b in binds or []:
            ports.append(f"{b.get('HostIp') or ''}:{b.get('HostPort')}->{cport}".lstrip(":"))
    return {
        "source": "docker_inspect",
        "container": container or str(info.get("Name") or "").lstrip("/") or None,
        "image": cfg.get("Image"),
        "image_id": info.get("Image"),
        "repo_digests": list(info.get("RepoDigests") or []),
        "entrypoint": redact_argv(entry),
        "cmd": redact_argv(cmd),
        "args": redact_argv(list(info.get("Args") or [])),
        "command_line": shlex.join(argv) if argv else None,
        "env": redact_env(list(cfg.get("Env") or [])),
        "devices": _devices(info),
        "group_add": list(host.get("GroupAdd") or []),
        "ipc_mode": host.get("IpcMode"),
        "shm_size": host.get("ShmSize"),
        "network_mode": host.get("NetworkMode"),
        "port_bindings": ports,
        "restart_policy": (host.get("RestartPolicy") or {}).get("Name"),
        "container_started_at": (info.get("State") or {}).get("StartedAt"),
    }


def find_server_process(ps_output: str, port: int | None = None) -> list[str] | None:
    """Pick the vLLM server command line from ``ps -eo pid=,args=`` output."""
    best: list[str] | None = None
    for line in ps_output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not SERVER_PROC_RE.search(parts[1]):
            continue
        try:
            argv = shlex.split(parts[1])
        except ValueError:
            argv = parts[1].split()
        if port is not None and str(port) in argv + [a.partition("=")[2] for a in argv]:
            return argv
        best = best or argv
    return best


def launch_from_process(argv: list[str]) -> dict[str, Any]:
    red = redact_argv(argv)
    return {"source": "process_cmdline", "cmd": red, "command_line": shlex.join(red)}


def capture(
    *,
    inspect_info: dict[str, Any] | None = None,
    container: str | None = None,
    process_argv: list[str] | None = None,
    boot_log: str | None = None,
    endpoint: str | None = None,
    problems: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the launch record. Never raises on missing pieces."""
    utc, local = stamp_pair()
    if inspect_info:
        rec = launch_from_inspect(inspect_info, container)
    elif process_argv:
        rec = launch_from_process(process_argv)
    else:
        rec = {"source": "none", "command_line": None}
    rec = {
        "captured_utc": utc,
        "captured_local": local,
        "display_timezone": local_tz_name(),
        "endpoint": endpoint,
        **rec,
        "vllm_boot_config": boot_config_lines(boot_log),
        "problems": list(problems or []),
        "read_only": True,
    }
    return rec


def header_text(launch: dict[str, Any], title: str = "") -> str:
    """Human-readable header for the top of run.log."""
    bar = "=" * 78
    lines = [bar, f"xpu-llm-bench {title}".rstrip(), bar]
    lines.append(f"captured (UTC):   {launch.get('captured_utc')}")
    lines.append(f"captured ({launch.get('display_timezone') or 'local'}): {launch.get('captured_local')}")
    lines.append(f"source:           {launch.get('source')} (read-only)")
    for key in ("endpoint", "container", "image", "image_id", "container_started_at", "ipc_mode", "shm_size",
                "restart_policy"):
        if launch.get(key) not in (None, "", []):
            lines.append(f"{key + ':':<18}{launch[key]}")
    for key in ("repo_digests", "devices", "group_add", "port_bindings"):
        if launch.get(key):
            lines.append(f"{key + ':':<18}{', '.join(map(str, launch[key]))}")
    lines.append("")
    lines.append("start command:")
    lines.append(f"  {launch.get('command_line') or '(not captured)'}")
    env = launch.get("env") or {}
    if env:
        lines.append("")
        lines.append("environment (secret-looking values redacted):")
        lines += [f"  {k}={v}" for k, v in sorted(env.items())]
    boot = launch.get("vllm_boot_config") or []
    lines.append("")
    lines.append("vLLM boot config (from the server's log):")
    lines += [f"  {b}" for b in boot] or ["  (no 'non-default args' / engine config line found)"]
    for p in launch.get("problems") or []:
        lines.append(f"problem: {p}")
    lines.append(bar)
    return "\n".join(lines) + "\n"
