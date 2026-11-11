"""Build fingerprint: image ID + sha256 of every mounted patch file, from docker inspect.

All test containers on the bench host may share one local image (no RepoDigest) and differ
only by bind mounts, so the image ID alone does not identify a build.
Everything here is read-only: ``docker inspect`` and ``sha256sum``.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass, field
from typing import Any

from xpu_llm_bench.hostexec import HostExec
from xpu_llm_bench.safety import assert_container_allowed

SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASS", "AUTH")
# Mount destinations that hold weights; hashed by file list + sizes only.
DEFAULT_SKIP_CONTENT = ("/model", "/draft", "/root/.cache", "/data")


@dataclass
class MountHash:
    source: str
    destination: str
    kind: str  # file | dir | weights | missing
    sha256: str | None
    files: int = 0


@dataclass
class Fingerprint:
    container: str
    container_id: str | None
    image_tag: str | None
    image_id: str | None
    repo_digests: list[str]
    started_at: str | None
    mounts: list[MountHash] = field(default_factory=list)
    cmd: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)
    build_fingerprint: str | None = None
    launch_fingerprint: str | None = None
    host_ports: list[int] = field(default_factory=list)
    oom_killed: bool | None = None
    state_status: str | None = None
    problems: list[str] = field(default_factory=list)
    # Redacted start command and settings (xpu_llm_bench.launch); not hashed.
    launch: dict[str, Any] | None = None

    def short(self) -> str | None:
        return self.build_fingerprint[:12] if self.build_fingerprint else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "container": self.container,
            "container_id": self.container_id,
            "image_tag": self.image_tag,
            "image_id": self.image_id,
            "repo_digests": self.repo_digests,
            "started_at": self.started_at,
            "mounts": [m.__dict__ for m in self.mounts],
            "cmd": self.cmd,
            "env": self.env,
            "build_fingerprint": self.build_fingerprint,
            "launch_fingerprint": self.launch_fingerprint,
            "host_ports": self.host_ports,
            "problems": self.problems,
        }


def redact_env(env: list[str]) -> list[str]:
    out = []
    for item in env:
        k, _, v = item.partition("=")
        if any(h in k.upper() for h in SECRET_HINTS):
            out.append(f"{k}=<redacted>")
        else:
            out.append(item)
    return out


def _hash_script(src: str, skip_content: bool) -> str:
    q = shlex.quote(src)
    if skip_content:
        # Weights: name + size listing only (hashing tens of GB is not worth it).
        return (
            f"if [ -d {q} ]; then cd {q} && find . -type f -printf '%P %s\\n' | LC_ALL=C sort | sha256sum; "
            f"elif [ -f {q} ]; then stat -c '%n %s' {q} | sha256sum; else echo MISSING; fi"
        )
    return (
        f"if [ -f {q} ]; then sha256sum < {q}; "
        f"elif [ -d {q} ]; then cd {q} && find . -type f ! -path '*/__pycache__/*' ! -path './.git/*' -print0 "
        f"| LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum; else echo MISSING; fi"
    )


def hash_mount(exec_: HostExec, src: str, skip_content: bool) -> tuple[str | None, str]:
    res = exec_.sh(_hash_script(src, skip_content), timeout=600)
    out = res.stdout.strip()
    if res.returncode != 0 or not out or out == "MISSING":
        return None, "missing"
    return out.split()[0], "weights" if skip_content else "path"


def collect_fingerprint(
    exec_: HostExec,
    container: str,
    skip_content_dest: tuple[str, ...] = DEFAULT_SKIP_CONTENT,
    inspect_json: str | None = None,
) -> Fingerprint:
    assert_container_allowed(container)
    if inspect_json is None:
        res = exec_.run(["docker", "inspect", container], timeout=30)
        if res.returncode != 0:
            fp = Fingerprint(container, None, None, None, [], None)
            fp.problems.append(f"docker inspect failed: {res.stderr.strip()[:200]}")
            return fp
        inspect_json = res.stdout
    data = json.loads(inspect_json)
    info = data[0] if isinstance(data, list) else data
    return fingerprint_from_inspect(exec_, container, info, skip_content_dest)


def fingerprint_from_inspect(
    exec_: HostExec | None,
    container: str,
    info: dict[str, Any],
    skip_content_dest: tuple[str, ...] = DEFAULT_SKIP_CONTENT,
) -> Fingerprint:
    from xpu_llm_bench.launch import launch_from_inspect, redact_argv
    from xpu_llm_bench.safety import bound_host_ports

    cfg = info.get("Config") or {}
    state = info.get("State") or {}
    fp = Fingerprint(
        container=container,
        container_id=info.get("Id"),
        image_tag=cfg.get("Image"),
        image_id=info.get("Image"),
        repo_digests=list(info.get("RepoDigests") or []),
        started_at=state.get("StartedAt"),
        # Secret-looking flag values (--api-key X, --hf-token=X) are redacted.
        cmd=redact_argv(list(cfg.get("Cmd") or []) + list(info.get("Args") or [])),
        env=redact_env(list(cfg.get("Env") or [])),
        oom_killed=state.get("OOMKilled"),
        state_status=state.get("Status"),
    )
    fp.host_ports = sorted(bound_host_ports(info))

    fp.launch = launch_from_inspect(info, container)
    if not fp.repo_digests:
        fp.problems.append("local image: no RepoDigest; image ID + mount hashes identify the build")
    for m in sorted(info.get("Mounts") or [], key=lambda x: x.get("Destination", "")):
        src, dest = m.get("Source", ""), m.get("Destination", "")
        if m.get("Type") not in (None, "bind"):
            fp.mounts.append(MountHash(src, dest, str(m.get("Type")), None))
            continue
        skip = any(dest == d or dest.startswith(d.rstrip("/") + "/") for d in skip_content_dest)
        if exec_ is None:
            fp.mounts.append(MountHash(src, dest, "unhashed", None))
            continue
        digest, kind = hash_mount(exec_, src, skip)
        if digest is None:
            fp.problems.append(f"could not hash mount {src} -> {dest}")
        fp.mounts.append(MountHash(src, dest, kind, digest))
    h = hashlib.sha256()
    h.update((fp.image_id or "no-image-id").encode())
    for mh in fp.mounts:
        h.update(f"\n{mh.destination}\t{mh.kind}\t{mh.sha256}".encode())
    fp.build_fingerprint = h.hexdigest()
    lh = hashlib.sha256()
    lh.update(json.dumps({"cmd": fp.cmd, "env": sorted(fp.env)}, sort_keys=True).encode())
    fp.launch_fingerprint = lh.hexdigest()
    return fp


# ---------------------------------------------------------------------------
# Expected-vs-actual launch check (env / launch_args / image are expectations
# about the container the operator booted; they are verified, never applied).
# ---------------------------------------------------------------------------


def _cmd_tokens(cmd: list[str]) -> list[str]:
    out: list[str] = []
    for item in cmd:
        if " " in item:
            try:
                out.extend(shlex.split(item))
            except ValueError:
                out.extend(item.split())
        else:
            out.append(item)
    return out


def _arg_present(tokens: list[str], flag: str, value: str | None) -> bool:
    if value is None:
        if flag in tokens:
            return True
        if "=" in flag:
            f, _, v = flag.partition("=")
            return any(tokens[i] == f and tokens[i + 1] == v for i in range(len(tokens) - 1))
        return False
    if f"{flag}={value}" in tokens:
        return True
    return any(tokens[i] == flag and tokens[i + 1] == value for i in range(len(tokens) - 1))


def verify_expected(
    fp: Fingerprint,
    env: dict[str, str],
    launch_args: list[str],
    image: str | None = None,
    image_digest: str | None = None,
    is_placeholder=lambda v: False,
) -> list[str]:
    """Mismatches between what the spec says the container runs and docker inspect."""
    problems: list[str] = []
    actual: dict[str, str] = {}
    for item in fp.env:
        k, _, v = item.partition("=")
        actual[k] = v
    for key, want in (env or {}).items():
        if key not in actual:
            problems.append(f"env {key}={want!r} expected, but {key} is not set in container {fp.container}")
        elif actual[key] == "<redacted>":
            continue
        elif actual[key] != str(want):
            problems.append(f"env {key}: expected {want!r}, container has {actual[key]!r}")
    tokens = _cmd_tokens(fp.cmd)
    args = list(launch_args or [])
    i = 0
    while i < len(args):
        tok = args[i]
        val = None
        if tok.startswith("-") and "=" not in tok and i + 1 < len(args) and not args[i + 1].startswith("-"):
            val = args[i + 1]
            i += 1
        if not _arg_present(tokens, tok, val):
            shown = tok if val is None else f"{tok} {val}"
            problems.append(f"launch arg {shown!r} expected, not found in the container command")
        i += 1
    if image and not is_placeholder(image) and fp.image_tag and image != fp.image_tag:
        problems.append(f"image: expected {image!r}, container runs {fp.image_tag!r}")
    if image_digest and not is_placeholder(image_digest):
        known = [fp.image_id or ""] + list(fp.repo_digests)
        if not any(image_digest in d for d in known if d):
            problems.append(f"image digest {image_digest!r} not found in the container's image id / repo digests")
    return problems
