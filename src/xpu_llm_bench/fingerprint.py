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
        cmd=list(cfg.get("Cmd") or []) + list(info.get("Args") or []),
        env=redact_env(list(cfg.get("Env") or [])),
        oom_killed=state.get("OOMKilled"),
        state_status=state.get("Status"),
    )
    fp.host_ports = sorted(bound_host_ports(info))
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
