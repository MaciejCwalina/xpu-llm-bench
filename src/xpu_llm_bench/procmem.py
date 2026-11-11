"""RSS of the vLLM API server, engine core and each worker, read from /proc.

Read-only. Works locally or over SSH through HostExec. A vLLM process whose
cgroup is not the benchmark container's is reported as another resident stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from xpu_llm_bench.hostexec import HostExec

# pid \t rss_kib \t cgroup-line \t cmdline (NUL -> space)
PROC_SCAN = r"""
for d in /proc/[0-9]*; do
  [ "${d#/proc/}" = "$$" ] && continue
  c=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
  case "$c" in *"for d in /proc"*) continue;; esac
  case "$c" in *VLLM::*|*vllm\ serve*|*vllm.entrypoints*|*vllm/entrypoints*) ;; *) continue;; esac
  r=$(awk '/^VmRSS:/{print $2}' "$d/status" 2>/dev/null)
  g=$(head -n1 "$d/cgroup" 2>/dev/null)
  printf '%s\t%s\t%s\t%s\n' "${d#/proc/}" "${r:-0}" "$g" "$c"
done
"""

_CID = re.compile(r"(?:docker-|/docker/)([0-9a-f]{12,64})")
_WORKER = re.compile(r"VLLM::(Worker_TP\d+|Worker\S*)")


@dataclass
class VllmProc:
    pid: int
    rss_kib: int
    container_id: str | None
    role: str
    cmd: str


@dataclass
class ProcSnapshot:
    procs: list[VllmProc] = field(default_factory=list)
    ours: dict[str, int] = field(default_factory=dict)  # role -> rss_kib
    other_stacks: list[str] = field(default_factory=list)  # container ids (or "host")
    error: str | None = None

    @property
    def other_stack_resident(self) -> bool:
        return bool(self.other_stacks)


def role_of(cmd: str) -> str:
    if "VLLM::EngineCore" in cmd:
        return "engine"
    m = _WORKER.search(cmd)
    if m:
        return m.group(1).lower()
    if "VLLM::" in cmd:
        return "vllm_other"
    return "api_server"


def parse_scan(text: str, our_container_id: str | None) -> ProcSnapshot:
    snap = ProcSnapshot()
    others: set[str] = set()
    for line in text.splitlines():
        parts = line.split("\t", 3)
        if len(parts) < 4:
            continue
        pid_s, rss_s, cg, cmd = parts
        try:
            pid, rss = int(pid_s), int(rss_s or 0)
        except ValueError:
            continue
        m = _CID.search(cg)
        cid = m.group(1) if m else None
        proc = VllmProc(pid, rss, cid, role_of(cmd), cmd.strip()[:200])
        snap.procs.append(proc)
        if our_container_id is None:
            mine = True
        else:
            mine = bool(cid and our_container_id.startswith(cid[:12]))
        if mine:
            snap.ours[proc.role] = snap.ours.get(proc.role, 0) + rss
        else:
            others.add(cid[:12] if cid else "host")
    if our_container_id is None:
        groups = {(p.container_id or "host")[:12] for p in snap.procs}
        others = groups if len(groups) > 1 else set()
    snap.other_stacks = sorted(others)
    return snap


def collect(exec_: HostExec, our_container_id: str | None) -> ProcSnapshot:
    res = exec_.sh(PROC_SCAN, timeout=30)
    if res.returncode != 0 and not res.stdout:
        return ProcSnapshot(error=res.stderr.strip()[:200] or f"exit {res.returncode}")
    return parse_scan(res.stdout, our_container_id)


def peak_merge(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = max(out.get(k, 0), v)
    return out
