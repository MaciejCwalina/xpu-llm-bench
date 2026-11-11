from xpu_llm_bench.fingerprint import fingerprint_from_inspect, redact_env
from xpu_llm_bench.hostexec import HostExec
from xpu_llm_bench.procmem import parse_scan, role_of


class P:
    def __init__(self, out, code=0):
        self.returncode, self.stdout, self.stderr = code, out, ""


INSPECT = {
    "Id": "abc123" + "0" * 58,
    "Image": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "Config": {"Image": "example/vllm-xpu:test", "Cmd": ["--port", "8102"], "Env": ["K8V4_QSA=1", "HF_TOKEN=secret"]},
    "State": {"StartedAt": "2026-11-11T16:03:00Z", "OOMKilled": False, "Status": "running"},
    "HostConfig": {"NetworkMode": "host"},
    "Mounts": [
        {"Type": "bind", "Source": "/srv/patches/attn.py", "Destination": "/opt/venv/lib/python3.12/site-packages/vllm/attn.py"},
        {"Type": "bind", "Source": "/models/example", "Destination": "/model"},
    ],
}


def _exec(hashes):
    def runner(cmd, **kw):
        script = cmd[-1]
        for key, h in hashes.items():
            if key in script:
                return P(h + "  -\n")
        return P("MISSING\n")

    return HostExec(runner=runner)


def test_fingerprint_changes_with_patch_hash() -> None:
    a = fingerprint_from_inspect(_exec({"attn.py": "aa" * 32, "/models/example": "bb" * 32}), "vllm-bench-x", INSPECT)
    b = fingerprint_from_inspect(_exec({"attn.py": "cc" * 32, "/models/example": "bb" * 32}), "vllm-bench-x", INSPECT)
    assert a.build_fingerprint != b.build_fingerprint
    assert a.image_id.startswith("sha256:01234567")
    assert a.host_ports == [8102]
    kinds = {m.destination: m.kind for m in a.mounts}
    assert kinds["/model"] == "weights"  # listed by name+size, not content-hashed
    assert "HF_TOKEN=<redacted>" in a.env
    assert any("no RepoDigest" in p for p in a.problems)
    again = fingerprint_from_inspect(_exec({"attn.py": "aa" * 32, "/models/example": "bb" * 32}), "vllm-bench-x", INSPECT)
    assert again.build_fingerprint == a.build_fingerprint


def test_redact_env() -> None:
    assert redact_env(["API_KEY=x", "PATH=/bin"]) == ["API_KEY=<redacted>", "PATH=/bin"]


def test_proc_scan_roles_and_other_stacks() -> None:
    ours = "abc123" + "0" * 58
    other = "def456" + "0" * 58
    text = "\n".join(
        [
            f"101\t2000000\t0::/system.slice/docker-{ours}.scope\tpython3 -m vllm.entrypoints.openai.api_server --port 8102",
            f"102\t3000000\t0::/system.slice/docker-{ours}.scope\tVLLM::EngineCore",
            f"103\t4000000\t0::/system.slice/docker-{ours}.scope\tVLLM::Worker_TP0",
            f"104\t4100000\t0::/system.slice/docker-{ours}.scope\tVLLM::Worker_TP1",
            f"201\t1000000\t0::/system.slice/docker-{other}.scope\tVLLM::EngineCore",
        ]
    )
    snap = parse_scan(text, ours)
    assert snap.ours == {"api_server": 2000000, "engine": 3000000, "worker_tp0": 4000000, "worker_tp1": 4100000}
    assert snap.other_stacks == [other[:12]] and snap.other_stack_resident
    assert role_of("VLLM::Worker_TP1") == "worker_tp1"
    unknown = parse_scan(text, None)
    assert len(unknown.other_stacks) == 2
