"""The build fingerprint never reads model weights, whatever the mount path.

Weight-type files (.safetensors, .bin, ...) and files over 16 MiB are listed by
name + size in one metadata-only find pass. Only small non-weight files
(config / tokenizer json, patch .py) are content-hashed. Mount hashing runs
after validation and never on the per-rung reboot check.
"""

import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from xpu_llm_bench.fingerprint import (
    _hash_script,
    fingerprint_from_inspect,
    hash_fingerprint_mounts,
    hash_mount,
)
from xpu_llm_bench.hostexec import HostExec
from xpu_llm_bench.results import RunManifest
from xpu_llm_bench.runner import Runner

GIB = 1024 ** 3
IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


def _sparse(p: Path, size: int) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        f.truncate(size)


def _weights_dir(root: Path) -> Path:
    """A model dir mounted somewhere that is NOT /model: 1 GiB sparse weights,
    a large non-weight file, small configs, and an HF-cache style blobs/ dir."""
    d = root / "llm-weights"
    _sparse(d / "model-00001-of-00002.safetensors", GIB)
    _sparse(d / "model-00002-of-00002.safetensors", GIB)
    _sparse(d / "consolidated.dat", GIB)  # large, no weight extension
    (d / "small.bin").write_bytes(b"\x00" * 64)  # weight type, even when small
    (d / "config.json").write_text('{"hidden_size": 64}')
    (d / "tokenizer.json").write_text('{"vocab": {}}')
    (d / ".git").mkdir()
    (d / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (d / "__pycache__").mkdir()
    (d / "__pycache__" / "x.pyc").write_bytes(b"pyc")
    hf = d / "hub" / "models--org--m"
    _sparse(hf / "blobs" / "a1b2c3", GIB)
    (hf / "snapshots" / "rev").mkdir(parents=True)
    os.symlink("../../blobs/a1b2c3", hf / "snapshots" / "rev" / "model.safetensors")
    return d


def _lock_big_files(d: Path) -> list[Path]:
    """chmod 000 on every large / weight file: any read of one fails the hash."""
    locked = [p for p in d.rglob("*") if p.is_file() and not p.is_symlink()
              and (p.stat().st_size > 16 * 1024 * 1024 or p.suffix in (".safetensors", ".bin"))]
    for p in locked:
        p.chmod(0)
    return locked


def _digest(src: Path, skip: bool = False) -> str | None:
    t0 = time.time()
    digest, _kind = hash_mount(HostExec(), str(src), skip)
    assert time.time() - t0 < 10, "hashing must not read the weights"
    return digest


def test_weights_anywhere_are_never_read(tmp_path: Path) -> None:
    d = _weights_dir(tmp_path)
    locked = _lock_big_files(d)
    try:
        res = HostExec().sh(_hash_script(str(d), False), timeout=20)
        assert res.returncode == 0, res.stderr
        assert "Permission denied" not in res.stderr  # nothing locked was opened
        a = _digest(d)
        assert a and len(a) == 64
    finally:
        for p in locked:
            p.chmod(0o644)

    # Same size, different bytes in a weight / large file: same digest (never read).
    for name in ("model-00001-of-00002.safetensors", "consolidated.dat"):
        with open(d / name, "r+b") as f:
            f.seek(GIB // 2)
            f.write(b"changed")
    (d / "small.bin").write_bytes(b"\x01" * 64)
    with open(d / "hub" / "models--org--m" / "blobs" / "a1b2c3", "r+b") as f:
        f.write(b"changed")
    (d / ".git" / "HEAD").write_text("ref: refs/heads/other\n")
    (d / "__pycache__" / "x.pyc").write_bytes(b"other")
    assert _digest(d) == a

    # A size change in a weight file, a renamed shard, or a new snapshot link changes it.
    _sparse(d / "model-00002-of-00002.safetensors", GIB + 4096)
    b = _digest(d)
    assert b != a
    os.symlink("../../blobs/a1b2c3", d / "hub" / "models--org--m" / "snapshots" / "rev" / "extra.safetensors")
    c = _digest(d)
    assert c != b
    # Small config / tokenizer json content is hashed.
    (d / "config.json").write_text('{"hidden_size": 128}')
    assert _digest(d) != c


def test_single_file_mounts(tmp_path: Path) -> None:
    big = tmp_path / "weights.gguf"
    _sparse(big, GIB)
    if not IS_ROOT:
        big.chmod(0)
    try:
        g = _digest(big)
        assert g == hashlib.sha256(f"size:{GIB}\n".encode()).hexdigest()
    finally:
        big.chmod(0o644)
    patch = tmp_path / "attn.py"
    patch.write_text("print('patched')\n")
    assert _digest(patch) == hashlib.sha256(patch.read_bytes()).hexdigest()
    assert hash_mount(HostExec(), str(tmp_path / "nope"), False) == (None, "missing")


def test_small_patch_tree_hashes_as_before(tmp_path: Path) -> None:
    """A tree of small source files keeps its previous digest."""
    d = tmp_path / "patch"
    (d / "pkg" / "sub").mkdir(parents=True)
    (d / "pkg" / "a.py").write_text("a = 1\n")
    (d / "pkg" / "sub" / "b c.py").write_text("b = 2\n")
    (d / "README").write_text("x\n")
    (d / "pkg" / "__pycache__").mkdir()
    (d / "pkg" / "__pycache__" / "a.pyc").write_bytes(b"\x00")
    q = shlex.quote(str(d))
    old = (f"cd {q} && find . -type f ! -path '*/__pycache__/*' ! -path './.git/*' -print0 "
           "| LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum")
    want = subprocess.run(["sh", "-c", old], capture_output=True, text=True, check=True).stdout.split()[0]
    assert _digest(d) == want


INSPECT = {
    "Id": "c0ffee",
    "Image": "sha256:" + "0" * 64,
    "Config": {"Image": "example/vllm:test", "Cmd": ["serve"], "Env": []},
    "State": {"Status": "running", "StartedAt": "2026-01-01T00:00:00Z"},
}


def _recording_exec(info: dict, scripts: list[str]) -> HostExec:
    def runner(cmd, **kw):
        if cmd[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0, json.dumps([info]), "")
        if cmd[:2] == ["sh", "-c"]:
            scripts.append(cmd[2])
            if "sha256sum" in cmd[2]:
                return subprocess.CompletedProcess(cmd, 0, "ab" * 32 + "  -\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return HostExec(runner=runner)


def test_inspect_only_then_hash(tmp_path: Path) -> None:
    info = {**INSPECT, "Mounts": [
        {"Type": "bind", "Source": "/srv/llm", "Destination": "/llm"},
        {"Type": "bind", "Source": "/srv/patch.py", "Destination": "/app/patch.py"},
    ]}
    scripts: list[str] = []
    ex = _recording_exec(info, scripts)
    fp = fingerprint_from_inspect(ex, "vllm-bench-x", info, hash_mounts=False)
    assert scripts == [] and fp.build_fingerprint is None
    assert {m.kind for m in fp.mounts} == {"unhashed"} and fp.launch_fingerprint
    hash_fingerprint_mounts(ex, fp)
    assert len(scripts) == 2 and fp.build_fingerprint
    full = fingerprint_from_inspect(_recording_exec(info, []), "vllm-bench-x", info)
    assert full.build_fingerprint == fp.build_fingerprint


def _spec(tmp_path: Path, base: str) -> Path:
    spec = {
        "name": "fp",
        "profile": "custom",
        "endpoint": {"base_url": f"{base}/v1", "metrics_url": f"{base}/metrics", "health_url": f"{base}/health",
                     "model": "m"},
        "server": {"container_name": "vllm-bench-mock"},
        "hardware": {"gpu_count": 1, "gpu_mem_gib": 1},
        "prompts": {"types": ["multi_turn_chat"], "seed": 3},
        "quality": {"enabled": False},
        "ladder": {"context_rungs_tokens": [256, 512], "concurrency_steps": [1], "output_tokens_per_rung": 16,
                   "min_window_steps": 2, "metrics_poll_s": 0.05},
    }
    p = tmp_path / "fp.yaml"
    p.write_text(yaml.safe_dump(spec))
    return p


@pytest.mark.parametrize("status", ["running", "exited"])
def test_runner_hashes_mounts_once_after_validation(mock_server, tmp_path: Path, status: str) -> None:
    from xpu_llm_bench.runner import RunRefused
    from xpu_llm_bench.spec import load_spec

    info = {**INSPECT, "State": {**INSPECT["State"], "Status": status},
            "Mounts": [{"Type": "bind", "Source": "/srv/llm", "Destination": "/llm"}]}
    scripts: list[str] = []
    spec = load_spec(_spec(tmp_path, mock_server["base"]))
    r = Runner(spec, tmp_path / "out", host_exec=_recording_exec(info, scripts), fetch_logs=lambda: "",
               host_fn=lambda: {"gpus": []}, ps_fn=lambda: "", no_site_config=True, assume_yes=True,
               proc_fn=lambda: __import__("xpu_llm_bench.procmem", fromlist=["x"]).ProcSnapshot())
    hash_scripts = lambda: [s for s in scripts if "sha256sum" in s]  # noqa: E731
    if status == "exited":
        with pytest.raises(RunRefused):
            r.run()
        assert hash_scripts() == []  # refused before any mount was hashed
        return
    man = r.run()
    assert man.status == "completed"
    # One hash per mount for the whole run: the per-rung reboot check is inspect-only.
    assert len(hash_scripts()) == 1
    m = RunManifest.load(Path(man.output_jsonl).parent / "manifest.json")
    assert m.fingerprint["build_fingerprint"] and m.fingerprint["mounts"][0]["kind"] == "path"
