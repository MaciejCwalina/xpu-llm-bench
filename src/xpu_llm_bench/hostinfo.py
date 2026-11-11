"""Host and GPU facts without xpu-smi: sysfs, /proc, uname, dpkg-query. Read-only."""

from __future__ import annotations

from xpu_llm_bench.hostexec import HostExec

HOST_SCAN = r"""
echo "kernel=$(uname -r)"
awk '/^MemTotal:/{print "mem_total_kib="$2}' /proc/meminfo
awk -F': ' '/^model name/{print "cpu="$2; exit}' /proc/cpuinfo
echo "vcpus=$(nproc)"
for c in /sys/class/drm/card[0-9]*; do
  [ -e "$c/device/vendor" ] || continue
  case "$c" in *-*) continue;; esac
  drv=$(basename "$(readlink -f "$c/device/driver")" 2>/dev/null)
  echo "gpu=$(basename "$c") $(cat "$c/device/vendor") $(cat "$c/device/device") driver=$drv"
done
dpkg-query -W -f='pkg=${Package} ${Version}\n' intel-opencl-icd libze1 libze-intel-gpu1 intel-igc-core-2 intel-level-zero-gpu 2>/dev/null
[ -r /opt/intel/oneapi/compiler/latest/../../setvars.sh ] && ls -d /opt/intel/oneapi/compiler/20* 2>/dev/null | sed 's#.*/#oneapi_compiler=#'
"""


def parse_host_scan(text: str) -> dict:
    out: dict = {"gpus": [], "packages": {}}
    for line in text.splitlines():
        key, _, val = line.partition("=")
        if not _:
            continue
        if key == "gpu":
            out["gpus"].append(val.strip())
        elif key == "pkg":
            name, _, ver = val.partition(" ")
            out["packages"][name] = ver.strip()
        else:
            out[key] = val.strip()
    if "mem_total_kib" in out:
        try:
            out["host_ram_gib"] = round(int(out["mem_total_kib"]) / 1024 / 1024, 2)
        except ValueError:
            pass
    return out


def collect(exec_: HostExec) -> dict:
    res = exec_.sh(HOST_SCAN, timeout=30)
    data = parse_host_scan(res.stdout)
    if res.returncode != 0 and not res.stdout:
        data["error"] = res.stderr.strip()[:200]
    data["source"] = "sysfs/proc/dpkg-query (never xpu-smi)"
    return data
