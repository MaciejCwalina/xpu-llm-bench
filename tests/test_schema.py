import json
import re
from pathlib import Path

import jsonschema

from xpu_llm_bench.results import HardwareSnapshot, RungResult, RunManifest, stamp_pair

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schemas" / "result.schema.json").read_text()
)


def test_rung_and_manifest_match_schema() -> None:
    utc, et = stamp_pair()
    row = RungResult(
        run_id="x",
        rung_id="code_edits:closed:1u:8192:meas",
        users=1,
        ctx_tokens=[8192],
        ctx_per_user=8192,
        prompt_type="code_edits",
        load_mode="closed_all_decoding",
        started_utc=utc,
        started_local=et,
        finished_utc=utc,
        finished_local=et,
    )
    jsonschema.validate(row.model_dump(), SCHEMA)
    man = RunManifest(
        run_id="x",
        spec_name="s",
        profile="quick",
        started_utc=utc,
        started_local=et,
        hardware=HardwareSnapshot(),
    )
    jsonschema.validate(man.model_dump(), SCHEMA)


def test_stamps_are_offset_aware(monkeypatch) -> None:
    monkeypatch.setenv("XPU_BENCH_TZ", "Asia/Kolkata")
    utc, local = stamp_pair()
    assert utc.endswith("+00:00") or utc.endswith("Z") or "+00:00" in utc or utc.endswith("+0000")
    assert local.endswith("+05:30")
    monkeypatch.delenv("XPU_BENCH_TZ")
    assert re.search(r"[+-]\d\d:\d\d$", stamp_pair()[1])  # system local, still offset-aware


def test_committed_schema_matches_models() -> None:
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    rc = subprocess.run([sys.executable, str(root / "scripts" / "gen_schema.py"), "--check"]).returncode
    assert rc == 0, "schemas/result.schema.json is stale: run python scripts/gen_schema.py"
