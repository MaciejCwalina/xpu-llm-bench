"""Regenerate schemas/result.schema.json from the Pydantic models.

    python scripts/gen_schema.py          # write
    python scripts/gen_schema.py --check  # exit 1 if the committed file is stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pydantic.json_schema import models_json_schema  # noqa: E402

from xpu_llm_bench.results import RungResult, RunManifest  # noqa: E402

OUT = ROOT / "schemas" / "result.schema.json"


def build() -> dict:
    _, defs = models_json_schema(
        [(RungResult, "validation"), (RunManifest, "validation")],
        ref_template="#/$defs/{model}",
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:xpu-llm-bench:schemas:result",
        "title": "xpu-llm-bench results (generated from xpu_llm_bench.results; do not hand-edit)",
        "anyOf": [{"$ref": "#/$defs/RungResult"}, {"$ref": "#/$defs/RunManifest"}],
        "$defs": defs["$defs"],
    }


def render() -> str:
    return json.dumps(build(), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        sys.exit(0 if OUT.read_text(encoding="utf-8") == text else 1)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
