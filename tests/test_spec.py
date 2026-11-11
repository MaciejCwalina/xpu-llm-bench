from pathlib import Path

import pytest

from xpu_llm_bench.spec import RunSpec, apply_variant, load_spec

SPECS = Path(__file__).resolve().parents[1] / "specs"


@pytest.mark.parametrize("name", sorted(p.name for p in SPECS.glob("*.yaml")))
def test_example_specs_load(name: str) -> None:
    spec = load_spec(SPECS / name)
    assert spec.name
    assert spec.hardware.mem_available_floor_gib == 2
    assert spec.safety.never_use_xpu_smi is True
    assert "vllm-production" not in spec.safety.allowed_containers


def test_ladder_must_increase() -> None:
    with pytest.raises(ValueError):
        RunSpec.model_validate(
            {"name": "bad", "ladder": {"context_rungs_tokens": [32768, 8192]}}
        )


def test_quick_profile_shortens_ladder() -> None:
    spec = load_spec(SPECS / "qsa-full-fix-mtp6.yaml", profile="quick")
    assert spec.profile == "quick"
    assert spec.ladder.context_rungs_tokens == [8192, 32768]
    assert spec.ladder.load_modes == ["closed_all_decoding"]
    assert spec.ladder.output_tokens_per_rung == 96  # was 32; 32 tokens is ~8 MTP verify steps (SHORT_WINDOW)
    assert spec.ladder.target_minutes == 10.0


def test_variant_overlay_changes_env() -> None:
    spec = load_spec(SPECS / "qsa-vs-dense.yaml")
    assert len(spec.variants) == 2
    qsa = apply_variant(spec, spec.variants[0])
    dense = apply_variant(spec, spec.variants[1])
    assert qsa.server.env["K8V4_QSA"] == "1"
    assert dense.server.env["K8V4_QSA"] == "0"
    assert qsa.server.attention_requested == "qsa"
    assert dense.server.attention_requested == "dense"
    assert qsa.server.container_name.endswith("-qsa")


def test_api_key_is_env_name_not_secret() -> None:
    spec = load_spec(SPECS / "prod-dense-baseline.yaml")
    assert spec.endpoint.api_key_env == "OPENAI_API_KEY"
    raw = (SPECS / "prod-dense-baseline.yaml").read_text()
    assert "sk-" not in raw
