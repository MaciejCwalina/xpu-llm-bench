import pytest

from xpu_llm_bench.import_legacy import summed_ratio_correction
from xpu_llm_bench.metrics import ChunkStats, MTPWindow, spec_counter_check, true_hit_rate
from xpu_llm_bench.spec import RunSpec, SpeculativeSettings


def test_true_hit_rate_formula() -> None:
    # 100 verify steps, 400 delivered decode tokens, k=6: (400-100)/(6*100) = 0.5
    assert true_hit_rate(400, 100, 6) == pytest.approx(0.5)
    # DFlash2 k=7: same stream, different k
    assert true_hit_rate(400, 100, 7) == pytest.approx(300 / 700)
    # MTP-3
    assert true_hit_rate(250, 100, 3) == pytest.approx(0.5)
    assert true_hit_rate(100, 100, 6) == 0.0  # nothing accepted
    assert true_hit_rate(700, 100, 6) == 1.0  # every slot accepted
    assert true_hit_rate(10, 0, 6) is None
    assert true_hit_rate(10, 5, None) is None


def _w(D, T, A):
    return MTPWindow(drafts=D, draft_tokens=T, accepted_drafts=A, accepted_drafts_per_step=A / D,
                     hit_rate=A / T, tokens_per_step=1 + A / D)


def test_counter_check_catches_production_107_percent() -> None:
    # Seen in a real log: Accepted 61 tokens, Drafted 57 tokens (107%).
    chk = spec_counter_check(_w(D=9.5, T=57, A=61), k=6, delivered=None, verify_steps_stream=None, n_requests=1)
    assert not chk.ok
    assert any("accepted" in p for p in chk.problems)


def test_counter_check_k_mismatch_and_identity() -> None:
    ok = spec_counter_check(_w(100, 600, 300), k=6, delivered=400, verify_steps_stream=100, n_requests=1, per_pos_sum=300)
    assert ok.ok and ok.k_observed == 6
    bad_k = spec_counter_check(_w(100, 600, 300), k=7, delivered=400, verify_steps_stream=100, n_requests=1)
    assert not bad_k.ok
    bad_pos = spec_counter_check(_w(100, 600, 300), k=6, delivered=400, verify_steps_stream=100, n_requests=1, per_pos_sum=250)
    assert not bad_pos.ok
    # Accepted far below delivered - drafts: tokens appeared from nowhere
    bad_e = spec_counter_check(_w(100, 600, 100), k=6, delivered=400, verify_steps_stream=100, n_requests=1)
    assert not bad_e.ok
    # Truncation at max_tokens: up to k discarded accepted drafts per request is fine
    trunc = spec_counter_check(_w(100, 600, 305), k=6, delivered=400, verify_steps_stream=100, n_requests=1)
    assert trunc.ok


def test_chunk_stats() -> None:
    cs = ChunkStats(tokens=97, chunks=25, t_first=10.0, t_last=12.0)
    assert cs.decode_tokens == 96 and cs.verify_steps == 24
    assert cs.steps_s == pytest.approx(12.0) and cs.decode_tok_s == pytest.approx(48.0)


def test_k_is_a_parameter_never_assumed() -> None:
    assert SpeculativeSettings().num_speculative_tokens is None
    assert SpeculativeSettings(method="dflash2", num_speculative_tokens=7).k == 7
    assert SpeculativeSettings(method="mtp", num_speculative_tokens=3).k == 3
    assert SpeculativeSettings(method="none").k is None
    with pytest.raises(ValueError):
        SpeculativeSettings(method="dflash2", num_speculative_tokens=7, dflash_k=6)
    with pytest.raises(ValueError):
        SpeculativeSettings(num_speculative_tokens=0)
    assert RunSpec(name="x").server.speculative.k is None


def test_legacy_correction_general_k() -> None:
    assert summed_ratio_correction(6) == pytest.approx(7 / 12)
    assert summed_ratio_correction(7) == pytest.approx(8 / 14)
    assert summed_ratio_correction(3) == pytest.approx(4 / 6)
