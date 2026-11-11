"""ladder.repeats / --repeats: N decode windows per rung on the same context.

- Rows store the median and per-repeat values (min, max, stdev from 3).
- The context is not regrown and quality is checked once per rung.
- The planner time estimate counts the repeats.
- The HTML shows error bars and 'median (min to max, n=N)' only when N > 1.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from xpu_llm_bench import cli
from xpu_llm_bench.html_report import (
    average_rows,
    averages_cards,
    numbers_table,
    render_html,
    svg_chart,
)
from xpu_llm_bench.planner import estimate_seconds, make_plan
from xpu_llm_bench.results import REPEAT_METRICS, RungResult, RunManifest, read_jsonl, repeat_stat
from xpu_llm_bench.runner import Runner, apply_repeats
from xpu_llm_bench.schedule import build_schedule
from xpu_llm_bench.spec import RunSpec

OUT_TOKENS = 40  # distinct from growth / needle / warmup request sizes


def _spec(base: str, repeats: int | None = None) -> dict:
    ladder = {"context_rungs_tokens": [256, 512], "concurrency_steps": [1, 2], "output_tokens_per_rung": OUT_TOKENS,
              "min_window_steps": 4, "metrics_poll_s": 0.05}
    if repeats is not None:
        ladder["repeats"] = repeats
    return {
        "name": "rep",
        "profile": "custom",
        "endpoint": {"base_url": f"{base}/v1", "metrics_url": f"{base}/metrics", "health_url": f"{base}/health",
                     "model": "m"},
        "server": {"container_name": "vllm-bench-mock", "speculative": {"method": "mtp", "num_speculative_tokens": 6}},
        "hardware": {"gpu_count": 1, "gpu_mem_gib": 1},
        "prompts": {"types": ["multi_turn_chat"], "seed": 3},
        "quality": {"code_edit": False},
        "ladder": ladder,
    }


def _run_cli(tmp_path: Path, mock_server, monkeypatch, name: str, extra: list[str], repeats_in_spec: int | None = None):
    from xpu_llm_bench import runner

    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(_spec(mock_server["base"], repeats_in_spec)))
    real_init = Runner.__init__

    def init(self, *a, **kw):
        kw.setdefault("require_container", False)
        kw.setdefault("fingerprint_fn", lambda: None)
        kw.setdefault("fetch_logs", lambda: "")
        kw.setdefault("host_fn", lambda: {"gpus": []})
        kw.setdefault("ps_fn", lambda: "")
        real_init(self, *a, **kw)

    monkeypatch.setattr(runner.Runner, "__init__", init)
    out = tmp_path / name
    state = mock_server["state"]
    state.chat_log.clear()
    rc = cli.main(["run", str(p), "--output-dir", str(out), "--no-site-config", "--yes", *extra])
    assert rc == 0
    rows = read_jsonl(out / "rep" / "results.jsonl")
    man = RunManifest.load(out / "rep" / "manifest.json")
    return rows, man, list(state.chat_log), (out / "rep")


def test_cli_repeats_measure_same_context_and_quality_once(mock_server, tmp_path: Path, monkeypatch) -> None:
    rows1, man1, log1, _ = _run_cli(tmp_path, mock_server, monkeypatch, "one", [])
    rows3, man3, log3, run_dir = _run_cli(tmp_path, mock_server, monkeypatch, "three", ["--repeats", "3"])
    assert man1.status == man3.status == "completed"
    assert len(rows1) == len(rows3) == 4

    # Default: one window, no repeat stats.
    assert all(r.repeats == 1 and r.repeat_stats == {} for r in rows1)

    # Measured windows triple; every other request (growth, warmup, needle) is unchanged.
    meas1 = [pt for pt, mt in log1 if mt == OUT_TOKENS]
    meas3 = [pt for pt, mt in log3 if mt == OUT_TOKENS]
    assert len(meas3) == 3 * len(meas1)
    assert sorted(x for x in log1 if x[1] != OUT_TOKENS) == sorted(x for x in log3 if x[1] != OUT_TOKENS)

    # Same context for every repeat (nothing regrown), and the conversation
    # afterwards is what a single-window run sees.
    assert sorted(meas3) == sorted(pt for pt in meas1 for _ in range(3))
    assert [r.ctx_tokens for r in rows1] == [r.ctx_tokens for r in rows3]

    for r in rows3:
        assert r.repeats == 3
        assert set(r.repeat_stats) == set(REPEAT_METRICS)
        for m in ("decode_tok_s_per_user", "steps_s_per_user", "accepted_drafts_per_step", "hit_rate", "ttft_ms"):
            st = r.repeat_stats[m]
            assert len(st.values) == 3 and st.n == 3
            assert st.min <= st.median <= st.max and st.stdev is not None
            assert getattr(r, m) == pytest.approx(st.median)
        assert r.decode_tok_s_total == pytest.approx(r.decode_tok_s_per_user * r.users)
        pf = r.repeat_stats["prefill_tok_s"]
        assert pf.values == [r.prefill_tok_s] and pf.n == 1  # prefill is measured once per rung
        assert "REPEAT_FAILED" not in r.flags
    # Quality: identical checks, once per rung.
    assert [r.quality.needle_pass for r in rows1] == [r.quality.needle_pass for r in rows3]

    # The report written at the end of the run shows the repeats.
    html = (run_dir / "report.html").read_text()
    assert 'class="ebar"' in html and "n=3)" in html


def test_spec_repeats_and_cli_override(tmp_path: Path) -> None:
    assert RunSpec.model_validate({"name": "x"}).ladder.repeats == 1
    assert RunSpec.model_validate({"name": "x", "ladder": {"repeats": 5}}).ladder.repeats == 5
    with pytest.raises(ValidationError):
        RunSpec.model_validate({"name": "x", "ladder": {"repeats": 0}})
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump({"name": "s", "ladder": {"context_rungs_tokens": [8192]}}))
    with pytest.raises(SystemExit):
        cli.main(["plan", str(p), "--repeats", "0"])


def test_planner_time_counts_repeats(tmp_path: Path, capsys) -> None:
    base = {"name": "x", "ladder": {"context_rungs_tokens": [8192, 32768], "concurrency_steps": [1, 2]}}
    s1 = RunSpec.model_validate(base)
    s3 = RunSpec.model_validate({**base, "ladder": {**base["ladder"], "repeats": 3}})
    rungs = build_schedule(s1)
    a = s1.planner
    decode_s = sum(r.output_tokens / a.assumed_decode_tok_s_per_user for r in rungs)
    assert estimate_seconds(s3, rungs) - estimate_seconds(s1, rungs) == pytest.approx(2 * decode_s)
    assert make_plan(s3).repeats == 3
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump(base))
    assert cli.main(["plan", str(p), "--mem-available", "100"]) == 0
    one = capsys.readouterr().out
    assert cli.main(["plan", str(p), "--mem-available", "100", "--repeats", "4"]) == 0
    four = capsys.readouterr().out
    assert "repeats=" not in one and "repeats=4 per rung" in four


def _row(pt: str, users: int, ctx: int, vals: dict[str, list[float | None]] | None, base: float) -> RungResult:
    r = RungResult(run_id="r", rung_id=f"{pt}:closed:{users}u:{ctx}:C", users=users, ctx_tokens=[ctx] * users,
                   ctx_per_user=ctx, prompt_type=pt, load_mode="closed_all_decoding",
                   decode_tok_s_per_user=base, steps_s_per_user=base / 3, accepted_drafts_per_step=2.0,
                   hit_rate=0.4, ttft_ms=500.0, prefill_tok_s=1500.0, k=6)
    if vals is not None:
        first = {m: getattr(r, m) for m in REPEAT_METRICS}
        extra = [{m: vals[m][i] for m in vals} for i in range(len(next(iter(vals.values()))))]
        apply_repeats(r, [{**first, **e} for e in extra])
    return r


def test_apply_repeats_median_min_max_stdev() -> None:
    r = _row("long_form", 2, 8192, {"decode_tok_s_per_user": [40.0, 60.0], "ttft_ms": [300.0, 900.0]}, 50.0)
    st = r.repeat_stats["decode_tok_s_per_user"]
    assert r.repeats == 3 and st.values == [50.0, 40.0, 60.0]
    assert (st.median, st.min, st.max) == (50.0, 40.0, 60.0) and st.stdev == pytest.approx(10.0)
    assert r.ttft_ms == 500.0 and r.repeat_stats["ttft_ms"].max == 900.0
    assert r.decode_tok_s_total == pytest.approx(100.0)
    # A failed repeat is left out; with 2 values there is no stdev.
    r2 = RungResult(run_id="r", rung_id="x", users=1, ctx_tokens=[1], ctx_per_user=1, prompt_type="p",
                    load_mode="closed_all_decoding", decode_tok_s_per_user=10.0)
    apply_repeats(r2, [{"decode_tok_s_per_user": 20.0}, None])
    st2 = r2.repeat_stats["decode_tok_s_per_user"]
    assert st2.values == [10.0, 20.0, None] and st2.n == 2 and st2.stdev is None and st2.median == 15.0
    assert repeat_stat([None, None]).n == 0


def test_html_error_bars_and_ranges_only_with_repeats() -> None:
    reps = {"decode_tok_s_per_user": [40.0, 60.0], "steps_s_per_user": [12.0, 20.0]}
    rows = [_row("long_form", 1, 8192, reps, 50.0), _row("multi_turn_chat", 1, 8192, reps, 50.0)]
    plain = [_row("long_form", 1, 8192, None, 50.0), _row("multi_turn_chat", 1, 8192, None, 50.0)]

    chart = svg_chart([("r", rows[:1])], "decode_tok_s_per_user", "Decode speed", "tok/s per user", 1.0, [1])
    assert 'class="ebar"' in chart and "median of 3 repeats" in chart
    assert 'class="ebar"' not in svg_chart([("r", plain[:1])], "decode_tok_s_per_user", "D", "u", 1.0, [1])

    table = numbers_table(rows)
    assert "50.00 (40.00 to 60.00, n=3)" in table and 'title="stdev 10.00 over 3 repeats"' in table
    assert "40.0% (40.0 to 40.0%, n=3)" in table  # hit rate in percent
    assert ", n=" not in numbers_table(plain)

    _avg, recs = average_rows(rows)
    cards = averages_cards(recs)
    assert "(40.00 to 60.00, n=3)" in cards  # headline and per-type breakdown
    assert cards.count("n=3") >= 4
    _avg, recs_plain = average_rows(plain)
    assert "n=" not in averages_cards(recs_plain)

    full = render_html(rows)
    assert "point = median of 3 repeats" in full and "svg .ebar" in full and "repeats" in full
    plain_html = render_html(plain)
    assert "ebar" not in plain_html and "n=3" not in plain_html and "repeats" not in plain_html
