from pathlib import Path

from xpu_llm_bench.cli import main

SPECS = Path(__file__).resolve().parents[1] / "specs"


def test_plan_cli(capsys) -> None:
    # Pin MemAvailable: without it, access.mode local reads this machine's /proc/meminfo.
    rc = main(["plan", str(SPECS / "qsa-vs-dense.yaml"), "--profile", "quick", "--mem-available", "64"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Plan for" in out
    assert "code_edits" in out


def test_report_and_compare_cli(tmp_path: Path, capsys) -> None:
    # Tiny synthetic jsonl
    from xpu_llm_bench.results import RungResult, write_jsonl_row

    def row(run, users=1, toks=50.0, path="qsa"):
        return RungResult(
            run_id=run,
            rung_id=f"code_edits:closed:{users}u:8192:meas",
            users=users,
            ctx_tokens=[8192] * users,
            ctx_per_user=8192,
            prompt_type="code_edits",
            load_mode="closed_all_decoding",
            decode_tok_s_per_user=toks,
            decode_tok_s_total=toks * users,
            attention_path=path,
        )

    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    write_jsonl_row(a, row("a", toks=50))
    write_jsonl_row(b, row("b", toks=60))
    assert main(["report", str(a)]) == 0
    assert "decode_tok_s_per_user" in capsys.readouterr().out
    assert main(["compare", str(a), str(b)]) == 0
    assert "+10" in capsys.readouterr().out or "+10.000" in capsys.readouterr().out


def test_import_legacy_cli(tmp_path: Path, capsys) -> None:
    src = Path(__file__).resolve().parent / "fixtures" / "legacy_rungs.jsonl"
    dest = tmp_path / "out.jsonl"
    assert main(["import-legacy", str(src), str(dest), "--correct-mtp6-hit-rate"]) == 0
    assert dest.exists()
    assert "imported 3" in capsys.readouterr().out
