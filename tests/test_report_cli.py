"""F8: `bench report` prints markdown to stdout by default; results/README.md documents sessions.json."""

from pathlib import Path

from test_runner_harness import _runner, _spec
from xpu_llm_bench.cli import main

ROOT = Path(__file__).resolve().parents[1]


def _run(mock_server, tmp_path: Path) -> Path:
    _runner(_spec(mock_server["base"]), tmp_path / "o", mock_server, tmp_path).run()
    return tmp_path / "o"


def test_report_prints_markdown_to_stdout(mock_server, tmp_path: Path, capsys) -> None:
    out = _run(mock_server, tmp_path)
    capsys.readouterr()
    assert main(["report", str(out / "results.jsonl")]) == 0
    md = capsys.readouterr().out
    assert md.startswith("# ") and "## Ladder" in md and "| rung_id |" in md


def test_report_accepts_run_dir_and_csv_still_prints(mock_server, tmp_path: Path, capsys) -> None:
    out = _run(mock_server, tmp_path)
    capsys.readouterr()
    assert main(["report", str(out), "--csv", str(tmp_path / "r.csv")]) == 0
    cap = capsys.readouterr()
    assert "## Ladder" in cap.out and (tmp_path / "r.csv").exists()


def test_report_output_writes_file(mock_server, tmp_path: Path, capsys) -> None:
    out = _run(mock_server, tmp_path)
    capsys.readouterr()
    assert main(["report", str(out / "results.jsonl"), "--output", str(tmp_path / "r.md")]) == 0
    cap = capsys.readouterr()
    assert cap.out == "" and "wrote" in cap.err
    assert "## Ladder" in (tmp_path / "r.md").read_text()


def test_results_readme_documents_sessions_json() -> None:
    text = (ROOT / "results" / "README.md").read_text()
    assert "sessions.json" in text and "regenerat" in text and "dry-run" in text
