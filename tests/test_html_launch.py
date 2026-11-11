"""HTML report (self-contained SVG charts) and read-only launch capture."""

import json
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

import pytest

from test_runner_harness import _runner, _spec
from xpu_llm_bench import launch
from xpu_llm_bench.cli import main
from xpu_llm_bench.fingerprint import fingerprint_from_inspect
from xpu_llm_bench.html_report import render_html
from xpu_llm_bench.import_legacy import import_legacy_file
from xpu_llm_bench.results import RunManifest, read_jsonl

FIX = Path(__file__).resolve().parent / "fixtures"
EXAMPLE = FIX / "example_run"
SECRETS = ("not-a-real-key-0001", "not-a-real-token-0002", "not-a-real-token-0003", "not-a-real-key-0004",
           "not-a-real-password-0005", "not-a-real-key-0006", "hf_notarealtokenabcdefghij")
VOID = {"meta", "br", "hr", "img", "input", "link", "area", "base", "col", "embed", "source", "track", "wbr"}


class _Checker(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.tags: list[tuple[str, dict]] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unbalanced </{tag}> (open: {self.stack[-3:]})")
            return
        self.stack.pop()


def _example():
    return read_jsonl(EXAMPLE / "results.jsonl"), RunManifest.load(EXAMPLE / "manifest.json")


def _check_self_contained(doc: str) -> _Checker:
    assert doc.startswith("<!DOCTYPE html>")
    c = _Checker()
    c.feed(doc)
    c.close()
    assert not c.errors and c.stack == [], (c.errors, c.stack)
    for tag, attrs in c.tags:
        assert tag not in {"script", "link", "img", "iframe", "object", "embed"}, tag
        for key in ("src", "href", "xlink:href"):
            assert not re.match(r"\s*(https?:)?//", attrs.get(key) or ""), (tag, key, attrs.get(key))
    assert "url(" not in doc and "@import" not in doc
    for svg in re.findall(r"<svg\b.*?</svg>", doc, flags=re.S):
        ET.fromstring(svg)  # every chart is well-formed XML
    return c


def test_example_report_is_valid_and_self_contained() -> None:
    rows, man = _example()
    doc = render_html(rows, man)
    _check_self_contained(doc)
    assert "prefers-color-scheme" in doc
    for title in ("Decode speed", "Prefill speed", "Time to first token", "Verify steps",
                  "MTP accepted drafts per step", "True hit rate"):
        assert title in doc
    assert "tok/s per user" in doc and "context per user (tokens" in doc
    # numbers table has every row
    assert doc.count("<tr") >= len(rows) + 1


def test_one_series_per_concurrency() -> None:
    rows, man = _example()
    doc = render_html(rows, man)
    charts = re.findall(r'<div class="card"><h3>(.*?)</h3>(.*?)</div>', doc, flags=re.S)
    assert charts
    for _title, body in charts:
        users_lines = re.findall(r'<path class="s u(\d+) r0"', body)
        assert len(users_lines) == len(set(users_lines))  # one line per user count
        assert set(users_lines) <= {"1", "2", "4", "8"}
    decode = next(b for t, b in charts if t.startswith("Decode speed"))
    assert sorted(set(re.findall(r'<path class="s u(\d+) r0"', decode)), key=int) == ["1", "2", "4", "8"]
    # x axis follows the ladder contexts
    assert all(lab in decode for lab in (">8K<", ">32K<", ">64K<", ">131K<", ">163K<", ">196K<", ">260K<"))


def _points(body: str, users: int) -> list[tuple[float, str]]:
    return [(float(cx), t) for cx, t in re.findall(
        rf'<circle cx="([\d.]+)"[^>]*data-users="{users}"[^>]*><title>([^<]*)</title>', body)]


def test_lines_end_at_each_user_count_ceiling() -> None:
    rows, man = _example()
    doc = render_html(rows, man)
    charts = re.findall(r'<div class="card"><h3>(.*?)</h3>(.*?)</div>', doc, flags=re.S)
    decode = next(b for t, b in charts if t.startswith("Decode speed"))  # first group
    last = {u: max(_points(decode, u))[1] for u in (1, 2, 4, 8)}
    assert "260K context" in last[1] and "260K context" in last[2]
    assert "163K context" in last[4] and "64K context" in last[8]
    marks = re.findall(r'<g class="ceil-mark" data-users="(\d+)"><title>([^<]*)</title>', decode)
    assert sorted(u for u, _ in marks) == ["4", "8"]
    tip4 = next(t for u, t in marks if u == "4")
    assert "Last measured 163K; stopped at 196K (kv_capacity)" in tip4 and "exceeds the server KV capacity" in tip4
    # log-scaled x axis: 8K->32K (2 doublings) is twice as wide as 32K->64K (1 doubling)
    xs = sorted({x for x, _ in _points(decode, 1)})
    assert abs((xs[1] - xs[0]) - 2 * (xs[2] - xs[1])) < 1.0
    assert "log scale" in decode


def test_header_shows_max_model_len_and_per_user_ceilings() -> None:
    rows, man = _example()
    doc = render_html(rows, man)
    assert "max_model_len 262144 (from /v1/models)" in doc
    assert "720,896 tokens (from boot_log)" in doc
    reach = doc[doc.index("Context reached per user count"):doc.index("Server launch")]
    assert "<td>1</td><td>260K</td><td>reached the top rung</td>" in reach
    assert "<td>4</td><td>163K</td><td>stopped at 196K (kv_capacity)</td>" in reach
    assert "<td>8</td><td>64K</td><td>stopped at 131K (kv_capacity)</td>" in reach
    assert "SYNTHETIC" in json.dumps(man.notes)


def test_group_headings_and_quality_pass_counts() -> None:
    from xpu_llm_bench.html_report import quality_line

    rows, man = _example()
    doc = render_html(rows, man)
    assert "<h3>Code edits, closed loop</h3>" in doc and "<h3>Multi-turn chat, closed loop</h3>" in doc
    assert "code_edits · closed_all_decoding" not in doc
    code = [r for r in rows if r.prompt_type == "code_edits"]
    line = quality_line(code)
    ran = [r for r in code if r.quality.code_edit_pass is not None]
    assert f"Code edit: {len(ran) - 1} of {len(ran)} pass" in line
    assert line.startswith("Code edit:") and "Greedy match:" in line and "Logprob: ok" in line
    assert f"<b>Quality:</b> {line}" in doc  # above the group
    assert "<b>Quality, all rows:</b>" in doc  # overall, at the top
    allnum = doc[doc.index("<h2>All numbers</h2>"):]
    assert "Quality pass counts per group" in allnum and f"Code edits, closed loop: {line}" in allnum
    # checks that never ran are skipped, None is not-run
    long_form = quality_line([r for r in rows if r.prompt_type == "long_form"])
    assert "Code edit" not in long_form


def test_averages_across_prompt_types() -> None:
    from xpu_llm_bench.html_report import average_rows

    rows, man = _example()
    avg, table = average_rows(rows)
    by = {(t["users"], t["ctx"]): t for t in table}
    # 1 user at 8K: every prompt type clean -> plain mean of the three
    one = [r for r in rows if r.users == 1 and r.ctx_per_user == 8192]
    assert len(one) == 3 and by[(1, 8192)]["averaged"]
    assert by[(1, 8192)]["decode_tok_s_per_user"] == pytest.approx(sum(r.decode_tok_s_per_user for r in one) / 3)
    # a cell where any prompt type is flagged or failed is NOT averaged over fewer types
    for cell, (ptype, why) in {(2, 32768): ("multi_turn_chat", "NOT_AS_LABELED"),
                               (4, 65536): ("code_edits", "QSA_FALLBACK"),
                               (8, 8192): ("multi_turn_chat", "HIT_RATE_UNKNOWN")}.items():
        rec = by[cell]
        assert not rec["averaged"] and rec["decode_tok_s_per_user"] is None
        assert why in rec["excluded"][ptype] and ptype not in rec["breakdown"]
    assert (4, 196608) not in by  # past the ceiling: no clean row anywhere
    assert len(avg) == sum(t["averaged"] for t in table)
    doc = render_html(rows, man)
    assert "<h2>Averages across prompt types</h2>" in doc and "Flagged and invalid rows are excluded" in doc
    assert "<h3>Average across prompt types, closed loop (Code edits · Long-form writing · Multi-turn chat)</h3>" in doc
    assert "each line = average of Code edits · Long-form writing · Multi-turn chat" in doc


def test_average_cards_label_prompt_types() -> None:
    rows, man = _example()
    doc = render_html(rows, man)
    sec = doc[doc.index("<h2>Averages across prompt types</h2>"):doc.index("<h2>All numbers</h2>")]
    assert "<table><thead><tr><th>load</th><th>users</th><th>context</th>" not in sec  # the old table is gone
    cards = re.findall(r'<div class="acard( partial)?" data-users="(\d+)" data-ctx="(\d+)">(.*?)</tbody></table></div>',
                       sec, flags=re.S)
    assert cards
    full = {(u, c): body for p, u, c, body in cards if not p}
    part = {(u, c): body for p, u, c, body in cards if p}
    c18 = full[("1", "8192")]
    assert "Avg of Code edits · Long-form writing · Multi-turn chat" in c18
    assert "steps/s per user" in c18 and "decode tok/s per user" in c18
    assert "drafts/step" in c18 and "hit rate" in c18 and "TTFT" in c18 and "prefill" in c18
    for name in ("Code edits", "Long-form writing", "Multi-turn chat"):
        assert f"<tr><td>{name}</td>" in c18  # per-type breakdown row
    p232 = part[("2", "32768")]
    assert "Not averaged" in p232 and "excluded: flagged NOT_AS_LABELED" in p232 and "Multi-turn chat" in p232
    assert "<script" not in doc

def test_flagged_points_are_hollow_with_tooltip() -> None:
    rows, man = _example()
    doc = render_html(rows, man)
    assert 'class="pt hollow"' in doc
    hollow = re.findall(r'class="pt hollow"[^>]*><title>([^<]*)</title>', doc)
    assert any("NOT_AS_LABELED" in t for t in hollow) and any("QSA_FALLBACK" in t for t in hollow)
    assert any("HIT_RATE_UNKNOWN" in t for t in hollow)


def test_header_has_fingerprint_and_launch() -> None:
    rows, man = _example()
    doc = render_html(rows, man)
    for needle in ("image", "attention path", "KV format", "graphs", "speculative decoding", "hardware",
                   "Server launch", "vllm serve /models/target", "Caveats", "started"):
        assert needle in doc, needle
    for s in SECRETS:
        assert s not in doc


def test_missing_metrics_and_no_spec_decode(tmp_path: Path) -> None:
    dest = tmp_path / "imported.jsonl"
    import_legacy_file(FIX / "legacy_speedfix_rungs.jsonl", dest, run_id="legacy")
    rows = read_jsonl(dest)
    for r in rows:
        r.ttft_ms = None  # a metric with no data at all
    doc = render_html(rows, None)
    _check_self_contained(doc)
    assert "No data for this metric." in doc and "No manifest.json" in doc
    plain = [r.model_copy(update={"k": None, "accepted_drafts_per_step": None, "hit_rate": None}) for r in rows]
    doc2 = render_html(plain, None)
    assert "No speculative decoding in this run" in doc2 and "True hit rate" not in doc2
    _check_self_contained(render_html([], None))


def test_compare_view_dashed(tmp_path: Path) -> None:
    rows, man = _example()
    other = [r.model_copy(update={"decode_tok_s_per_user": (r.decode_tok_s_per_user or 0) * 0.9}) for r in rows]
    doc = render_html(rows, man, compare=[("baseline", other)])
    _check_self_contained(doc)
    assert 'stroke-dasharray="6 4"' in doc and "Compare view" in doc


def test_cli_report_html_and_compare_html(tmp_path: Path) -> None:
    out = tmp_path / "r.html"
    assert main(["report", str(EXAMPLE), "--html-out", str(out)]) == 0
    _check_self_contained(out.read_text())
    cmp_out = tmp_path / "c.html"
    assert main(["compare", str(EXAMPLE), str(EXAMPLE / "results.jsonl"), "--html", str(cmp_out),
                 "--output", str(tmp_path / "c.md")]) == 0
    assert "Compare view" in cmp_out.read_text()


# -- launch capture --------------------------------------------------------------
def _inspect() -> dict:
    return json.loads((FIX / "docker_inspect_example.json").read_text())[0]


def test_launch_from_inspect_redacts_secrets() -> None:
    rec = launch.capture(inspect_info=_inspect(), boot_log=(FIX / "boot_log_example.txt").read_text())
    text = json.dumps(rec)
    for s in SECRETS:
        assert s not in text, s
    env = rec["env"]
    assert env["HF_TOKEN"] == env["VLLM_API_KEY"] == env["DB_PASSWORD"] == "<redacted>"
    assert env["NUM_SPECULATIVE_TOKENS"] == "6" and env["PATH"].startswith("/usr")
    assert rec["cmd"][rec["cmd"].index("--api-key") + 1] == "<redacted>"
    assert "--hf-token=<redacted>" in rec["cmd"]
    assert rec["command_line"].startswith("/opt/example/entrypoint.sh vllm serve /models/target")
    assert rec["devices"] == ["/dev/dri:/dev/dri"] and rec["group_add"] == ["render"]
    assert rec["image"] == "example-registry/vllm-xpu:example" and rec["source"] == "docker_inspect"
    assert len(rec["vllm_boot_config"]) == 3
    assert any("non-default args" in ln and "'num_speculative_tokens': 6" in ln for ln in rec["vllm_boot_config"])
    assert rec["captured_utc"].endswith("+00:00") and rec["captured_local"] and rec["read_only"] is True


def test_secret_name_detection() -> None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY", "--api-key", "db.password", "AWS_SECRET"):
        assert launch.is_secret_name(name), name
    for name in ("num_speculative_tokens", "max_num_batched_tokens", "tokenizer", "PATH", "KV_CACHE_DTYPE"):
        assert not launch.is_secret_name(name), name
    assert "<redacted>" in launch.redact_text("token=hf_abcdefghijklmnopqrstuvwx")


def test_process_cmdline_capture() -> None:
    ps = (" 10 /usr/bin/python3 other.py\n"
          " 11 /usr/bin/python3 -m vllm.entrypoints.openai.api_server --port 9000 --api-key s3cr3t-value-xx\n"
          " 12 vllm serve /m --port 8102\n")
    argv = launch.find_server_process(ps, port=8102)
    assert argv == ["vllm", "serve", "/m", "--port", "8102"]
    rec = launch.launch_from_process(launch.find_server_process(ps, port=9000))
    assert "s3cr3t-value-xx" not in json.dumps(rec) and rec["source"] == "process_cmdline"
    assert launch.find_server_process(" 1 bash\n") is None


def test_header_text_has_both_timestamps() -> None:
    rec = launch.capture(inspect_info=_inspect())
    text = launch.header_text(rec, "run x")
    assert "captured (UTC):" in text and "start command:" in text and "<redacted>" in text
    assert launch.capture()["source"] == "none"


def test_runner_writes_launch_runlog_and_html(mock_server, tmp_path: Path) -> None:
    info = _inspect()
    mock_server["logs"].extend((FIX / "boot_log_example.txt").read_text().splitlines())
    spec = _spec(mock_server["base"])
    r = _runner(spec, tmp_path / "o", mock_server, tmp_path)
    r._fingerprint_fn = lambda: fingerprint_from_inspect(None, "vllm-bench-mock", info)
    man = r.run()
    out = tmp_path / "o"
    rec = json.loads((out / "launch.json").read_text())
    assert rec["source"] == "docker_inspect" and man.launch["command_line"] == rec["command_line"]
    assert any("non-default args" in ln for ln in rec["vllm_boot_config"])
    log = (out / "run.log").read_text()
    assert log.startswith("=" * 78) and "captured (UTC):" in log and "start command:" in log
    assert "rung multi_turn_chat:closed:1u:256:S" in log and "run completed" in log
    for s in SECRETS:
        assert s not in log and s not in (out / "manifest.json").read_text()
    html = (out / "report.html").read_text()
    _check_self_contained(html)
    assert "Server launch" in html and "vllm serve /models/target" in html


def test_runner_no_html_and_process_fallback(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"])
    r = _runner(spec, tmp_path / "o", mock_server, tmp_path, html=False,
                ps_fn=lambda: " 7 vllm serve /m --port 1 --api-key zzzz-secret-zzzz\n")
    r.run()
    out = tmp_path / "o"
    assert not (out / "report.html").exists()
    rec = json.loads((out / "launch.json").read_text())
    assert rec["source"] == "process_cmdline" and "zzzz-secret-zzzz" not in json.dumps(rec)


def test_dry_run_writes_no_launch_or_html(mock_server, tmp_path: Path) -> None:
    r = _runner(_spec(mock_server["base"]), tmp_path / "o", mock_server, tmp_path, dry_run=True)
    r.run()
    assert not (tmp_path / "o" / "dry-run" / "report.html").exists()
    assert not (tmp_path / "o" / "dry-run" / "run.log").exists()
