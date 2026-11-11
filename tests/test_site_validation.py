"""F2: site config placeholders are rejected, the cwd is never searched, a live
run needs a real site file (or --no-site-config), and the loaded file is shown."""

from pathlib import Path

import pytest

from xpu_llm_bench import site
from xpu_llm_bench.cli import main

GOOD = (
    "host: lab-box\n"
    "protected_containers: [prod-a]\n"
    "forbidden_ports: [9100]\n"
)


def _use(monkeypatch, path) -> None:
    monkeypatch.setenv(site.SITE_ENV, str(path))
    site.clear_cache()


@pytest.mark.parametrize(
    "body,needle",
    [
        ("host: <bench-host>\nprotected_containers: [prod-a]\nforbidden_ports: [9100]\n", "host"),
        ("host: lab\nprotected_containers: [TODO]\nforbidden_ports: [9100]\n", "protected_containers[0]"),
        ("host: example-host\nprotected_containers: [prod-a]\nforbidden_ports: [9100]\n", "host"),
        ("host: lab\nprotected_containers: [changeme]\nforbidden_ports: [9100]\n", "changeme"),
        ("host: lab\nprotected_containers: [prod-a]\nforbidden_ports: [<production-port>]\n", "forbidden_ports"),
        ("host: lab\nprotected_containers: []\nforbidden_ports: [9100]\n", "protected_containers is missing"),
        ("host: lab\nprotected_containers: [prod-a]\n", "forbidden_ports is missing"),
    ],
)
def test_placeholders_and_missing_fields_rejected(monkeypatch, tmp_path: Path, body: str, needle: str) -> None:
    p = tmp_path / "s.yaml"
    p.write_text(body)
    _use(monkeypatch, p)
    with pytest.raises(site.SiteConfigError, match="not filled in") as err:
        site.load_site()
    assert needle in str(err.value)


def test_cwd_is_not_searched(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(site.SITE_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / site.LOCAL_SITE_FILE).write_text(GOOD)
    site.clear_cache()
    cands = site.candidate_paths()
    assert tmp_path / site.LOCAL_SITE_FILE not in cands
    assert all(c.is_absolute() for c in cands)
    root = site.repo_root()
    assert root is not None and cands[0] == root / site.LOCAL_SITE_FILE
    assert cands[-1] == tmp_path / "home" / ".config" / "xpu-llm-bench" / "site.yaml"
    if not cands[0].exists():
        assert site.load_site().source is None


def test_user_config_dir_is_found(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(site.SITE_ENV, raising=False)
    home = tmp_path / "home"
    cfg = home / ".config" / "xpu-llm-bench"
    cfg.mkdir(parents=True)
    (cfg / "site.yaml").write_text(GOOD)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(site, "repo_root", lambda: None)
    site.clear_cache()
    assert site.load_site().source == str(cfg / "site.yaml")
    assert site.describe_site() == f"site config: {cfg / 'site.yaml'}"


def test_describe_site_says_none_found(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(site.SITE_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(site, "repo_root", lambda: None)
    site.clear_cache()
    assert "NONE FOUND" in site.describe_site()


def _spec_file(tmp_path: Path) -> Path:
    p = tmp_path / "spec.yaml"
    p.write_text(
        "name: s\nprofile: custom\n"
        "endpoint: {base_url: 'http://127.0.0.1:1/v1', metrics_url: 'http://127.0.0.1:1/metrics', "
        "health_url: 'http://127.0.0.1:1/health', model: m}\n"
        "server: {container_name: vllm-bench-x}\n"
        "ladder: {context_rungs_tokens: [64], concurrency_steps: [1]}\n"
    )
    return p


def test_live_run_without_site_config_is_refused(tmp_path: Path, capsys) -> None:
    code = main(["run", str(_spec_file(tmp_path)), "--output-dir", str(tmp_path / "out")])
    err = capsys.readouterr()
    assert code == 3
    assert "no site config found" in err.err and "--no-site-config" in err.err
    assert "Traceback" not in err.err
    assert "site config: disabled" in err.out


def test_plan_prints_loaded_site_file(monkeypatch, tmp_path: Path, capsys) -> None:
    p = tmp_path / "s.yaml"
    p.write_text(GOOD)
    _use(monkeypatch, p)
    assert main(["plan", str(_spec_file(tmp_path))]) == 0
    assert f"site config: {p}" in capsys.readouterr().out


def test_placeholder_site_gives_clean_cli_error(monkeypatch, tmp_path: Path, capsys) -> None:
    p = tmp_path / "s.yaml"
    p.write_text(GOOD.replace("lab-box", "<bench-host>"))
    _use(monkeypatch, p)
    code = main(["plan", str(_spec_file(tmp_path))])
    err = capsys.readouterr().err
    assert code == 3 and "placeholder" in err and "Traceback" not in err


def test_default_host_is_not_a_placeholder() -> None:
    from xpu_llm_bench.results import HardwareSnapshot
    from xpu_llm_bench.spec import RunSpec, is_placeholder

    assert not is_placeholder(RunSpec.model_validate({"name": "x"}).hardware.host)
    assert not is_placeholder(HardwareSnapshot().host)
