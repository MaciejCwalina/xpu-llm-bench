"""Command-line entry: bench plan | run | report | compare | import-legacy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xpu_llm_bench import __version__
from xpu_llm_bench.import_legacy import import_legacy_file
from xpu_llm_bench.planner import GRID_WARNING, format_plan, make_plan, measured_host_gib_from_rows
from xpu_llm_bench.report import (
    COLUMNS,
    compare_runs,
    format_compare,
    load_run,
    rows_to_csv,
    rows_to_markdown,
    run_report,
)
from xpu_llm_bench.results import RunManifest
from xpu_llm_bench.runner import RunRefused, run_spec
from xpu_llm_bench.safety import SafetyError
from xpu_llm_bench.site import SiteConfigError, describe_site
from xpu_llm_bench.spec import load_spec


def _add_profile(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--profile",
        choices=["quick", "full", "custom"],
        default=None,
        help="Override the spec profile. quick ≈ 10 min; full is the complete matrix.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="Consistent, comparable LLM-serving benchmarks on Intel XPU (vLLM).",
    )
    parser.add_argument("--version", action="version", version=f"xpu-llm-bench {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan", help="Print rung order, time estimate, and KV/RAM feasibility")
    plan.add_argument("spec", type=Path)
    plan.add_argument(
        "--measured",
        type=Path,
        default=None,
        help="results.jsonl from an earlier run: use its measured server RSS for the host-RAM check",
    )
    _add_profile(plan)

    run = sub.add_parser("run", help="Drive the ladder against an OpenAI-compatible endpoint")
    run.add_argument("spec", type=Path)
    run.add_argument("--output-dir", type=Path, default=Path("results"))
    run.add_argument("--resume", action="store_true", help="Skip rungs already completed in the output dir")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Safety preflight + plan + placeholder rows. No HTTP, no bench-hold, no GPU.",
    )
    run.add_argument(
        "--ram-wait-min",
        type=float,
        default=0.0,
        help="After a RAM-floor abort, wait up to N minutes for MemAvailable >= floor+1 GiB, then retry the rung",
    )
    run.add_argument(
        "--no-site-config",
        action="store_true",
        help="Acknowledge running without a site config (only built-in neutral protection applies)",
    )
    run.add_argument("--variant", default=None, help="Run only this variant (boot its container first)")
    run.add_argument(
        "--pause-between-variants",
        action="store_true",
        help="Prompt before each variant so you can boot that variant's container",
    )
    _add_profile(run)

    report = sub.add_parser(
        "report", help="Print a run's Markdown report to stdout (--output FILE to write it instead)"
    )
    report.add_argument("run", type=Path, help="results.jsonl from a bench run, or its run directory")
    report.add_argument("--csv", type=Path, default=None, help="Write every column as CSV")
    report.add_argument("--output", type=Path, default=None, help="Write the markdown here instead of stdout")
    report.add_argument("--wide", action="store_true", help="Markdown table with every column")

    compare = sub.add_parser("compare", help="Compare two or more runs; flag non-comparable rows")
    compare.add_argument("runs", nargs="+", type=Path, help="results.jsonl files, A B [C...]")
    compare.add_argument("--output", type=Path, default=None)

    imp = sub.add_parser("import-legacy", help="Convert older ad-hoc rung JSONL into the current schema")
    imp.add_argument("src", type=Path)
    imp.add_argument("dest", type=Path)
    imp.add_argument("--run-id", default="legacy")
    imp.add_argument(
        "--correct-mtp6-hit-rate",
        action="store_true",
        help="Multiply the imported acceptance counter by 7/12 (old MTP-6 summed ratio). "
        "Lands in acceptance_rate_vllm; hit_rate is always derived from tokens_per_step/drafts",
    )
    imp.add_argument(
        "--correct-summed-ratio",
        type=int,
        default=None,
        metavar="K",
        help="Convert the old summed counter ratio 2A/((k+1)D) to A/(kD) in acceptance_rate_vllm: multiply by (K+1)/(2K)",
    )
    imp.add_argument("--port", type=int, default=None, help="Server port of the legacy run (a protected production port -> PROTECTED_PORT flag)")
    imp.add_argument("--prompt-type", default="unknown")
    return parser


def _fail(msg: str, code: int) -> int:
    print(f"bench: {msg}", file=sys.stderr)
    return code


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _main(parser, args)
    except RunRefused as exc:
        where = ""
        if exc.manifest is not None:
            where = f"\n(manifest marked {exc.manifest.status}: {exc.manifest.output_jsonl or ''})".replace(
                "results.jsonl", "manifest.json"
            )
        return _fail(f"run refused:\n{exc}{where}", 3)
    except (SafetyError, SiteConfigError) as exc:
        return _fail(f"refused: {exc}", 3)
    except (ValueError, FileNotFoundError) as exc:
        return _fail(str(exc), 2)


def _main(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.cmd == "plan":
        spec = load_spec(args.spec, profile=args.profile)
        measured = None
        if args.measured:
            measured = measured_host_gib_from_rows(load_run(args.measured))
        plan = make_plan(spec, measured_host_gib=measured)
        print(describe_site())
        sys.stdout.write(format_plan(plan))
        return 0 if plan.feasibility.ok else 1
    if args.cmd == "run":
        spec = load_spec(args.spec, profile=args.profile)
        print(describe_site())
        if spec.ladder.shape == "grid":
            print(GRID_WARNING, file=sys.stderr)
        if args.dry_run:
            sys.stdout.write(format_plan(make_plan(spec)))
        manifests = run_spec(
            spec,
            args.output_dir,
            dry_run=args.dry_run,
            resume=args.resume,
            ram_wait_min=args.ram_wait_min,
            no_site_config=args.no_site_config,
            variant=args.variant,
            pause_between_variants=args.pause_between_variants,
        )
        for man in manifests:
            if man.status == "dry_run":
                out = Path(man.output_jsonl or "").parent
                print(f"{man.spec_name}: DRY RUN run_id={man.run_id} planned_rungs={len(man.planned_rung_ids)} -> {out}")
                print("  nothing was sent; this manifest is ignored by --resume")
                print("  safety commands a live run would execute:")
                for line in (man.hold or {}).get("would_run", []):
                    print(f"    {line}")
                for warn in (man.launch_check or {}).get("warnings", []):
                    print(f"  WARNING: {warn}")
                continue
            print(f"{man.spec_name}: {man.status} run_id={man.run_id} rungs={len(man.completed_rung_ids)}")
            for item in (man.launch_check or {}).get("mismatches", []):
                print(f"  NOT_AS_LABELED: {item}", file=sys.stderr)
            if man.abort_reason:
                print(f"  aborted: {man.abort_reason}", file=sys.stderr)
                return 2
        return 0
    if args.cmd == "report":
        run_path = args.run / "results.jsonl" if args.run.is_dir() else args.run
        rows = load_run(run_path)
        man_path = run_path.parent / "manifest.json"
        manifest = RunManifest.load(man_path) if man_path.exists() else None
        md = rows_to_markdown(rows, COLUMNS) if args.wide else run_report(rows, manifest)
        if args.output:
            args.output.write_text(md, encoding="utf-8")
            print(f"wrote {args.output}", file=sys.stderr)
        else:
            sys.stdout.write(md)  # markdown on stdout by default
        if args.csv:
            args.csv.write_text(rows_to_csv(rows, COLUMNS), encoding="utf-8")
            print(f"wrote {args.csv}", file=sys.stderr)
        return 0
    if args.cmd == "compare":
        if len(args.runs) < 2:
            parser.error("compare needs at least two run files")
        texts = []
        base = load_run(args.runs[0])
        for other in args.runs[1:]:
            texts.append(
                format_compare(
                    compare_runs(base, load_run(other)),
                    name_a=str(args.runs[0]),
                    name_b=str(other),
                )
            )
        out = "\n".join(texts)
        if args.output:
            args.output.write_text(out, encoding="utf-8")
        else:
            sys.stdout.write(out)
        return 0
    if args.cmd == "import-legacy":
        n = import_legacy_file(
            args.src,
            args.dest,
            run_id=args.run_id,
            correct_mtp6_hit_rate=args.correct_mtp6_hit_rate,
            prompt_type=args.prompt_type,
            correct_k=args.correct_summed_ratio,
            port=args.port,
        )
        print(f"imported {n} rows -> {args.dest}")
        return 0
    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

