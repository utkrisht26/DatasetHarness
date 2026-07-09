"""``adh`` command-line entry point.

Subcommands:
  * ``adh validate <spec.yaml>``  — load + contract-check a Job Spec.
  * ``adh discover <spec.yaml>``  — run Source Discovery and print the ranked plan.
  * ``adh run <spec.yaml>``       — run Engine A end-to-end (with the approval gate).
  * ``adh build <spec.yaml>``     — run Engine B over an Engine A run dir -> dataset.

The CLI is policy on top of the :func:`adh.orchestration.run_engine_a` /
:func:`adh.orchestration.run_engine_b` mechanisms: it prompts a human at the approval
gate (unless ``--autonomous``/``--yes``) and wires the live channels + generation/judge
models when their optional extras are installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from adh.channels import DatasetsChannel, HfDatasetLoader, WebChannel
from adh.channels.web import Crawl4AIContentFetcher
from adh.contract import SourceChannel, check_job_spec, load_job_spec
from adh.llm import DEFAULT_CHAT_MODEL
from adh.logconfig import configure_logging
from adh.discovery import (
    ApprovalDecision,
    AutoApprover,
    CallbackApprover,
    SourceDiscoveryAgent,
    render_plan,
)
from adh.orchestration import run_engine_a, run_engine_b
import dotenv

dotenv.load_dotenv()

def _interactive_decision(plan) -> ApprovalDecision:
    print(render_plan(plan))
    print(
        "\nApprove sources to fetch: [a]ll / [n]one / comma-separated numbers "
        "(synthesis aspects always proceed unless you choose none):"
    )
    raw = input("> ").strip().lower()
    if raw in ("a", "all", ""):
        ids = frozenset(s.source_id for s in plan.sources)
        return ApprovalDecision(ids, approve_synthesis=True, reason="interactive: all")
    if raw in ("n", "none"):
        return ApprovalDecision(frozenset(), approve_synthesis=False, reason="interactive: none")
    chosen = set()
    for tok in raw.replace(" ", "").split(","):
        if tok.isdigit():
            i = int(tok) - 1
            if 0 <= i < len(plan.sources):
                chosen.add(plan.sources[i].source_id)
    return ApprovalDecision(frozenset(chosen), approve_synthesis=True, reason="interactive: subset")


def _build_live_channels() -> dict:
    """Wire real channels when their extras are importable; skip otherwise."""
    channels: dict = {}
    try:  # web channel (Crawl4AI)
        channels[SourceChannel.WEB] = WebChannel(fetcher=Crawl4AIContentFetcher())
    except Exception as exc:  # pragma: no cover
        print(f"[adh] web channel unavailable: {exc}", file=sys.stderr)
    try:  # datasets channel (HF datasets)
        channels[SourceChannel.DATASETS] = DatasetsChannel(loader=HfDatasetLoader())
    except Exception as exc:  # pragma: no cover
        print(f"[adh] datasets channel unavailable: {exc}", file=sys.stderr)
    return channels


def _cmd_validate(args: argparse.Namespace) -> int:
    spec = load_job_spec(args.spec)
    report = check_job_spec(spec, base_dir=Path(args.spec).resolve().parent.parent)
    print(f"Job spec {spec.name!r}: structural validation OK (tier {int(spec.verifiability_tier)})")
    for w in report.warnings:
        print(f"  warning: {w}")
    for e in report.errors:
        print(f"  ERROR: {e}")
    if not report.ok:
        return 1
    print("  filesystem checks OK")
    return 0


def _cmd_discover(args: argparse.Namespace) -> int:
    spec = load_job_spec(args.spec)
    work_dir = Path("./runs") / (spec.job_id or spec.name)
    plan = SourceDiscoveryAgent().discover(spec, work_dir=work_dir)
    print(render_plan(plan))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    spec = load_job_spec(args.spec)
    work_dir = args.work_dir or f"./runs/{spec.job_id or spec.name}"

    if args.autonomous or args.yes:
        approver = AutoApprover()
    elif spec.discovery.approval_gate:
        approver = CallbackApprover(_interactive_decision)
    else:
        approver = AutoApprover()

    channels = {} if args.no_fetch else _build_live_channels()
    try:
        result = run_engine_a(
            spec,
            work_dir=work_dir,
            approver=approver,
            channels=channels,
            resume=not args.fresh,
        )
    finally:
        # We created the live channels here, so we tear down their browsers here
        # (the web channel owns a background event loop + AsyncWebCrawler).
        for channel in channels.values():
            close = getattr(channel, "close", None)
            if callable(close):
                close()
    print(f"\nEngine A finished: exit_reason={result.exit_reason!r}")
    print(f"  corpus chunks:      {result.n_chunks}")
    print(f"  synthesis handoffs: {result.n_handoffs}")
    print(f"  metrics:            {result.metrics.to_dict()}")
    print(f"  manifest:           {result.manifest_path}")
    print(f"  lineage:            {result.lineage_path}")
    print(f"  summary:            {result.summary_path}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    """Run Engine B (synthesis + curation) over an Engine A run directory."""
    spec = load_job_spec(args.spec)
    work_dir = args.work_dir or f"./runs/{spec.job_id or spec.name}"
    if not Path(work_dir, "manifest.jsonl").exists():
        print(
            f"[adh] no Engine A output at {work_dir}/manifest.jsonl — run `adh run {args.spec}` first",
            file=sys.stderr,
        )
        return 1
    # record_schema_ref / decontaminate_against in the spec resolve against the repo
    # root (the spec lives in e.g. examples/), same base check_job_spec uses.
    base_dir = Path(args.spec).resolve().parent.parent
    result = run_engine_b(
        spec, work_dir=work_dir, resume=not args.fresh, schema_base_dir=base_dir
    )
    c = result.counts
    print(f"\nEngine B finished: exit_reason={result.exit_reason!r}")
    print(f"  records:      {result.n_records}  (from {c.get('generated', 0)} generated, "
          f"accept rate {c.get('accept_rate', 0.0):.1%})")
    print(f"  by split:     {result.delivery.counts_by_split}")
    print(f"  dropped:      quality={c.get('dropped_quality', 0)} verifier={c.get('dropped_verifier', 0)} "
          f"jury={c.get('dropped_jury', 0)} dup={c.get('duplicates', 0)} dead={c.get('dead_lettered', 0)}")
    print(f"  metrics:      {result.metrics.to_dict()}")
    print(f"  dataset:      {result.dataset_dir}  ({result.delivery.format})")
    if result.docs:
        print(f"  docs:         {', '.join(p.name for p in result.docs)}")
    if result.coverage.thin_aspects:
        print(f"  thin aspects (re-discover candidates): {list(result.coverage.thin_aspects)}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """Run the gateway service: crawl + runs routers under one origin (/api/*)."""
    try:
        import uvicorn

        from adh.service.app import create_app
    except ImportError as exc:  # pragma: no cover - needs the serve extra
        print(f"[adh] service needs the serve extra: pip install 'adh[serve]' ({exc})",
              file=sys.stderr)
        return 1
    app = create_app(
        runs_dir=args.runs_dir,
        discovery_model=None if args.no_llm else args.discovery_model,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _cmd_crawl_serve(args: argparse.Namespace) -> int:
    """Run the legacy standalone crawl service (deprecated: use ``adh serve``)."""
    try:
        import uvicorn

        from adh.channels.web.app import create_app
    except ImportError as exc:  # pragma: no cover - needs the serve extra
        print(f"[adh] crawl service needs the serve extra: pip install 'adh[serve]' ({exc})",
              file=sys.stderr)
        return 1
    print("[adh] crawl-serve is deprecated; use `adh serve` (crawl moved to /api/crawl/*)",
          file=sys.stderr)
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def _logging_parent() -> argparse.ArgumentParser:
    """Shared ``--log-level`` / ``-v`` / ``-q`` flags for every subcommand.

    A parent parser so the flags work *after* the subcommand (``adh serve -v``),
    which is where users reach for them.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--log-level",
        default=None,
        metavar="LEVEL",
        help="log level (debug/info/warning/error); default INFO or $ADH_LOG_LEVEL",
    )
    common.add_argument(
        "-v", "--verbose", action="store_true", help="verbose logging (DEBUG); shows tool results"
    )
    common.add_argument(
        "-q", "--quiet", action="store_true", help="quiet logging (WARNING and above only)"
    )
    return common


def _level_from_args(args: argparse.Namespace) -> str | None:
    """Resolve the log level: explicit --log-level > -v > -q > env/default."""
    if args.log_level:
        return args.log_level
    if args.verbose:
        return "DEBUG"
    if args.quiet:
        return "WARNING"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adh", description="Agentic Dataset Harness — Engine A (discover+acquire) + Engine B (synthesize+curate)"
    )
    common = _logging_parent()
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser(
        "validate", parents=[common], help="validate a Job Spec against the contract"
    )
    p_val.add_argument("spec", help="path to a Job Spec YAML")
    p_val.set_defaults(func=_cmd_validate)

    p_disc = sub.add_parser(
        "discover", parents=[common], help="run source discovery and print the plan"
    )
    p_disc.add_argument("spec")
    p_disc.set_defaults(func=_cmd_discover)

    p_run = sub.add_parser("run", parents=[common], help="run Engine A end-to-end")
    p_run.add_argument("spec")
    p_run.add_argument("--work-dir", default=None, help="output directory (default ./runs/<job>)")
    p_run.add_argument("--yes", action="store_true", help="approve all discovered sources")
    p_run.add_argument("--autonomous", action="store_true", help="run without the approval gate")
    p_run.add_argument("--fresh", action="store_true", help="ignore prior state (no resume)")
    p_run.add_argument(
        "--no-fetch",
        action="store_true",
        help="discovery + approval only; configure no acquisition channels",
    )
    p_run.set_defaults(func=_cmd_run)

    p_build = sub.add_parser(
        "build", parents=[common], help="run Engine B over an Engine A run dir -> dataset"
    )
    p_build.add_argument("spec")
    p_build.add_argument("--work-dir", default=None, help="the Engine A run dir (default ./runs/<job>)")
    p_build.add_argument("--fresh", action="store_true", help="ignore prior Engine B state (no resume)")
    p_build.set_defaults(func=_cmd_build)

    p_serve = sub.add_parser(
        "serve", parents=[common], help="run the gateway service: /api/crawl/* + /api/runs/*"
    )
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--runs-dir", default="./runs", help="base directory for run dirs")
    p_serve.add_argument(
        "--discovery-model",
        default=DEFAULT_CHAT_MODEL,
        help=f"chat model id for deep discovery (default: {DEFAULT_CHAT_MODEL})",
    )
    p_serve.add_argument(
        "--no-llm",
        action="store_true",
        help="heuristic-only discovery (no model calls)",
    )
    p_serve.set_defaults(func=_cmd_serve)

    p_crawl_serve = sub.add_parser(
        "crawl-serve",
        parents=[common],
        help="run the legacy standalone crawl service (deprecated: use serve)",
    )
    p_crawl_serve.add_argument("--host", default="127.0.0.1")
    p_crawl_serve.add_argument("--port", type=int, default=8000)
    p_crawl_serve.set_defaults(func=_cmd_crawl_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Install the stderr trace before dispatch so discovery/acquisition/build all
    # log (deep-agent tool + model turns, crawler progress, Engine B drops).
    configure_logging(_level_from_args(args))
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
