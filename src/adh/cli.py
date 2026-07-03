"""``adh`` command-line entry point.

Subcommands:
  * ``adh validate <spec.yaml>``  — load + contract-check a Job Spec.
  * ``adh discover <spec.yaml>``  — run Source Discovery and print the ranked plan.
  * ``adh run <spec.yaml>``       — run Engine A end-to-end (with the approval gate).

The CLI is policy on top of the :func:`adh.orchestration.run_engine_a` mechanism: it
prompts a human at the approval gate (unless ``--autonomous``/``--yes``) and wires the
live channels when their optional extras are installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from adh.channels import DatasetsChannel, HfDatasetLoader, WebChannel
from adh.channels.web import Crawl4AIContentFetcher
from adh.contract import SourceChannel, check_job_spec, load_job_spec
from adh.discovery import (
    ApprovalDecision,
    AutoApprover,
    CallbackApprover,
    SourceDiscoveryAgent,
    render_plan,
)
from adh.orchestration import run_engine_a


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
    plan = SourceDiscoveryAgent().discover(spec)
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


def _cmd_crawl_serve(args: argparse.Namespace) -> int:
    """Run the crawl microservice (the web channel behind an HTTP boundary)."""
    try:
        import uvicorn

        from adh.channels.web.app import create_app
    except ImportError as exc:  # pragma: no cover - needs the serve extra
        print(f"[adh] crawl service needs the serve extra: pip install 'adh[serve]' ({exc})",
              file=sys.stderr)
        return 1
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adh", description="Agentic Dataset Harness — Engine A")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="validate a Job Spec against the contract")
    p_val.add_argument("spec", help="path to a Job Spec YAML")
    p_val.set_defaults(func=_cmd_validate)

    p_disc = sub.add_parser("discover", help="run source discovery and print the plan")
    p_disc.add_argument("spec")
    p_disc.set_defaults(func=_cmd_discover)

    p_run = sub.add_parser("run", help="run Engine A end-to-end")
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

    p_serve = sub.add_parser("crawl-serve", help="run the crawl microservice (adh[serve])")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=_cmd_crawl_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
