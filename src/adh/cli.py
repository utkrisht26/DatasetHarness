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
import json
import sys
from pathlib import Path

from adh.contract import check_job_spec, load_job_spec
from adh.delivery import (
    NotPublishableError,
    PublishRefusedError,
    assert_publishable,
    default_hub_client,
    evaluate_dataset,
    publish_dataset,
)
from adh.llm import UsageTracker
from adh.logconfig import configure_logging
from adh.discovery import (
    ApprovalDecision,
    AutoApprover,
    CallbackApprover,
    render_plan,
)
from adh.generation import HeuristicGenerator
from adh.orchestration import run_coverage_loop, run_engine_a, run_engine_b
from adh.verification import HeuristicJudge
from adh.runtime import live_agent_factory, live_channels
from adh.settings import DISCOVERY_MODEL_ENV, get_settings, load_dotenv_file


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


def _discovery_model_from_args(args: argparse.Namespace) -> str | None:
    """The configured discovery model id, or None for heuristic-only discovery."""
    return None if getattr(args, "no_llm", False) else getattr(args, "discovery_model", None)


def _cmd_discover(args: argparse.Namespace) -> int:
    spec = load_job_spec(args.spec)
    work_dir = Path("./runs") / (spec.job_id or spec.name)
    usage = UsageTracker()
    agent = live_agent_factory(_discovery_model_from_args(args))(usage)
    plan = agent.discover(spec, work_dir=work_dir)
    print(render_plan(plan))
    print(f"\ndiscovery LLM spend: {usage.total.total_tokens} token(s)")
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

    usage = UsageTracker()
    agent = live_agent_factory(_discovery_model_from_args(args))(usage)

    channels = {} if args.no_fetch else live_channels()
    try:
        result = run_engine_a(
            spec,
            work_dir=work_dir,
            approver=approver,
            channels=channels,
            resume=not args.fresh,
            usage=usage,
            agent=agent,
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
    print(f"  discovery tokens:   {usage.total.total_tokens}")
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
    result = run_engine_b(spec, work_dir=work_dir, resume=not args.fresh, schema_base_dir=base_dir)
    c = result.counts
    print(f"\nEngine B finished: exit_reason={result.exit_reason!r}")
    print(
        f"  records:      {result.n_records}  (from {c.get('generated', 0)} generated, "
        f"accept rate {c.get('accept_rate', 0.0):.1%})"
    )
    print(f"  by split:     {result.delivery.counts_by_split}")
    print(
        f"  dropped:      quality={c.get('dropped_quality', 0)} verifier={c.get('dropped_verifier', 0)} "
        f"jury={c.get('dropped_jury', 0)} dup={c.get('duplicates', 0)} dead={c.get('dead_lettered', 0)}"
    )
    print(f"  skipped:      {c.get('skipped_grounding', 0)} seed(s) with non-substantive grounding")
    print(f"  metrics:      {result.metrics.to_dict()}")
    print(f"  dataset:      {result.dataset_dir}  ({result.delivery.format})")
    if result.docs:
        print(f"  docs:         {', '.join(p.name for p in result.docs)}")
    if result.coverage.thin_aspects:
        print(f"  thin aspects (re-discover candidates): {list(result.coverage.thin_aspects)}")
    return 0


def _cmd_publish(args: argparse.Namespace) -> int:
    """Quality-gate a built dataset, then upload it to the Hub (private by default)."""
    spec = load_job_spec(args.spec)
    work_dir = args.work_dir or f"./runs/{spec.job_id or spec.name}"
    dataset_dir = Path(work_dir) / "dataset"
    if not dataset_dir.is_dir():
        print(
            f"[adh] no dataset at {dataset_dir} — run `adh build {args.spec}` first",
            file=sys.stderr,
        )
        return 1

    # record_schema_ref / decontaminate_against resolve against the repo root
    # (the spec lives in e.g. examples/), the same rule _cmd_build/curation use.
    base_dir = Path(args.spec).resolve().parent.parent
    try:
        schema: dict | None = None
        if spec.record_schema_ref:
            schema_ref = Path(spec.record_schema_ref)
            if not schema_ref.is_absolute():
                schema_ref = base_dir / schema_ref
            schema = json.loads(schema_ref.read_text(encoding="utf-8"))
        references = None
        if spec.decontaminate_against:
            references = [
                ref if Path(ref).is_absolute() else base_dir / ref
                for ref in spec.decontaminate_against
            ]

        report = evaluate_dataset(dataset_dir, schema=schema, references=references)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[adh] could not evaluate {dataset_dir}: {exc}", file=sys.stderr)
        return 1

    print(f"Quality gate for {dataset_dir}:")
    print(f"  records:      {report.total_records}  ({report.counts_by_split})")
    print(f"  duplicate rate: {report.duplicate_rate:.1%}")
    print(
        f"  schema pass:  {report.schema_validation_pass_rate:.1%}  "
        f"contamination residuals: {report.contamination_residual_count}  "
        f"pii residuals: {report.pii_residual_count}"
    )
    if getattr(report, "unknown_license_assets", 0):
        print(
            f"  unknown-license assets: {report.unknown_license_assets}  "
            "(public upload requires --allow-unknown-licenses)"
        )
    for warning in report.warnings:
        print(f"  warning:     {warning}")
    try:
        assert_publishable(report, min_records=args.min_records)
    except NotPublishableError as exc:
        for reason in exc.reasons:
            print(f"  BLOCKED: {reason}", file=sys.stderr)
        print(
            "[adh] refusing to publish — fix the gate failures above or adjust --min-records",
            file=sys.stderr,
        )
        return 1

    try:
        result = publish_dataset(
            dataset_dir,
            repo_id=args.repo_id,
            license=spec.output_license,
            private=not args.public,
            client=None if args.dry_run else default_hub_client(),
            dry_run=args.dry_run,
            allow_unknown_licensed_assets=args.allow_unknown_licenses,
            task_categories=list(spec.task_categories),
            languages=list(spec.languages),
            commit_message=args.message
            or "Publish dataset generated by the Agentic Dataset Harness",
        )
    except PublishRefusedError as exc:
        # typed refusal: the two-layer license policy's distribution consent gate
        print(f"[adh] {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:  # e.g. missing 'huggingface_hub' extra
        print(f"[adh] publishing unavailable: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # auth/network failures mid-upload: clean CLI error
        print(f"[adh] upload failed: {exc}", file=sys.stderr)
        return 1
    access = "public" if args.public else "private"
    if result.dry_run:
        print(
            f"\nDry run: would upload {len(result.uploads)} file(s) to {args.repo_id} ({access}):"
        )
        for item in result.uploads:
            print(f"  {item.repo_path}")
    else:
        print(f"\nPublished {len(result.uploads)} file(s) to {args.repo_id} ({access})")
        print(f"  repo: {result.url}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """Run the gateway service: crawl + runs routers under one origin (/api/*)."""
    try:
        import uvicorn

        from adh.service.app import create_app
        from adh.service.security import API_KEY_ENV, is_loopback_host
    except ImportError as exc:  # pragma: no cover - needs the serve extra
        print(
            f"[adh] service needs the serve extra: pip install 'adh[serve]' ({exc})",
            file=sys.stderr,
        )
        return 1
    api_key = get_settings().service_api_key_value
    if (
        not api_key
        and not is_loopback_host(args.host)
        and not getattr(args, "allow_unauthenticated", False)
    ):
        print(
            f"[adh] refusing unauthenticated service on {args.host!r}; set {API_KEY_ENV} "
            "or pass --allow-unauthenticated to acknowledge the risk",
            file=sys.stderr,
        )
        return 2
    app = create_app(
        runs_dir=args.runs_dir,
        discovery_model=None if args.no_llm else args.discovery_model,
        api_key=api_key,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _cmd_crawl_serve(args: argparse.Namespace) -> int:
    """Run the legacy standalone crawl service (deprecated: use ``adh serve``)."""
    try:
        import uvicorn

        from adh.channels.web.app import create_app
    except ImportError as exc:  # pragma: no cover - needs the serve extra
        print(
            f"[adh] crawl service needs the serve extra: pip install 'adh[serve]' ({exc})",
            file=sys.stderr,
        )
        return 1
    print(
        "[adh] crawl-serve is deprecated; use `adh serve` (crawl moved to /api/crawl/*)",
        file=sys.stderr,
    )
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
        "-v",
        "--verbose",
        action="store_true",
        help="verbose logging (DEBUG); shows tool results",
    )
    common.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="quiet logging (WARNING and above only)",
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


def _cmd_loop(args: argparse.Namespace) -> int:
    """Run the coverage-gap loop (build -> re-discover thin aspects -> rebuild)."""
    spec = load_job_spec(args.spec)
    work_dir = args.work_dir or f"./runs/{spec.job_id or spec.name}"
    if not Path(work_dir, "manifest.jsonl").exists():
        print(
            f"[adh] no Engine A output at {work_dir}/manifest.jsonl — run `adh run {args.spec}` first",
            file=sys.stderr,
        )
        return 1

    if args.autonomous or args.yes:
        approver = AutoApprover()
    elif spec.discovery.approval_gate:
        approver = CallbackApprover(_interactive_decision)
    else:
        approver = AutoApprover()

    usage = UsageTracker()
    agent = live_agent_factory(_discovery_model_from_args(args))(usage)
    channels = {} if getattr(args, "no_fetch", False) else live_channels()
    base_dir = Path(args.spec).resolve().parent.parent
    # --no-llm promises zero model calls: pin the deterministic collaborators so
    # neither the generator ladder nor the judge panel probes provider credentials.
    offline = getattr(args, "no_llm", False)
    try:
        outcome = run_coverage_loop(
            spec,
            work_dir=work_dir,
            agent=agent,
            channels=channels,
            approver=approver,
            generator=HeuristicGenerator() if offline else None,
            judge=HeuristicJudge() if offline else None,
            usage=usage,
            max_rounds=args.max_rounds,
            schema_base_dir=base_dir,
        )
    finally:
        for channel in channels.values():
            close = getattr(channel, "close", None)
            if callable(close):
                close()

    r = outcome.result
    c = r.counts
    print(f"\nCoverage loop finished: exit_reason={outcome.exit_reason!r} rounds={outcome.rounds}")
    print(f"  records:      {r.n_records}  (accept rate {c.get('accept_rate', 0.0):.1%})")
    print(f"  thin aspects: {list(outcome.thin_aspects) or 'none'}")
    if outcome.acquisitions:
        print(f"  acquisitions: {[a.exit_reason for a in outcome.acquisitions]}")
    for note in outcome.notes:
        print(f"  note:         {note}")
    print(f"  metrics:      {r.metrics.to_dict()}")
    print(f"  dataset:      {r.dataset_dir}  ({r.delivery.format})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    default_discovery_model = get_settings().discovery_model
    parser = argparse.ArgumentParser(
        prog="adh",
        description="Agentic Dataset Harness — Engine A (discover+acquire) + Engine B (synthesize+curate)",
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
    p_disc.add_argument(
        "--discovery-model",
        default=default_discovery_model,
        help=(
            "chat model id for deep discovery "
            f"(default: {default_discovery_model}; ${DISCOVERY_MODEL_ENV})"
        ),
    )
    p_disc.add_argument(
        "--no-llm",
        action="store_true",
        help="heuristic-only discovery (no model calls)",
    )
    p_disc.set_defaults(func=_cmd_discover)

    p_run = sub.add_parser("run", parents=[common], help="run Engine A end-to-end")
    p_run.add_argument("spec")
    p_run.add_argument("--work-dir", default=None, help="output directory (default ./runs/<job>)")
    p_run.add_argument("--yes", action="store_true", help="approve all discovered sources")
    p_run.add_argument("--autonomous", action="store_true", help="run without the approval gate")
    p_run.add_argument("--fresh", action="store_true", help="ignore prior state (no resume)")
    p_run.add_argument(
        "--discovery-model",
        default=default_discovery_model,
        help=(
            "chat model id for deep discovery "
            f"(default: {default_discovery_model}; ${DISCOVERY_MODEL_ENV})"
        ),
    )
    p_run.add_argument(
        "--no-llm",
        action="store_true",
        help="heuristic-only discovery (no model calls)",
    )
    p_run.add_argument(
        "--no-fetch",
        action="store_true",
        help="discovery + approval only; configure no acquisition channels",
    )
    p_run.set_defaults(func=_cmd_run)

    p_build = sub.add_parser(
        "build",
        parents=[common],
        help="run Engine B over an Engine A run dir -> dataset",
    )
    p_build.add_argument("spec")
    p_build.add_argument(
        "--work-dir", default=None, help="the Engine A run dir (default ./runs/<job>)"
    )
    p_build.add_argument(
        "--fresh", action="store_true", help="ignore prior Engine B state (no resume)"
    )
    p_build.set_defaults(func=_cmd_build)

    p_loop = sub.add_parser(
        "loop",
        parents=[common],
        help="coverage-gap loop: build, re-discover thin aspects, re-acquire, rebuild",
    )
    p_loop.add_argument("spec")
    p_loop.add_argument(
        "--work-dir", default=None, help="the Engine A run dir (default ./runs/<job>)"
    )
    p_loop.add_argument(
        "--max-rounds",
        type=int,
        default=2,
        help="re-discovery round budget after the initial build (default 2)",
    )
    p_loop.add_argument("--yes", action="store_true", help="approve re-discovered sources")
    p_loop.add_argument("--autonomous", action="store_true", help="run without the approval gate")
    # no --fresh here on purpose: the loop always resumes durable state (that
    # continuity is its whole point); use `adh run --fresh` to restart Engine A.
    p_loop.add_argument(
        "--discovery-model",
        default=default_discovery_model,
        help=(
            "chat model id for deep discovery "
            f"(default: {default_discovery_model}; ${DISCOVERY_MODEL_ENV})"
        ),
    )
    p_loop.add_argument(
        "--no-llm",
        action="store_true",
        help="heuristic discovery agent (offline; no model calls)",
    )
    p_loop.add_argument(
        "--no-fetch",
        action="store_true",
        help="skip acquisition channels (plan + synthesis only)",
    )
    p_loop.set_defaults(func=_cmd_loop)

    p_pub = sub.add_parser(
        "publish",
        parents=[common],
        help="quality-gate a built dataset, then upload it to the Hub (private by default)",
    )
    p_pub.add_argument("spec")
    p_pub.add_argument(
        "--work-dir",
        default=None,
        help="the run dir holding dataset/ (default ./runs/<job>)",
    )
    p_pub.add_argument("--repo-id", required=True, help="target Hub dataset id (org/name)")
    p_pub.add_argument(
        "--public",
        action="store_true",
        help="make the dataset public (default: private)",
    )
    p_pub.add_argument(
        "--allow-unknown-licenses",
        action="store_true",
        help="allow a PUBLIC upload while assets carry unknown licenses "
        "(two-layer license policy override; private uploads never need it)",
    )
    p_pub.add_argument(
        "--dry-run",
        action="store_true",
        help="plan the upload without touching the Hub",
    )
    p_pub.add_argument(
        "--min-records",
        type=int,
        default=1,
        help="refuse to publish below this record count (default: 1)",
    )
    p_pub.add_argument("--message", default=None, help="upload commit message")
    p_pub.set_defaults(func=_cmd_publish)

    p_serve = sub.add_parser(
        "serve",
        parents=[common],
        help="run the gateway service: /api/crawl/* + /api/runs/*",
    )
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--runs-dir", default="./runs", help="base directory for run dirs")
    p_serve.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help="allow a non-loopback bind without ADH_API_KEY (unsafe; explicit opt-out)",
    )
    p_serve.add_argument(
        "--discovery-model",
        default=default_discovery_model,
        help=(
            "chat model id for deep discovery "
            f"(default: {default_discovery_model}; ${DISCOVERY_MODEL_ENV})"
        ),
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
    # Load the repo .env here, not at import time: an import must have no side
    # effects (importing adh.cli from tests or tools otherwise silently injects
    # real API keys into os.environ for the whole process).
    load_dotenv_file()
    parser = build_parser()
    args = parser.parse_args(argv)
    # Install the stderr trace before dispatch so discovery/acquisition/build all
    # log (deep-agent tool + model turns, crawler progress, Engine B drops).
    configure_logging(_level_from_args(args))
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
