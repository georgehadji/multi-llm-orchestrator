"""
Batch website generation command — the factory front-end.

Builds every site in a manifest, gates each one, and delivers each as its own
git repository. Designed to be run unattended: a single bad site is a row in
the report, not the end of the run, and the process exit code tells CI whether
anything needs a human.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def register(subparsers) -> None:
    """Register the 'website-batch' subcommand."""
    bp = subparsers.add_parser(
        "website-batch",
        help="Generate many websites from a manifest, gate each, deliver as git repos",
    )
    bp.add_argument("manifest", help="Path to the batch manifest (.yaml/.yml/.json)")
    bp.add_argument(
        "--output-root",
        "-o",
        default="outputs/factory",
        help="Directory that will contain one sub-directory per site",
    )
    bp.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=3,
        help="How many sites to build at once (provider rate limits and spend scale with this)",
    )
    bp.add_argument(
        "--no-git",
        action="store_true",
        default=False,
        help="Skip per-site git repo initialisation and commit",
    )
    bp.add_argument(
        "--report-name",
        default="factory-report.json",
        help="Filename for the batch report written under --output-root",
    )
    bp.set_defaults(func=execute)


def exit_code_for(outcomes) -> int:
    """Worst outcome wins: 1 (build failed) beats 2 (not launch-ready) beats 0.

    A build failure is worse news than a rejection: a rejected site exists and
    can be inspected, a failed one may not exist at all.

    Exit 0 needs every site to be WF-100 ``launch``. Pending and not_audited
    both exit 2, because the operator's next move is the same for all three —
    go and look — and a batch that exits 0 has said "ship it".
    """
    if any(not o.success for o in outcomes):
        return 1
    if any(o.quality_gate_passed is False for o in outcomes):
        return 2
    if any(o.verdict not in ("launch", "") for o in outcomes):
        return 2
    return 0


def execute(args) -> None:
    """Load the manifest, run the batch, print a summary, exit with the verdict."""
    from ..generators.website_factory import WebsiteFactory, load_manifest

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"[FAIL] manifest not found: {manifest_path}")
        sys.exit(1)

    try:
        specs = load_manifest(manifest_path)
    except (ValueError, OSError) as exc:
        print(f"[FAIL] invalid manifest: {exc}")
        sys.exit(1)

    output_root = Path(args.output_root)
    print(f"\n>>> Website factory: {len(specs)} site(s) -> {output_root}")
    print(f"    Concurrency: {args.concurrency}   Git delivery: {not args.no_git}")

    factory = WebsiteFactory(
        concurrency=args.concurrency,
        git_delivery=not args.no_git,
    )
    outcomes = asyncio.run(
        factory.run(specs, output_root=output_root, report_name=args.report_name)
    )

    _print_summary(outcomes, output_root / args.report_name)
    sys.exit(exit_code_for(outcomes))


def _print_summary(outcomes, report_path: Path) -> None:
    print("\n--- Factory results ---")
    for outcome in outcomes:
        if not outcome.success:
            status = "BUILD FAILED"
        elif outcome.quality_gate_passed is False or outcome.verdict == "no_launch":
            status = "REJECTED"
        elif outcome.verdict == "launch":
            status = "LAUNCH"
        elif outcome.verdict == "pending":
            status = "pending"
        elif outcome.verdict == "not_audited":
            status = "not audited"
        else:
            status = "ungated"
        # Two numbers, because they are two claims: the build-time validator's
        # 0-1 score, and WF-100's percentage with the ceiling it could reach if
        # every outstanding check were verified.
        wf100 = (
            f"  wf100 {outcome.wf100_score:.0f}/{outcome.wf100_ceiling:.0f}"
            if outcome.verdict not in ("", "not_audited")
            else ""
        )
        print(
            f"  {outcome.slug:28s} {status:13s} "
            f"score {outcome.score:.2f}{wf100}  ${outcome.cost_usd:.4f}  "
            f"{outcome.duration_seconds:.1f}s" + ("  [committed]" if outcome.committed else "")
        )
        for failure in outcome.failures:
            print(f"      - {failure}")
        if outcome.error:
            print(f"      ! {outcome.error}")

    launch = sum(1 for o in outcomes if o.verdict == "launch")
    pending = sum(1 for o in outcomes if o.verdict == "pending")
    rejected = sum(
        1 for o in outcomes if o.quality_gate_passed is False or o.verdict == "no_launch"
    )
    failed = sum(1 for o in outcomes if not o.success)
    total_cost = sum(o.cost_usd for o in outcomes)
    print(
        f"\n{len(outcomes)} site(s): {launch} launch-ready, {pending} pending verification, "
        f"{rejected} rejected, {failed} build failure(s).  Total cost: ${total_cost:.4f}"
    )
    if pending:
        print("Pending sites are built and clean — run `website-audit <dir>` for what is open.")
    print(f"Report: {report_path}")
