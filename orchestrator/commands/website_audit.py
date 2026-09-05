"""`website-audit` — run WF-100 against a build directory or a live URL.

    website-audit ./outputs/kalamaria
    website-audit https://kalamaria-dental.gr --record clients/kalamaria.yaml
    website-audit ./outputs/kalamaria --format markdown -o report.md

Exit codes are meant for a pipeline gate, and the two failure modes are kept
apart on purpose:

    0  launch      — score >= 90, no critical failure, no unverified critical check
    1  no launch   — something is broken, or the score cannot reach the threshold
    2  pending     — nothing is broken; verification is still owed

A build that exits 2 is not a build that failed. It is a build waiting on the
checks a tool cannot make: a person tabbing through the site, a clinician
approving the medical copy, field data from real visitors.
"""

from __future__ import annotations

import sys
from pathlib import Path


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "website-audit",
        help="Audit a site against the WF-100 quality standard (100 checks, 8 categories)",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="",
        help="Build directory, or a URL to audit live",
    )
    parser.add_argument(
        "--catalogue",
        action="store_true",
        help="Print the WF-100 standard itself (all 100 checks) and exit",
    )
    parser.add_argument(
        "--record",
        "-r",
        default="",
        help="Business record (YAML/JSON) to check the published name, address and phone against",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=("text", "markdown", "json"),
        default="text",
        help="text for the terminal, markdown for the client, json for a pipeline",
    )
    parser.add_argument("--output", "-o", default="", help="Write the report to a file")
    parser.add_argument(
        "--max-pages", type=int, default=25, help="Page cap when crawling a URL (default 25)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Explain each outstanding check in the terminal",
    )
    parser.add_argument(
        "--fail-on",
        choices=("critical", "threshold", "outstanding"),
        default="threshold",
        help=(
            "What makes this command exit non-zero: critical failures only, the launch "
            "threshold (default), or any unresolved check"
        ),
    )
    parser.set_defaults(func=execute)


def _load_record(path: Path):
    from ..generators.wf100.evidence import BusinessRecord

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text) or {}
    else:
        import json

        data = json.loads(text)
    # A client file written for `website-template apply` carries the same facts
    # under the same names, so one file can drive both commands.
    return BusinessRecord.from_dict(data)


def execute(args) -> None:
    from ..generators.wf100.auditor import audit_directory, audit_url
    from ..generators.wf100.render import catalogue_markdown

    if args.catalogue:
        text = catalogue_markdown()
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(f"Wrote the WF-100 catalogue to {out}")
        else:
            print(text)
        raise SystemExit(0)

    if not args.target:
        print("error: give a build directory or a URL to audit", file=sys.stderr)
        raise SystemExit(2)

    from ..generators.wf100.render import render_json, render_markdown, render_text
    from ..generators.wf100.report import Verdict

    record = None
    if args.record:
        record_path = Path(args.record)
        if not record_path.is_file():
            print(f"error: no business record at {record_path}", file=sys.stderr)
            raise SystemExit(2)
        try:
            record = _load_record(record_path)
        except Exception as exc:  # noqa: BLE001 - surface the parse error plainly
            print(f"error: could not read {record_path}: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc

    target = str(args.target)
    try:
        if target.startswith(("http://", "https://")):
            report = audit_url(target, record=record, max_pages=args.max_pages)
        else:
            report = audit_directory(target, record=record)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    rendered = {
        "text": lambda: render_text(report, verbose=args.verbose),
        "markdown": lambda: render_markdown(report),
        "json": lambda: render_json(report),
    }[args.format]()

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.format} report to {out}")
        print(
            f"{report.verdict.headline} — {report.score:g}/100, "
            f"{report.outstanding_points} point(s) outstanding"
        )
    else:
        print(rendered)

    raise SystemExit(_exit_code(report, args.fail_on, Verdict))


def _exit_code(report, fail_on: str, verdict_enum) -> int:
    """Map the verdict to an exit code under the caller's chosen strictness."""
    if report.blockers or report.critical_failures():
        return 1
    if fail_on == "critical":
        return 0
    if fail_on == "outstanding":
        return 0 if report.verdict is verdict_enum.LAUNCH and not report.outstanding() else 2
    return {verdict_enum.LAUNCH: 0, verdict_enum.NO_LAUNCH: 1, verdict_enum.PENDING: 2}[
        report.verdict
    ]
