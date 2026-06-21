"""cmd_analyze command handler — extracted from cli.py."""

from __future__ import annotations

def execute(args) -> None:
    """
    Handle the 'analyze' subcommand: read a codebase and produce an analysis report.

    Uses CodebaseReader to scan files and CodebaseAnalyzer to run multi-LLM analysis.
    """
    from pathlib import Path

    from orchestrator.analyzer import CodebaseAnalyzer
    from orchestrator.secure_execution import InputValidator

    # SECURITY FIX: Validate input path to prevent path traversal
    try:
        # Sanitize and resolve path
        base_path = Path.cwd()
        path = (base_path / args.path).resolve()

        # Verify path is within allowed base directory
        try:
            path.relative_to(base_path)
        except ValueError:
            print(f"ERROR: Path traversal detected: {args.path}", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: Invalid path: {e}", file=sys.stderr)
        sys.exit(1)

    if not path.exists():
        print(f"ERROR: Path does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    # SECURITY FIX: Validate focus areas and extensions
    focus = None
    if args.focus:
        focus = []
        for f in args.focus.split(","):
            f = f.strip()
            if f:  # Only add non-empty values
                # Basic validation: alphanumeric and common separators only
                if not re.match(r"^[\w\s\-_,]+$", f):
                    print(f"ERROR: Invalid focus area: {f}", file=sys.stderr)
                    sys.exit(1)
                focus.append(f)

    include_exts = None
    if args.extensions:
        include_exts = set()
        for ext in args.extensions.split(","):
            ext = ext.strip()
            if ext:
                # Validate extension format
                if not re.match(r"^[\w.]+$", ext):
                    print(f"ERROR: Invalid extension: {ext}", file=sys.stderr)
                    sys.exit(1)
                # Ensure extension starts with dot
                if not ext.startswith("."):
                    ext = f".{ext}"
                include_exts.add(ext)

    analyzer = CodebaseAnalyzer(
        max_context_tokens=args.context_tokens,
        max_concurrency=args.concurrency,
    )

    print(f"Analyzing: {path}")
    if focus:
        print(f"Focus areas: {', '.join(focus)}")
    print(f"Budget: ${args.budget:.2f} | Context limit: {args.context_tokens:,} tokens")
    print("-" * 60)

    report = asyncio.run(
        analyzer.analyze(
            path=path,
            focus=focus,
            budget_usd=args.budget,
            include_exts=include_exts,
            max_tokens_per_section=args.section_tokens,
        )
    )

    # Print summary
    print(
        f"\nAnalysis complete: {len(report.sections)} sections | "
        f"${report.total_cost:.4f} | {report.elapsed_s:.1f}s"
    )
    print(
        f"Files analyzed: {report.files_analyzed} | "
        f"Languages: {', '.join(sorted(report.languages))}"
    )

    # Write report
    # SECURITY FIX: Validate output path
    if args.output:
        output_path = Path(args.output).resolve()
        # Ensure output path is safe (not traversing outside working dir)
        try:
            output_path.relative_to(Path.cwd())
        except ValueError:
            print("ERROR: Output path must be within current directory", file=sys.stderr)
            sys.exit(1)
        # Validate filename
        safe_filename = InputValidator.sanitize_filename(output_path.name)
        if safe_filename != output_path.name:
            print(f"WARNING: Output filename sanitized to: {safe_filename}", file=sys.stderr)
            output_path = output_path.parent / safe_filename
    else:
        output_path = path / "ANALYSIS_REPORT.md"

    output_path.write_text(report.markdown, encoding="utf-8")
    print(f"\nReport written to: {output_path}")

    # Print preview
    if not args.quiet:
        print("\n" + "=" * 60)
        preview = report.markdown[:2000]
        print(preview)
        if len(report.markdown) > 2000:
            print(f"\n... ({len(report.markdown):,} chars total — see {output_path})")


def register(subparsers) -> None:
    """Register the 'analyze' subcommand on the given subparsers action."""
    ap = subparsers.add_parser(
        "analyze",
        help="Analyze a codebase and produce an improvement report",
    )
    ap.add_argument(
        "--path",
        "-p",
        required=True,
        help="Root directory of the codebase to analyze",
    )
    ap.add_argument(
        "--focus",
        "-f",
        default="",
        help=(
            "Comma-separated focus areas: architecture, quality, security, "
            "performance, improvements (default: all)"
        ),
    )
    ap.add_argument(
        "--extensions",
        "-e",
        default="",
        help="Comma-separated file extensions to include (e.g. .py,.ts). Default: all code files",
    )
    ap.add_argument(
        "--budget",
        "-b",
        type=float,
        default=3.0,
        help="Max API spend in USD (default: 3.0)",
    )
    ap.add_argument(
        "--context-tokens",
        dest="context_tokens",
        type=int,
        default=60_000,
        help="Max tokens for the codebase context passed to each LLM (default: 60000)",
    )
    ap.add_argument(
        "--section-tokens",
        dest="section_tokens",
        type=int,
        default=4096,
        help="Max output tokens per analysis section (default: 4096)",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Max simultaneous API calls (default: 2)",
    )
    ap.add_argument(
        "--output",
        "-o",
        default="",
        help="Output file path for the report (default: <path>/ANALYSIS_REPORT.md)",
    )
    ap.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Suppress report preview in terminal",
    )
    ap.set_defaults(func=cmd_analyze)
