"""cmd_agent command handler — extracted from cli.py."""

from __future__ import annotations

def execute(args) -> None:
    """
    Handle the 'agent' subcommand: NL intent → draft specs → submit to ControlPlane.
    """
    import re

    from orchestrator.engine_core.control_plane import ControlPlane
    from orchestrator.orchestration_agent import OrchestrationAgent
    from orchestrator.secure_execution import CommandInjectionError

    # SECURITY FIX: Validate intent input length and content
    intent = args.intent.strip()
    if len(intent) > 10000:
        print("ERROR: Intent description too long (max 10000 chars)", file=sys.stderr)
        sys.exit(1)

    # Basic check for potential injection patterns
    dangerous_patterns = [
        r"`.*?`",  # Backtick execution
        r"\$\(",  # Command substitution
        r"\$\{",  # Variable expansion
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, intent):
            print(
                "WARNING: Intent contains potentially dangerous characters", file=sys.stderr
            )  # Post-build setup: create venv, install deps
            print("  Setting up virtual environment...")
            print(f"  cd {output_dir}")
            print(f"  python -m venv venv")
            print(f"  venv\\Scripts\\activate")
            print(f"  pip install -e .")
            print(f"  python main.py\n")

            # Don't block, just warn - natural language can contain backticks

    agent = OrchestrationAgent()
    draft = asyncio.run(agent.draft(intent))

    print("\n=== Draft Job Spec ===")
    import json as _json
    from dataclasses import asdict

    print(_json.dumps(asdict(draft.job), indent=2, default=str))
    print("\n=== Draft Policy Spec ===")
    print(_json.dumps(asdict(draft.policy), indent=2, default=str))
    print(f"\nRationale: {draft.rationale}")

    if not args.interactive:
        return

    while True:
        try:
            feedback = input("\nFeedback (or 'submit' to run, 'quit' to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            return

        # SECURITY FIX: Validate feedback length
        if len(feedback) > 5000:
            print("ERROR: Feedback too long (max 5000 chars)", file=sys.stderr)
            continue

        if feedback.lower() == "quit":
            return
        if feedback.lower() == "submit":
            break

        # SECURITY FIX: Additional validation for feedback
        try:
            draft = asyncio.run(agent.refine(draft, feedback))
        except CommandInjectionError as e:
            print(f"ERROR: Security violation in feedback: {e}", file=sys.stderr)
            continue

        print("\n=== Revised Job Spec ===")
        print(_json.dumps(asdict(draft.job), indent=2, default=str))
        print(f"\nRationale: {draft.rationale}")

    print("\nSubmitting to ControlPlane...")
    cp = ControlPlane()
    state = asyncio.run(cp.submit(draft.job, draft.policy))
    print(f"Status: {state.status.value}")


def register(subparsers) -> None:
    """Register the 'agent' subcommand."""
    ap = subparsers.add_parser(
        "agent",
        help="Convert NL intent to typed specs and optionally run via ControlPlane",
    )
    ap.add_argument(
        "--intent",
        "-i",
        required=True,
        help="Natural language description of the job to run",
    )
    ap.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Enter interactive refine loop before submitting",
    )
    ap.set_defaults(func=cmd_agent)
