"""
PreToolUse hook: remind Claude of architecture rules when editing core modules.

Fires before Edit or Write on:
  - engine.py      → R2: Mediator rule (no business logic)
  - models.py      → R2: Pure data rule (no behavior, no I/O)
  - __init__.py    → reminder to follow export tier conventions

Outputs additionalContext so Claude sees the rule before proceeding.
Does NOT block the edit (exit 0 always) — advisory only.
"""
import json
import os
import sys

# Rules keyed by filename substring
RULES: dict[str, str] = {
    "engine.py": (
        "ARCHITECTURE RULE R2 — engine.py is the Mediator:\n"
        "• Add ONLY wiring code (self._service = NewService(...))\n"
        "• Business logic belongs in a NEW module, not here\n"
        "• If you're adding more than 5 lines of logic → stop and create a separate file\n"
        "• Reference: docs/ARCHITECTURE_ROADMAP.md Section 2, Κανόνας 1"
    ),
    "models.py": (
        "ARCHITECTURE RULE R2 — models.py is Pure Data:\n"
        "• ONLY dataclasses and enums — no methods with behavior\n"
        "• No I/O, no asyncio, no imports beyond stdlib\n"
        "• If adding a method: ask 'does this read/write data or compute something?'\n"
        "  If compute → belongs in a service. If just property access → OK.\n"
        "• Reference: docs/ARCHITECTURE_ROADMAP.md Section 2, Κανόνας 2"
    ),
    "__init__.py": (
        "ARCHITECTURE NOTE — orchestrator/__init__.py export tiers:\n"
        "• Tier 1 (Core): always-available imports (models, engine)\n"
        "• Tier 2+: wrap in try/except ImportError for optional features\n"
        "• Add new class to __all__ list\n"
        "• Pattern: try: from .<module> import <Class> / HAS_<MODULE> = True\n"
        "• Reference: docs/ARCHITECTURE_ROADMAP.md Section 3"
    ),
}


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    file_path = data.get("tool_input", {}).get("file_path", "")
    normalized = file_path.replace("\\", "/")

    # Only fire for orchestrator/ Python files
    if "orchestrator" not in normalized or not normalized.endswith(".py"):
        sys.exit(0)
    if ".claude" in normalized:
        sys.exit(0)

    filename = os.path.basename(normalized)

    rule_message = RULES.get(filename)
    if rule_message is None:
        sys.exit(0)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": rule_message,
        }
    }
    print(json.dumps(output))
    sys.exit(0)  # Never block — advisory only


if __name__ == "__main__":
    main()
