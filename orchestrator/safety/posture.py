"""Report the security posture this process would actually run with (T10).

Every control added for SEC-001…006 is switched by an environment variable, so
"is this deployment hardened?" is currently answered by reading six modules.
This answers it in one command:

    python -m orchestrator.safety.posture

It reports whether a setting is configured and what it resolves to — never the
value of a secret. `ORCHESTRATOR_API_KEY_PEPPER` shows as set/unset, never as
its contents, because posture output is exactly the sort of thing that ends up
pasted into an issue.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from orchestrator.ide_backend import auth as ide_auth
from orchestrator.ide_backend import security as ide_security
from orchestrator.safety import api_keys, outbound
from orchestrator.tools import shell_tool

__all__ = ["Setting", "posture", "render"]


@dataclass(frozen=True)
class Setting:
    name: str
    value: str
    hardened: bool
    note: str = ""


def _yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def posture() -> list[Setting]:
    """The effective security settings, worst-first is the caller's problem."""
    remote = ide_security.remote_allowed()
    origins = _origins()
    pepper_set = bool(os.getenv(api_keys.PEPPER_ENV, "").strip())
    key_store_path = os.getenv(api_keys.STORE_PATH_ENV, "").strip()
    allowed_hosts = sorted(outbound.allowed_hosts())

    return [
        Setting(
            "ide.remote_binding",
            _yes_no(remote),
            hardened=not remote,
            note=f"{ide_security.ALLOW_REMOTE_ENV}; loopback-only unless set",
        ),
        Setting(
            "ide.auth_required",
            _yes_no(ide_auth.auth_required()),
            hardened=ide_auth.auth_required(),
            note=f"{ide_auth.AUTH_REQUIRED_ENV}; requests are LOCAL_PRINCIPAL when off",
        ),
        Setting(
            "ide.allowed_origins",
            ", ".join(origins) if origins else "<invalid>",
            hardened=bool(origins) and "*" not in origins,
            note=ide_security.ALLOWED_ORIGINS_ENV,
        ),
        Setting(
            "api_keys.pepper",
            "set" if pepper_set else "unset",
            hardened=pepper_set,
            note=f"{api_keys.PEPPER_ENV}; value never printed",
        ),
        Setting(
            "api_keys.persistence",
            key_store_path or "in-memory",
            hardened=bool(key_store_path),
            note=f"{api_keys.STORE_PATH_ENV}; in-memory keys die with the process",
        ),
        Setting(
            "outbound.allowed_hosts",
            ", ".join(allowed_hosts) if allowed_hosts else "<any public address>",
            hardened=bool(allowed_hosts),
            note=(
                f"{outbound.ALLOWED_HOSTS_ENV}; internal addresses are refused either "
                "way, an allowlist also blunts DNS rebinding"
            ),
        ),
        Setting(
            "outbound.plain_http",
            _yes_no(bool(os.getenv(outbound.ALLOW_HTTP_ENV, "").strip())),
            hardened=not os.getenv(outbound.ALLOW_HTTP_ENV, "").strip(),
            note=outbound.ALLOW_HTTP_ENV,
        ),
        Setting(
            "tools.shell_enabled",
            _yes_no(shell_tool.tool_enabled()),
            hardened=not shell_tool.tool_enabled(),
            note=f"{shell_tool.ENABLED_ENV}; argv allowlist even when enabled",
        ),
    ]


def _origins() -> list[str]:
    try:
        return ide_security.allowed_origins()
    except ide_security.InsecureBindError:
        # A wildcard in the origins env. Reported as invalid rather than
        # raising: the whole point of this command is to run on a bad config.
        return []


def render(settings: list[Setting], as_json: bool = False) -> str:
    if as_json:
        return json.dumps(
            [
                {"name": s.name, "value": s.value, "hardened": s.hardened, "note": s.note}
                for s in settings
            ],
            indent=2,
        )

    width = max(len(s.name) for s in settings)
    lines = ["Effective security posture", "=" * 60]
    for setting in settings:
        mark = "ok  " if setting.hardened else "WARN"
        lines.append(f"[{mark}] {setting.name.ljust(width)}  {setting.value}")
        if setting.note:
            lines.append(f"{' ' * (width + 9)}{setting.note}")

    weak = [s.name for s in settings if not s.hardened]
    lines.append("=" * 60)
    lines.append(
        "All hardened." if not weak else f"{len(weak)} setting(s) not hardened: {', '.join(weak)}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Report effective security settings")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    args = parser.parse_args(argv)

    settings = posture()
    print(render(settings, as_json=args.json))
    # Exit 1 when something is not hardened, so CI or a deploy step can gate
    # on it without parsing the output.
    return 0 if all(s.hardened for s in settings) else 1


if __name__ == "__main__":
    raise SystemExit(main())
