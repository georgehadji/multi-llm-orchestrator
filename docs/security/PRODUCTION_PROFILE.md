# Production security profile

Every control added for SEC-001…006 is switched by an environment variable, and
they all default to the *local development* posture: loopback binding, no
authentication, in-memory keys. That is deliberate — a hardened default that
makes local work annoying is a default people route around, which is exactly how
an insecure standalone server became the shipped IDE launcher.

Running anywhere other than a developer laptop means setting the variables below.

## Check what you are actually running

```bash
python -m orchestrator.safety.posture
```

Prints each effective setting with `ok`/`WARN`, and exits non-zero if anything is
unhardened, so a deploy step can gate on it. It reports whether a secret is set,
never its value.

## Required settings

| Variable | Production value | What it prevents |
|---|---|---|
| `ORCHESTRATOR_IDE_AUTH_REQUIRED` | `true` | Unauthenticated access to every IDE session (SEC-001b) |
| `ORCHESTRATOR_IDE_ALLOWED_ORIGINS` | your front end's exact origins | A visited site acting as a same-origin client (SEC-001) |
| `ORCHESTRATOR_API_KEY_PEPPER` | 32+ random bytes from your secret manager | A leaked key store being enough to forge keys (T8) |
| `ORCHESTRATOR_API_KEY_STORE` | a path on persistent storage | Keys silently dying on restart, and revocations being lost (T8) |
| `ORCHESTRATOR_OUTBOUND_ALLOWED_HOSTS` | the hosts you actually fetch from | SSRF via DNS rebinding (SEC-004) |
| `ORCHESTRATOR_ADMIN_SECRET` | from your secret manager | Anonymous key registration |

Leave these **unset**:

| Variable | Why |
|---|---|
| `ORCHESTRATOR_IDE_ALLOW_REMOTE` | Binds beyond loopback. If you genuinely need it, `ORCHESTRATOR_IDE_AUTH_REQUIRED=true` is mandatory — the server refuses to start otherwise |
| `ORCHESTRATOR_SHELL_TOOL_ENABLED` | Enables command execution. Even then it is an argv allowlist, never a shell (SEC-003) |
| `ORCHESTRATOR_OUTBOUND_ALLOW_HTTP` | Permits plaintext outbound fetches |

## Deployment shape

- **Binding.** Loopback or a private interface, behind a reverse proxy that
  terminates TLS and authenticates. The application does not terminate TLS.
- **Secrets.** Injected from a secret manager as environment variables. Never in
  the image, the repo, or a `.env` that ships.
- **Key store.** The file the pepper protects. Restrict it to the service user;
  the pepper is what makes the file insufficient on its own, but both matter.
- **SQLite and Redis.** Owned by the service user, not world-readable. Redis on a
  private network with authentication — the cache holds prompt and response text.
- **Audit retention.** Keep denial events long enough to see a pattern. Repeated
  401/403 from one principal, or repeated `OutboundPolicyError`, is someone
  looking for a way in; alert on the rate, not on single events.

## Dependency exceptions

`pip-audit --strict` runs in CI and fails the build on any known advisory.

To accept one, add it to an `--ignore-vuln` flag in
`.github/workflows/ci.yml` with **both**:

1. why it does not apply here, or what compensating control covers it;
2. a review date, no more than 90 days out.

An entry missing either is unreviewed and should be deleted rather than
inherited. Prefer upgrading; an exception is a promise to come back.

## What CI enforces

| Job step | Enforces |
|---|---|
| `bandit` | HIGH-severity findings in `orchestrator/` |
| `pip-audit --strict` | No known-vulnerable dependencies |
| `gitleaks` | No secrets in the history, not just the diff |
| `pytest tests/unit/security/` | The SEC-001…006 regression guards |
| `lint-imports` | Architectural contracts |

The regression guards are the part that keeps fixed findings fixed: no pickle
deserialization, no wildcard CORS with credentials, no host shell, no unguarded
outbound fetch, ownership checked on every session-scoped route.
