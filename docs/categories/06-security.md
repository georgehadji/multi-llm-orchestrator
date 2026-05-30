# Category 6: Security & Validation

> **Focus:** Structured security auditing, vulnerability management, and AI-powered security rules generation.
> **Phases:** 1 | **Est. Days:** 2-3 | **Weight:** 2% of total roadmap

---

## Overview

This category provides **AI-powered security auditing** with structured findings, severity levels, CWE identifiers, and auto-generated security rules that the AI follows in future generations.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 1 | D2 | Dyad | **AI Security Review** — 6 categories (auth, data, injection, crypto, config, dependency), Critical/High/Medium/Low/Info severity, CWE IDs, one-click fixes, `SECURITY_RULES.md` knowledge base | 2-3 | N1 |

---

## Security Review Details

### Check Categories

| Category | Checks |
|----------|--------|
| **Auth** | Missing authentication, weak passwords, insecure sessions, missing rate limiting, JWT misconfig |
| **Data** | Exposed sensitive data, missing input validation, missing output encoding, IDOR, mass assignment |
| **Injection** | SQL injection, NoSQL injection, XSS, command injection, SSRF |
| **Crypto** | Weak hashing (MD5/SHA1), insecure RNG, hardcoded keys, missing at-rest encryption, insecure TLS |
| **Config** | Debug mode in production, exposed env vars, missing security headers, CORS misconfig, default creds |
| **Dependency** | Known vulnerable packages, unpinned versions, unmaintained packages |

### Key Innovations

- **Structured severity levels** (Critical → Info) with CWE identifiers for compliance
- **One-click fixes** for individual findings within the security review panel
- **SECURITY_RULES.md** auto-generated from findings — teaches the AI to avoid reintroducing vulnerabilities
- **Skip list** for false positives — user can mark findings as safe for the project
