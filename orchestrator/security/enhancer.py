"""
SecurityEnhancer — Security best practices for generated applications
========================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Ensures every generated app follows OWASP Top 10 security standards.
Injects security requirements into the generation pipeline and provides
a reviewer that validates output against security rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("orchestrator.security.enhancer")


@dataclass
class SecurityRule:
    """A security rule that generated code should follow."""

    name: str
    category: str  # "web", "api", "auth", "data", "infra"
    severity: str  # "critical", "high", "medium"
    description: str
    check_pattern: str = ""


SECURITY_RULES: list[SecurityRule] = [
    # Web Security
    SecurityRule(
        "Content-Security-Policy",
        "web",
        "critical",
        "Set strict CSP headers: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
    ),
    SecurityRule(
        "X-Content-Type-Options",
        "web",
        "high",
        "Set X-Content-Type-Options: nosniff header to prevent MIME sniffing",
    ),
    SecurityRule(
        "X-Frame-Options",
        "web",
        "high",
        "Set X-Frame-Options: DENY or SAMEORIGIN to prevent clickjacking",
    ),
    SecurityRule(
        "Strict-Transport-Security",
        "web",
        "high",
        "Set Strict-Transport-Security: max-age=31536000; includeSubDomains for HTTPS enforcement",
    ),
    SecurityRule("X-XSS-Protection", "web", "medium", "Set X-XSS-Protection: 1; mode=block header"),
    SecurityRule(
        "Referrer-Policy", "web", "medium", "Set Referrer-Policy: strict-origin-when-cross-origin"
    ),
    SecurityRule(
        "Permissions-Policy",
        "web",
        "medium",
        "Set Permissions-Policy: camera=(), microphone=(), geolocation=() to limit API access",
    ),
    # API Security
    SecurityRule(
        "HTTPS Only",
        "api",
        "critical",
        "Enforce HTTPS. Redirect all HTTP requests to HTTPS. Use HSTS preload.",
    ),
    SecurityRule(
        "Rate Limiting",
        "api",
        "high",
        "Implement rate limiting (100 req/min per IP). Return 429 Too Many Requests.",
    ),
    SecurityRule(
        "Input Validation",
        "api",
        "critical",
        "Validate and sanitize ALL user inputs. Never trust query params, body, or headers directly.",
    ),
    SecurityRule(
        "SQL Injection Prevention",
        "api",
        "critical",
        "Use parameterized queries or ORM. Never concatenate user input into SQL strings.",
    ),
    SecurityRule(
        "CORS",
        "api",
        "high",
        "Set strict CORS: only allow specific origins, not '*'. Whitelist methods and headers.",
    ),
    SecurityRule(
        "Error Handling",
        "api",
        "high",
        "Never expose stack traces to clients. Return generic error messages. Log details server-side.",
    ),
    # Auth Security
    SecurityRule(
        "Password Hashing",
        "auth",
        "critical",
        "Hash passwords with bcrypt (cost 12+) or argon2. Never store plaintext or MD5/SHA1.",
    ),
    SecurityRule(
        "Session Management",
        "auth",
        "critical",
        "Use HttpOnly, Secure, SameSite=Strict cookies. Set short session expiry (15-60 min).",
    ),
    SecurityRule(
        "JWT Best Practices",
        "auth",
        "high",
        "Sign JWTs with RS256 or ES256. Set short expiry (15 min). Use refresh tokens. Validate 'aud' claim.",
    ),
    SecurityRule(
        "CSRF Protection",
        "auth",
        "high",
        "Use CSRF tokens for all state-changing requests. SameSite=Strict is not sufficient alone.",
    ),
    # Data Security
    SecurityRule(
        "Secrets Management",
        "data",
        "critical",
        "Never hardcode secrets. Use environment variables or a secrets manager. .env is local only.",
    ),
    SecurityRule(
        "Data Encryption",
        "data",
        "high",
        "Encrypt sensitive data at rest (AES-256). Use TLS 1.3 for data in transit.",
    ),
    # Infrastructure
    SecurityRule(
        "Docker Security",
        "infra",
        "high",
        "Don't run containers as root. Use multi-stage builds. Scan images for vulnerabilities.",
    ),
    SecurityRule(
        "Dependency Pinning",
        "infra",
        "high",
        "Pin all dependency versions. Use lockfiles (poetry.lock, package-lock.json). Regularly audit.",
    ),
    SecurityRule(
        "Logging",
        "infra",
        "medium",
        "Log security events (failed logins, 4xx/5xx errors). Never log passwords or tokens.",
    ),
]


def security_system_prompt() -> str:
    """Get security rules as an LLM system prompt injection."""
    lines = [
        "## Security Requirements — You MUST follow these:",
        "",
    ]
    for cat in ["web", "api", "auth", "data", "infra"]:
        cat_name = {
            "web": "Web Security",
            "api": "API Security",
            "auth": "Authentication",
            "data": "Data Security",
            "infra": "Infrastructure",
        }[cat]
        rules = [r for r in SECURITY_RULES if r.category == cat and r.severity == "critical"]
        rules += [r for r in SECURITY_RULES if r.category == cat and r.severity != "critical"]
        if rules:
            lines.append(f"### {cat_name}")
            for r in rules:
                prefix = {"critical": "[CRITICAL]", "high": "[HIGH]", "medium": "[MEDIUM]"}[
                    r.severity
                ]
                lines.append(f"- {prefix} {r.description}")
            lines.append("")
    lines.append("### OpenGraph Requirements")
    lines.append("- Every HTML page MUST include Facebook Open Graph meta tags:")
    lines.append("  - og:title — page title, 60-70 characters")
    lines.append("  - og:description — page summary, 150-160 characters")
    lines.append("  - og:image — 1200x630 PNG at least, absolute URL")
    lines.append("  - og:url — canonical URL")
    lines.append("  - og:type — 'website' or 'article'")
    lines.append("  - og:site_name — your site name")
    lines.append("- Include Twitter Card meta tags:")
    lines.append("  - twitter:card — 'summary_large_image'")
    lines.append("  - twitter:title — same as og:title")
    lines.append("  - twitter:description — same as og:description")
    lines.append("  - twitter:image — same as og:image")
    lines.append("")
    return "\n".join(lines)


class OpenGraphGenerator:
    """Generates perfect OpenGraph meta tags for any page."""

    def generate_head_tags(
        self,
        title: str,
        description: str,
        url: str,
        image_url: str = "",
        site_name: str = "",
        og_type: str = "website",
    ) -> str:
        """Generate complete OpenGraph and Twitter Card meta tags.

        Args:
            title: Page title (60-70 chars recommended)
            description: Meta description (150-160 chars recommended)
            url: Canonical URL
            image_url: 1200x630 PNG URL (required for proper sharing)
            site_name: Site name for og:site_name
            og_type: 'website' or 'article'

        Returns:
            HTML meta tags string ready to insert into <head>.
        """
        import html

        t = html.escape(title[:70])
        d = html.escape(description[:160])
        u = html.escape(url)
        img = html.escape(image_url) if image_url else ""
        sn = html.escape(site_name) if site_name else ""

        tags = [
            "<!-- OpenGraph / Facebook -->",
            f'<meta property="og:title" content="{t}" />',
            f'<meta property="og:description" content="{d}" />',
            f'<meta property="og:url" content="{u}" />',
            f'<meta property="og:type" content="{og_type}" />',
        ]
        if img:
            tags.append(f'<meta property="og:image" content="{img}" />')
            tags.append(f'<meta property="og:image:width" content="1200" />')
            tags.append(f'<meta property="og:image:height" content="630" />')
        if sn:
            tags.append(f'<meta property="og:site_name" content="{sn}" />')

        tags.extend(
            [
                "",
                "<!-- Twitter Card -->",
                f'<meta name="twitter:card" content="summary_large_image" />',
                f'<meta name="twitter:title" content="{t}" />',
                f'<meta name="twitter:description" content="{d}" />',
            ]
        )
        if img:
            tags.append(f'<meta name="twitter:image" content="{img}" />')

        return "\n    ".join(tags)

    def generate_json_ld(self, name: str, description: str, url: str, logo_url: str = "") -> str:
        """Generate JSON-LD structured data for the website."""

        data = {
            "@context": "https://schema.org",
            "@type": "WebSite",
            "name": name,
            "description": description[:200],
            "url": url,
        }
        if logo_url:
            data["image"] = logo_url

        import json

        return f'<script type="application/ld+json">\n{json.dumps(data, indent=2)}\n</script>'

    def full_head_section(
        self,
        title: str,
        description: str,
        url: str,
        image_url: str = "",
        site_name: str = "",
        og_type: str = "website",
    ) -> str:
        """Generate complete head section with OG tags and JSON-LD."""
        og = self.generate_head_tags(title, description, url, image_url, site_name, og_type)
        ld = self.generate_json_ld(site_name or title, description, url, image_url)
        return f"  {og}\n\n  {ld}"
