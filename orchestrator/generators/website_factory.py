"""
Batch website factory — turn a manifest of site specs into N gated sites.

This is the "factory" layer on top of :class:`WebsiteGenerator`, which builds
exactly one site per invocation. What a factory adds beyond a loop:

- **Isolation.** One site failing is a row in the report, never the end of the
  run. A 50-site batch that dies on site 7 is worse than useless.
- **A verdict per site.** Every outcome carries the quality gate's decision.
  A site that was never judged is never reported as shippable.
- **Bounded concurrency.** Provider rate limits and spend both scale with
  parallelism, so it is capped rather than unbounded ``gather``.
- **Delivery.** Each site is initialised as its own git repository and
  committed, so the output is handed over as history rather than a folder.

The generation step is injected (``build_site``) so the batch logic is testable
without spending money on live model calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# Fields a manifest entry may set. Anything else is rejected loudly rather than
# silently ignored — a typo'd key that vanishes is exactly the "silent drop"
# failure this repo has been bitten by in config land.
_SPEC_FIELDS = {
    "slug",
    "description",
    "company_name",
    "industry",
    "page_type",
    "framework",
    "preset",
    "sections",
    "min_quality",
    "require_all_checks",
    "image_quality",
    "image_model",
    "atelier_theme",
    "source_url",
    "git_remote",
}

_DEFAULT_SECTIONS = ["hero", "features", "pricing", "testimonials", "faq", "cta", "footer"]


@dataclass
class SiteSpec:
    """One site to build. Mirrors the flags of the ``website`` subcommand."""

    slug: str
    description: str
    company_name: str = ""
    industry: str = "technology"
    page_type: str = "landing"
    framework: str = "html"  # html | react | next.js | svelte | sveltekit
    preset: str = "modern"
    sections: list[str] = field(default_factory=lambda: list(_DEFAULT_SECTIONS))
    min_quality: float = 0.0
    require_all_checks: bool = False
    image_quality: str = "balanced"
    image_model: str = "auto"
    atelier_theme: str = ""
    source_url: str = ""
    git_remote: str = ""


@dataclass
class SiteOutcome:
    """What happened to one site. Serialised verbatim into the batch report."""

    slug: str
    output_dir: str
    success: bool = False
    quality_gate_passed: bool | None = None
    score: float = 0.0
    # WF-100's launch decision, kept apart from `quality_gate_passed` because
    # they answer different questions: the gate asks whether the generator
    # produced something sound, the verdict asks whether it may go live.
    verdict: str = ""
    wf100_score: float = 0.0
    wf100_ceiling: float = 0.0
    failures: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    error: str = ""
    committed: bool = False
    commit_sha: str = ""


def _validate_slug(slug: Any) -> str:
    """A slug becomes a directory name, so reject anything path-unsafe."""
    if not slug or not isinstance(slug, str):
        raise ValueError("every site entry needs a non-empty 'slug'")
    if slug != Path(slug).name or slug in {".", ".."} or "/" in slug or "\\" in slug:
        raise ValueError(f"site slug must be a plain directory name, got {slug!r}")
    return slug


def load_manifest(path: str | Path) -> list[SiteSpec]:
    """Load a batch manifest (YAML or JSON) into validated :class:`SiteSpec` objects.

    Shape::

        defaults:            # optional, merged into every site
          framework: html
          min_quality: 0.85
        sites:
          - slug: gadini-barberia
            description: Premium Italian barber shop in Thessaloniki
            company_name: Gadini Barberia

    Per-site keys win over ``defaults``.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")

    if path.suffix.lower() in (".yaml", ".yml"):
        # Imported lazily and locally, matching ClientInfo.from_yaml — PyYAML is
        # used across this repo but is not a declared dependency.
        import yaml

        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)

    if not isinstance(data, dict):
        raise ValueError("manifest must be a mapping with a 'sites' list")

    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("'defaults' must be a mapping")

    sites = data.get("sites")
    if not isinstance(sites, list) or not sites:
        raise ValueError("manifest must define a non-empty 'sites' list")

    specs: list[SiteSpec] = []
    seen: set[str] = set()
    for index, entry in enumerate(sites):
        if not isinstance(entry, dict):
            raise ValueError(f"site #{index} must be a mapping")

        merged = {**defaults, **entry}
        unknown = set(merged) - _SPEC_FIELDS
        if unknown:
            raise ValueError(
                f"site #{index}: unknown manifest key(s) {sorted(unknown)}; "
                f"known keys are {sorted(_SPEC_FIELDS)}"
            )

        slug = _validate_slug(merged.get("slug"))
        if slug in seen:
            raise ValueError(f"duplicate site slug {slug!r} — slugs must be unique")
        seen.add(slug)

        if not merged.get("description"):
            raise ValueError(f"site {slug!r} needs a non-empty 'description'")

        specs.append(SiteSpec(**merged))

    return specs


class WebsiteFactory:
    """Run a batch of site builds with bounded concurrency and per-site delivery."""

    def __init__(
        self,
        build_site: Callable[[SiteSpec, Path], Awaitable[Any]] | None = None,
        concurrency: int = 3,
        git_delivery: bool = True,
    ) -> None:
        self._build_site = build_site or _build_site_with_generator
        self._concurrency = max(1, int(concurrency))
        self._git_delivery = git_delivery

    async def run(
        self,
        specs: list[SiteSpec],
        output_root: str | Path,
        report_name: str = "factory-report.json",
    ) -> list[SiteOutcome]:
        """Build every spec, then write a batch report next to the sites."""
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        semaphore = asyncio.Semaphore(self._concurrency)
        started = time.time()

        async def _guarded(spec: SiteSpec) -> SiteOutcome:
            async with semaphore:
                return await self._build_one(spec, output_root)

        outcomes = await asyncio.gather(*(_guarded(spec) for spec in specs))
        outcomes = list(outcomes)

        self._write_report(output_root / report_name, outcomes, time.time() - started)
        return outcomes

    async def _build_one(self, spec: SiteSpec, output_root: Path) -> SiteOutcome:
        """Build one site. Never raises — a failure becomes a reported outcome."""
        site_dir = output_root / spec.slug
        site_dir.mkdir(parents=True, exist_ok=True)
        outcome = SiteOutcome(slug=spec.slug, output_dir=str(site_dir))
        started = time.time()

        try:
            result = await self._build_site(spec, site_dir)
            outcome.success = bool(getattr(result, "success", False))
            outcome.quality_gate_passed = getattr(result, "quality_gate_passed", None)
            outcome.failures = list(getattr(result, "gate_failures", []) or [])
            outcome.cost_usd = float(getattr(result, "total_cost", 0.0) or 0.0)
            report = getattr(result, "quality_report", None)
            outcome.score = float(getattr(report, "score", 0.0) or 0.0)
        except Exception as exc:  # noqa: BLE001 — one site must not kill the batch
            logger.error("Site %s failed to build: %s", spec.slug, exc)
            outcome.success = False
            outcome.error = str(exc)
            # A crashed build is never shippable, whatever the gate never said.
            outcome.quality_gate_passed = False

        if outcome.success:
            try:
                verdict, score, ceiling, blocking = await asyncio.to_thread(_audit_site, site_dir)
                outcome.verdict = verdict
                outcome.wf100_score = score
                outcome.wf100_ceiling = ceiling
                outcome.failures.extend(blocking)
                # WF-100 vetoes, it never blesses. A critical failure means the
                # site cannot launch however well the build-time validator
                # scored it; a clean audit does not retroactively pass a build
                # the validator rejected.
                if verdict == "no_launch":
                    outcome.quality_gate_passed = False
            except Exception as exc:  # noqa: BLE001 — an audit crash is not a verdict
                logger.warning(
                    "Site %s: WF-100 audit failed (%s); verdict left unrecorded", spec.slug, exc
                )

        outcome.duration_seconds = time.time() - started

        if self._git_delivery:
            # Rejected builds are committed too — the artifact is the evidence
            # you diff against on the next attempt.
            outcome.commit_sha = _commit_site(site_dir, spec, outcome)
            outcome.committed = bool(outcome.commit_sha)

        return outcome

    @staticmethod
    def _write_report(path: Path, outcomes: list[SiteOutcome], elapsed: float) -> None:
        shippable = sum(1 for o in outcomes if o.quality_gate_passed is True)
        rejected = sum(1 for o in outcomes if o.quality_gate_passed is False)
        launch_ready = sum(1 for o in outcomes if o.verdict == "launch")
        payload = {
            "sites": [asdict(o) for o in outcomes],
            "total": len(outcomes),
            "shippable": shippable,
            "rejected": rejected,
            "unjudged": len(outcomes) - shippable - rejected,
            "launch_ready": launch_ready,
            "total_cost_usd": round(sum(o.cost_usd for o in outcomes), 6),
            "duration_seconds": round(elapsed, 3),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(
            "Factory report: %d site(s), %d WF-100 launch-ready, %d rejected, $%.4f -> %s",
            payload["total"],
            launch_ready,
            rejected,
            payload["total_cost_usd"],
            path,
        )


def _audit_site(site_dir: Path) -> tuple[str, float, float, list[str]]:
    """Hold one built site to WF-100. Returns (verdict, score, ceiling, blocking lines).

    A site with no readable HTML — an unbuilt Next.js project, say — comes back
    ``not_audited``: the standard was never applied, which is not the same as
    applying it and failing. That distinction is the whole point of the
    auditor's evidence model, and it has to survive the trip into the batch
    report, or "we did not look" reads as "we looked and it was fine".
    """
    from .wf100.auditor import audit
    from .wf100.evidence import SiteEvidence
    from .wf100.standard import Evidence

    evidence = SiteEvidence.from_directory(str(site_dir))
    # No markup, no website. Asked the other way round — "did any check score?"
    # — a directory holding only `page.tsx` earns points from the asset-only
    # checks and comes back PENDING, which is a verdict on a site that was
    # never there.
    if Evidence.MARKUP not in evidence.available:
        return "not_audited", 0.0, 0.0, []

    report = audit(evidence)

    blocking = [f"WF-100 {b.code}: {b.detail}" for b in report.blockers]
    blocking += [
        f"WF-100 {f.check.id} ({f.check.title}): {f.detail}" for f in report.critical_failures()
    ]
    return report.verdict.value, report.score, report.ceiling, blocking


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a git command with an argument list (never a shell string)."""
    return subprocess.run(  # noqa: S603 — fixed argv, no shell, no user-built command string
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def _commit_site(site_dir: Path, spec: SiteSpec, outcome: SiteOutcome) -> str:
    """Initialise ``site_dir`` as its own repo and commit the build. Returns the SHA.

    Best-effort: delivery problems must not turn a good build into a failed one,
    so this logs and returns "" rather than raising.
    """
    try:
        if not (site_dir / ".git").is_dir():
            init = _run_git(["init", "-q"], cwd=site_dir)
            if init.returncode != 0:
                logger.warning("git init failed for %s: %s", spec.slug, init.stderr.strip())
                return ""
            # Repo-local identity so delivery does not depend on global git config.
            _run_git(["config", "user.name", "multi-llm-orchestrator"], cwd=site_dir)
            _run_git(["config", "user.email", "orchestrator@localhost"], cwd=site_dir)

        if spec.git_remote and not _run_git(["remote"], cwd=site_dir).stdout.strip():
            _run_git(["remote", "add", "origin", spec.git_remote], cwd=site_dir)

        _run_git(["add", "-A"], cwd=site_dir)
        if not _run_git(["diff", "--cached", "--quiet"], cwd=site_dir).returncode:
            return ""  # nothing staged, nothing to commit

        verdict = (
            "shippable"
            if outcome.quality_gate_passed is True
            else "REJECTED" if outcome.quality_gate_passed is False else "ungated"
        )
        message = (
            f"build({spec.slug}): {verdict} — quality {outcome.score:.2f}\n\n"
            f"Framework: {spec.framework}  Preset: {spec.preset}\n"
            f"Cost: ${outcome.cost_usd:.4f}  Duration: {outcome.duration_seconds:.1f}s\n"
        )
        if outcome.failures:
            message += "\nFailing checks:\n" + "\n".join(f"  - {f}" for f in outcome.failures)
        if outcome.error:
            message += f"\nError: {outcome.error}\n"

        commit = _run_git(["commit", "-q", "-m", message], cwd=site_dir)
        if commit.returncode != 0:
            logger.warning("git commit failed for %s: %s", spec.slug, commit.stderr.strip())
            return ""
        return _run_git(["rev-parse", "HEAD"], cwd=site_dir).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("git delivery failed for %s: %s", spec.slug, exc)
        return ""


async def _build_site_with_generator(spec: SiteSpec, output_dir: Path):
    """Default build step: drive the real WebsiteGenerator for one spec."""
    from ..budget import Budget
    from ..design_system import DesignSystem
    from ..domain.ports import TaskExecutorAdapter
    from ..engine import Orchestrator
    from .website_generator import ClientInfo, WebsiteConfig, WebsiteGenerator

    engine = None
    try:
        engine = Orchestrator(budget=Budget(max_usd=3.0), max_concurrency=3)
    except Exception as exc:  # noqa: BLE001 — fall back to the content-brief path
        logger.warning("Site %s: engine unavailable (%s); using template fallback", spec.slug, exc)

    generator = WebsiteGenerator(
        executor=TaskExecutorAdapter(engine._execute_task) if engine else None,
        orchestrator_engine=engine,
    )
    config = WebsiteConfig(
        framework=spec.framework,
        styling="tailwind" if spec.framework != "html" else "css",
        page_type=spec.page_type,
        sections=list(spec.sections),
        image_model=spec.image_model,
        atelier_theme=spec.atelier_theme,
        description=spec.description,
        brand_name=spec.company_name or spec.slug,
        image_quality=spec.image_quality,
        source_url=spec.source_url,
        min_quality=spec.min_quality,
        require_all_checks=spec.require_all_checks,
    )
    return await generator.generate(
        design_system=DesignSystem(tone=spec.preset),
        client_info=ClientInfo(
            name=spec.company_name or spec.slug,
            industry=spec.industry,
            description=spec.description,
        ),
        config=config,
        output_dir=output_dir,
    )
