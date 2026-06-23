"""
Hallmark Design Plugin — Lifecycle hooks for structural design system.
=====================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Hooks into the orchestrator plugin lifecycle to inject Hallmark design
constraints before each frontend task, log design choices after each task,
and emit design.md after each project.

Usage:
    from orchestrator.plugins.base import PluginRegistry, get_plugin_registry
    from orchestrator.plugins.hallmark_design import HallmarkDesignPlugin
    registry = get_plugin_registry()
    registry.register(HallmarkDesignPlugin())
"""

from __future__ import annotations

import logging
from pathlib import Path

from .base import Plugin, PluginContext, PluginMetadata, PluginPriority

logger = logging.getLogger(__name__)


class HallmarkDesignPlugin(Plugin):
    """Plugin that wires Hallmark design system into the orchestrator lifecycle."""

    metadata = PluginMetadata(
        name="hallmark_design",
        version="1.0.0",
        description="Hallmark structural design system integration",
        author="Georgios-Chrysovalantis Chatzivantsidis",
        priority=PluginPriority.HIGH,
        dependencies=[],
    )

    def __init__(self) -> None:
        super().__init__()
        self._selector = None
        self._scanner = None
        self._design_log = None
        self._scope_detector = None
        self._preflight = None
        self._project_dir: Path | None = None
        self._feature_flags = None

    def _load(self) -> bool:
        """Lazy-load Hallmark components. Returns False if disabled."""
        if self._feature_flags is None:
            try:
                from orchestrator.crosscutting.config import FeatureFlags

                self._feature_flags = FeatureFlags()
            except Exception as exc:
                logger.debug("Hallmark plugin: cannot load feature flags: %s", exc)
                return False

        if not self._feature_flags.hallmark_enabled:
            return False

        # Lazy-init singletons
        if self._selector is None:
            try:
                from orchestrator.design.hallmark_selector import HallmarkSelector
                from orchestrator.design.preflight_scanner import PreflightScanner
                from orchestrator.design.scope_detector import ScopeDetector

                self._selector = HallmarkSelector()
                self._scanner = PreflightScanner()
                self._scope_detector = ScopeDetector()
            except Exception as exc:
                logger.warning("Hallmark plugin: failed to load components: %s", exc)
                return False

        return True

    async def initialize(self) -> None:
        """No-op: lazy-load on first use."""
        pass

    async def shutdown(self) -> None:
        """No-op."""
        pass

    async def on_pre_project(self, context: PluginContext) -> None:
        """Initialize design log and run preflight scan before project starts."""
        if not self._load():
            return

        if context.project_state is None:
            return

        # Determine project directory
        try:
            self._project_dir = Path(context.project_state.output_dir or ".")
        except Exception:
            self._project_dir = Path(".")

        # Initialize design log
        try:
            from orchestrator.design.design_log import DesignLog

            self._design_log = DesignLog(self._project_dir)
        except Exception as exc:
            logger.warning("Hallmark plugin: cannot init design log: %s", exc)

        # Run preflight scan (async)
        if self._feature_flags.hallmark_preflight:
            try:
                self._preflight = await self._scanner.scan(self._project_dir)
                logger.info(
                    "Hallmark preflight: %s",
                    (
                        self._preflight.findings[:3]
                        if self._preflight.findings
                        else "no prior design system"
                    ),
                )
            except Exception as exc:
                logger.warning("Hallmark plugin: preflight scan failed: %s", exc)
                self._preflight = None

    async def on_pre_task(self, context: PluginContext) -> None:
        """Inject Hallmark constraints into task before execution."""
        if not self._load():
            return
        if context.task is None:
            return

        task = context.task
        brief = getattr(task, "prompt", "")
        target = getattr(task, "target_path", "")
        domain = getattr(task, "domain", "")

        # Detect scope (component vs page)
        scope = self._scope_detector.detect_scope(brief, target)
        task.design_scope = scope

        # For component tasks, inject component-scope block
        if scope.value == "component":
            try:
                from orchestrator.design.prompt_builder import HallmarkPromptBuilder

                builder = HallmarkPromptBuilder()
                component_block = builder.build_component_scope_block()
                # Prepend to skill_prefix or prompt
                existing = getattr(task, "skill_prefix", "") or ""
                task.skill_prefix = component_block + "\n\n" + existing
                logger.debug("Hallmark plugin: injected component-scope block")
            except Exception as exc:
                logger.warning("Hallmark plugin: component block injection failed: %s", exc)
            return

        # For page tasks: select macrostructure, theme, nav, footer
        try:
            from orchestrator.design.prompt_builder import HallmarkPromptBuilder

            builder = HallmarkPromptBuilder()

            # Select macrostructure (with diversification)
            macro = self._selector.select_macrostructure(brief, domain, self._design_log)
            task.target_macrostructure = macro.slug

            # Select theme (respecting preflight + custom signals)
            theme = self._selector.select_theme(brief, macro, self._preflight)

            # Select nav + footer
            nav = self._selector.select_nav(theme.genre)
            footer = self._selector.select_footer(theme.genre)

            # Select archetypes
            archetypes = self._selector.select_archetypes(brief, macro, theme.genre)

            # Build constraint blocks
            macro_block = builder.build_macrostructure_block(macro)
            theme_block = builder.build_theme_block(theme)
            archetype_block = builder.build_archetype_block(archetypes)

            # Combine into skill_prefix
            prefix = "\n\n".join([macro_block, theme_block, archetype_block])

            # Inject self-critique if enabled
            if self._feature_flags.hallmark_self_critique:
                try:
                    from orchestrator.design.self_critique import inject_self_critique

                    prefix = inject_self_critique(prefix)
                except Exception as exc:
                    logger.debug("Hallmark plugin: self-critique injection failed: %s", exc)

            existing = getattr(task, "skill_prefix", "") or ""
            task.skill_prefix = prefix + "\n\n" + existing

            logger.info(
                "Hallmark plugin: %s → %s / %s / nav=%s / footer=%s",
                task.id,
                macro.slug,
                theme.name,
                nav.code if nav else "none",
                footer.code if footer else "none",
            )

            # Store selections in metadata for post_task logging
            context.metadata["hallmark_selections"] = {
                "macrostructure": macro.slug,
                "theme": theme.name,
                "genre": theme.genre.value,
                "nav": nav.code if nav else None,
                "footer": footer.code if footer else None,
            }

        except Exception as exc:
            logger.warning("Hallmark plugin: pre_task selection failed: %s", exc)

    async def on_post_task(self, context: PluginContext) -> None:
        """Log design choices and optionally run audit."""
        if not self._load():
            return
        if context.task is None:
            return

        task = context.task
        selections = context.metadata.get("hallmark_selections")

        # Log to design log
        if self._design_log and selections:
            try:
                from orchestrator.design.design_log import DesignLogEntry
                from orchestrator.design.self_critique import SelfCritiqueParser

                parser = SelfCritiqueParser()
                output = getattr(context.task_result, "output", "") if context.task_result else ""
                scores = parser.parse(output) if output else None

                entry = DesignLogEntry(
                    task_id=task.id,
                    macrostructure=selections.get("macrostructure", ""),
                    theme=selections.get("theme", ""),
                    genre=selections.get("genre", ""),
                    nav=selections.get("nav", ""),
                    footer=selections.get("footer", ""),
                    pre_emit_scores=scores,
                )
                self._design_log.append(entry)
                logger.debug("Hallmark plugin: logged design entry for %s", task.id)
            except Exception as exc:
                logger.warning("Hallmark plugin: design log append failed: %s", exc)

        # AUDIT variant: run audit on the output
        variant = getattr(task, "design_variant", None)
        if variant and variant.value == "audit" and context.task_result:
            try:
                from orchestrator.design.audit_engine import AuditEngine

                engine = AuditEngine()
                output_path = Path(getattr(context.task_result, "file_path", "."))
                files = [output_path] if output_path.exists() else []
                if files:
                    findings = await engine.audit(files)
                    report = engine.format_report(findings)
                    # Attach audit report to task result metadata
                    if not hasattr(context.task_result, "metadata"):
                        context.task_result.metadata = {}  # type: ignore[union-attr]
                    context.task_result.metadata["hallmark_audit"] = report  # type: ignore[union-attr]
                    logger.info(
                        "Hallmark plugin: audit complete for %s (%d findings)",
                        task.id,
                        len(findings),
                    )
            except Exception as exc:
                logger.warning("Hallmark plugin: audit failed: %s", exc)

    async def on_post_project(self, context: PluginContext) -> None:
        """Emit design.md if study mode, and persist design log."""
        if not self._load():
            return

        # STUDY variant: emit design.md from last known DNA
        if context.project_state and getattr(context.project_state, "design_variant", None):
            variant = context.project_state.design_variant
            if variant and variant.value == "study":
                try:
                    from orchestrator.design.study_engine import StudyEngine

                    engine = StudyEngine(client=None)  # No vision client needed for emit
                    # Try to build DNA from design log
                    if self._design_log and self._design_log.entries:
                        last = self._design_log.entries[-1]
                        from orchestrator.design.study_engine import DesignDNA

                        dna = DesignDNA(
                            source=f"project:{context.project_state.project_id}",
                            macrostructure=last.macrostructure,
                            genre=last.genre,
                        )
                        design_md = engine.emit_design_md(dna)
                        output_dir = Path(getattr(context.project_state, "output_dir", "."))
                        (output_dir / "design.md").write_text(design_md, encoding="utf-8")
                        logger.info("Hallmark plugin: emitted design.md for study project")
                except Exception as exc:
                    logger.warning("Hallmark plugin: study emit failed: %s", exc)

        # Persist design log
        if self._design_log:
            try:
                self._design_log.save()
            except Exception as exc:
                logger.warning("Hallmark plugin: design log save failed: %s", exc)
