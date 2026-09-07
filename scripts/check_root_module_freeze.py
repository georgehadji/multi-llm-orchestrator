#!/usr/bin/env python3
"""
Root-level module freeze check.

Detects new ``.py`` files added to ``orchestrator/`` at depth 1 (the root
package).  The baseline is embedded in this file; update it with ``--update``
when legitimate additions are reviewed and accepted.

Usage::

    python scripts/check_root_module_freeze.py          # check (CI)
    python scripts/check_root_module_freeze.py --update  # update baseline
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORCHESTRATOR_DIR = os.path.join(PROJECT_ROOT, "orchestrator")

# >>>GENERATED-BASELINE-BEGIN<<<
# BASELINE — auto-generated with --update.  DO NOT edit manually.
BASELINE: set[str] = {
    "__main__.py",
    "ab_testing.py",
    "accountability.py",
    "adaptive_router.py",
    "adaptive_templates.py",
    "agent_model_registry.py",
    "agent_safety.py",
    "analyzer.py",
    "api_builder.py",
    "api_clients.py",
    "api_server.py",
    "app_assembler.py",
    "app_builder.py",
    "app_detector.py",
    "app_store_assets.py",
    "app_store_validator.py",
    "app_verifier.py",
    "ara_execution_strategy.py",
    "ara_integration.py",
    "ara_pipelines.py",
    "architecture_advisor.py",
    "architecture_rules.py",
    "assembler.py",
    "assumption_gate.py",
    "async_event_store.py",
    "async_file_io.py",
    "audit.py",
    "autonomous_debugger.py",
    "benchmark_suite.py",
    "bm25_search.py",
    "brain.py",
    "breakpoints.py",
    "browser_testing.py",
    "budget.py",
    "cache.py",
    "cache_optimizer.py",
    "caching.py",
    "canary_deployment.py",
    "capability_logger.py",
    "checkpoints.py",
    "cicd_generator.py",
    "circuit_breaker.py",
    "cli.py",
    "cli_dashboard.py",
    "cli_nash.py",
    "cli_website.py",
    "code_executor.py",
    "code_post_processor.py",
    "code_validator.py",
    "codebase_analyzer.py",
    "codebase_context.py",
    "codebase_profile.py",
    "codebase_reader.py",
    "codebase_writer.py",
    "command_center.py",
    "command_center_integration.py",
    "command_center_server.py",
    "command_registry.py",
    "competitive.py",
    "component_library.py",
    "concurrency_controller.py",
    "config.py",
    "config_as_code.py",
    "config_sync.py",
    "constants.py",
    "container.py",
    "context_compressor.py",
    "context_condensing.py",
    "context_dedup.py",
    "context_sources.py",
    "context_truncator.py",
    "control_plane.py",
    "copy_generator.py",
    "cost.py",
    "cost_analytics.py",
    "cost_tracker.py",
    "dashboard.py",
    "dashboard_bridge.py",
    "data_sources.py",
    "database_generator.py",
    "dependency_scanner.py",
    "deployment_feedback.py",
    "deployment_service.py",
    "design_system.py",
    "design_to_code.py",
    "dev_server.py",
    "diagnostics.py",
    "diff_generator.py",
    "drift.py",
    "engine.py",
    "engine_flags.py",
    "engine_slimming.py",
    "enhancer.py",
    "error_handling.py",
    "events_resilient.py",
    "exceptions.py",
    "federated_learning.py",
    "feedback_loop.py",
    "frontend_rules.py",
    "frontend_security.py",
    "fullstack_generator.py",
    "git_hooks.py",
    "git_service.py",
    "git_sync.py",
    "github_sync.py",
    "gradual_rollout.py",
    "guardrails.py",
    "hierarchy.py",
    "hitl_workflow.py",
    "hybrid_search_pipeline.py",
    "image_generator.py",
    "image_optimizer.py",
    "indesign_plugin_rules.py",
    "input_validation.py",
    "integration_circuit_breaker.py",
    "ios_hig_prompts.py",
    "issue_tracking.py",
    "knowledge_base.py",
    "knowledge_graph.py",
    "learning_aggregator.py",
    "log_config.py",
    "logging.py",
    "logging_generator.py",
    "memory_bank.py",
    "memory_tier.py",
    "meta_config.py",
    "meta_integration.py",
    "meta_monitoring.py",
    "meta_orchestrator.py",
    "meta_performance.py",
    "meta_v2_integration.py",
    "metrics.py",
    "model_registry.py",
    "model_routing.py",
    "model_selector.py",
    "models.py",
    "models_skill.py",
    "monitoring.py",
    "multi_platform_generator.py",
    "multi_tenant_gateway.py",
    "nash_auto_tuning.py",
    "nash_events.py",
    "nash_stable_orchestrator.py",
    "native_features.py",
    "nexus_cli.py",
    "opengraph_generator.py",
    "openrouter_ab_testing.py",
    "optimization.py",
    "orchestration_agent.py",
    "output_organizer.py",
    "output_writer.py",
    "pareto_frontier.py",
    "performance.py",
    "persona.py",
    "persona_modes.py",
    "phase_aware_models.py",
    "plan_review_data.py",
    "plan_reviewer.py",
    "plan_then_build.py",
    "planner.py",
    "plugin_isolation.py",
    "plugin_isolation_secure.py",
    "policy.py",
    "policy_dsl.py",
    "policy_engine.py",
    "ports.py",
    "preflight.py",
    "preview_server.py",
    "product_manager.py",
    "progress.py",
    "progress_writer.py",
    "progressive_output.py",
    "project_analyzer.py",
    "project_assembler.py",
    "project_context.py",
    "project_copier.py",
    "project_file.py",
    "project_manager.py",
    "prompt_builder.py",
    "prompt_compressor.py",
    "prompt_enhancer.py",
    "provisioned_throughput.py",
    "quality_control.py",
    "rate_limiter.py",
    "red_team.py",
    "reference_monitor.py",
    "release_manager.py",
    "reranker.py",
    "resilience.py",
    "responsive_layouts.py",
    "retry_utils.py",
    "sandbox.py",
    "secrets_generator.py",
    "secrets_manager.py",
    "secure_cache.py",
    "secure_execution.py",
    "security_templates.py",
    "security_validator.py",
    "semantic_cache.py",
    "session_lifecycle.py",
    "session_watcher.py",
    "site_manager.py",
    "slack_integration.py",
    "slash_commands.py",
    "slash_integrations.py",
    "specs.py",
    "state.py",
    "streaming.py",
    "streaming_optimizer.py",
    "streaming_resilient.py",
    "structured_outputs.py",
    "swiftstack_integration.py",
    "task_factory.py",
    "task_handlers.py",
    "task_schemas.py",
    "task_verifier.py",
    "tdd_config.py",
    "telemetry.py",
    "telemetry_store.py",
    "tenancy.py",
    "test_first_generator.py",
    "test_fixer.py",
    "test_instructor_tenacity.py",
    "test_validator.py",
    "testing_templates.py",
    "token_budget.py",
    "token_optimizer.py",
    "tracing.py",
    "transfer_learning.py",
    "triggers.py",
    "validators.py",
    "version_manager.py",
    "visualization.py",
    "web_assembler.py",
    "website_generator.py",
    "website_validator.py",
    "wordpress_plugin_rules.py",
    "xai_search.py",
}
# >>>GENERATED-BASELINE-END<<<


def _current_modules() -> set[str]:
    """Return set of .py filenames at orchestrator/ root (excluding __init__.py)."""
    return {
        f
        for f in os.listdir(ORCHESTRATOR_DIR)
        if f.endswith(".py") and f != "__init__.py"
    }


def _update_baseline() -> None:
    """Regenerate the BASELINE set in this file from current filesystem state.

    The sentinels are ASSEMBLED at runtime rather than written as literals. The
    previous implementation searched for marker strings that appeared verbatim
    in its own source, so content.find() matched the marker *assignment* inside
    this function instead of the baseline block, and the script rewrote itself
    from the middle — producing a SyntaxError and a truncated baseline. That is
    why the committed baseline held 211 entries while 257 root modules already
    existed at the very commit that generated it.
    """
    # Assembled, not literal: a literal here would be found by find() below.
    begin = ">>>GENERATED-BASELINE" + "-BEGIN<<<"
    end = ">>>GENERATED-BASELINE" + "-END<<<"

    modules = sorted(_current_modules())
    new_block = f"# {begin}\n# BASELINE — auto-generated with --update.  DO NOT edit manually.\n"
    new_block += "BASELINE: set[str] = {\n"
    for m in modules:
        new_block += f'    "{m}",\n'
    new_block += "}\n"
    new_block += f"# {end}\n"

    with open(__file__, encoding="utf-8") as f:
        content = f.read()

    if content.count(f"# {begin}") != 1 or content.count(f"# {end}") != 1:
        print(
            "ERROR: baseline sentinels must appear exactly once each",
            file=sys.stderr,
        )
        sys.exit(1)

    start = content.index(f"# {begin}")
    stop = content.index(f"# {end}")
    stop = content.index("\n", stop) + 1

    with open(__file__, "w", encoding="utf-8") as f:
        f.write(content[:start] + new_block + content[stop:])

    print(f"Baseline updated: {len(modules)} root modules")


def main() -> int:
    parser = argparse.ArgumentParser(description="Root module freeze check")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update baseline to current module list (admin only)",
    )
    args = parser.parse_args()

    if args.update:
        _update_baseline()
        return 0

    current = _current_modules()
    new_modules = current - BASELINE

    if new_modules:
        print("FAIL: New root-level modules detected (not in freeze baseline):")
        for m in sorted(new_modules):
            print(f"  + {m}")
        print()
        print("Root-level modules require justification. Either:")
        print("  1. Move the module into a subpackage with a re-export shim")
        print("  2. Run with --update after architectural review")
        return 1

    print(f"OK: {len(current)} root-level modules match baseline (no new additions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
