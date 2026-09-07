"""
Infrastructure Layer — Concrete Adapters
=========================================
Driven adapters: LLM provider clients, caches, databases, external services.
Depended on by the composition root (engine_core/container.py); must never
be imported directly by domain, application, or root-kernel modules.

This file's absence (the directory was an implicit PEP 420 namespace package)
made every import-linter contract naming `orchestrator.infrastructure` as
`forbidden_modules` vacuously pass: grimp's static graph builder never
registered this package or any module under it, so no import edge into it —
violating or not — could ever be found. A deliberate probe import
(`orchestrator/domain/__zz_probe.py` importing `infrastructure.llm_client`)
confirmed the Domain-purity contract reported KEPT against it before this
file existed. See docs/plans/2026-09-07-patterns-convergence-and-wire-or-delete.md
S7 for the investigation.
"""
