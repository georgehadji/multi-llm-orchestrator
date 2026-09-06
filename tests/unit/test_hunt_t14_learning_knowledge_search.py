"""
Hunt T14 — learning/, knowledge/, nexus_search/, pattern_learner/, context_mgmt/, analysis/
==============================================================================================
Regression tests for the defects fixed in wave T14 of the backend-remainder defect hunt
(docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md).

C1: knowledge/knowledge_base.py's shim imported a nonexistent KnowledgeEntry name.
C2: knowledge/knowledge_graph.py's shim imported nonexistent ModelPerformanceGraph/
    KnowledgeGraph names.
C3: learning/federated_learning.py was an unshimmed duplicate whose stripped import
    made contribute_insight() raise NameError on OutcomeStatus.
C4: learning/transfer_learning.py was an unshimmed duplicate whose stripped import
    (justified by a stale, incorrect "module does not exist" comment) made
    _create_routing_proposal()/_create_budget_proposal() raise NameError on StrategyType.
C5: performance.py::QueryOptimizer.build_selective_query() built raw SQL via unvalidated
    f-string interpolation of table/column/order_by names — the sibling dead copy in
    analysis/performance.py had already independently fixed this, never backported.
    Fixed at the canonical source; analysis/performance.py converted to a shim.
"""

from __future__ import annotations

import pytest

from orchestrator.performance import QueryOptimizer

pytestmark = pytest.mark.unit


def test_c1_knowledge_base_shim_matches_canonical() -> None:
    from orchestrator.knowledge.knowledge_base import KnowledgeBase as ViaShim
    from orchestrator.knowledge_base import KnowledgeBase as Canonical

    assert ViaShim is Canonical


def test_c2_knowledge_graph_shim_matches_canonical() -> None:
    from orchestrator.knowledge.knowledge_graph import PerformanceKnowledgeGraph as ViaShim
    from orchestrator.knowledge_graph import PerformanceKnowledgeGraph as Canonical

    assert ViaShim is Canonical


def test_c3_federated_learning_shim_matches_canonical() -> None:
    from orchestrator.federated_learning import FederatedLearningOrchestrator as Canonical
    from orchestrator.learning.federated_learning import (
        FederatedLearningOrchestrator as ViaShim,
    )

    assert ViaShim is Canonical


def test_c4_transfer_learning_shim_matches_canonical() -> None:
    from orchestrator.learning.transfer_learning import TransferLearningEngine as ViaShim
    from orchestrator.transfer_learning import TransferLearningEngine as Canonical

    assert ViaShim is Canonical


def test_c5_query_optimizer_rejects_injected_table_name() -> None:
    qo = QueryOptimizer()
    with pytest.raises(ValueError):
        qo.build_selective_query("tasks; DROP TABLE users;--")


def test_c5_query_optimizer_rejects_unallowlisted_table() -> None:
    qo = QueryOptimizer()
    with pytest.raises(ValueError):
        qo.build_selective_query("not_a_real_table")


def test_c5_query_optimizer_rejects_injected_column_name() -> None:
    qo = QueryOptimizer()
    with pytest.raises(ValueError):
        qo.build_selective_query("tasks", columns=["id, (SELECT password FROM users)"])


def test_c5_query_optimizer_no_regression_valid_query_still_builds() -> None:
    qo = QueryOptimizer()
    query = qo.build_selective_query(
        "tasks",
        columns=["id", "name"],
        where={"status": "active"},
        order_by="created_at",
        limit=10,
    )
    assert query == "SELECT id, name FROM tasks WHERE status = ? ORDER BY created_at LIMIT 10"


def test_c5_analysis_performance_shim_matches_canonical() -> None:
    from orchestrator.analysis.performance import QueryOptimizer as ViaShim

    assert ViaShim is QueryOptimizer
