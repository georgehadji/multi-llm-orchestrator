"""
Tests for Policy DSL — load_policy_dict, load_policy_file, PolicyAnalyzer.
Covers: JSON loading, YAML soft-dependency, _parse_policy, AnalysisReport,
        contradiction detection, impossible constraints, cross-policy conflicts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from orchestrator.policy import EnforcementMode, Policy, PolicyHierarchy, RateLimit
from orchestrator.policy_dsl import (
    AnalysisReport,
    PolicyAnalyzer,
    load_policy_dict,
    load_policy_file,
)


# ─────────────────────────────────────────────────────────────────────────────
# load_policy_dict — basic structure parsing
# ─────────────────────────────────────────────────────────────────────────────

def test_load_policy_dict_empty_returns_hierarchy():
    result = load_policy_dict({})
    assert isinstance(result, PolicyHierarchy)


def test_load_policy_dict_global_key_parsed():
    d = {"global": [{"name": "gdpr", "allow_training_on_output": False}]}
    h = load_policy_dict(d)
    policies = h.policies_for()
    assert len(policies) == 1
    assert policies[0].name == "gdpr"
    assert policies[0].allow_training_on_output is False


def test_load_policy_dict_org_alias_works():
    """'org' is an alias for 'global'."""
    d = {"org": [{"name": "org_policy"}]}
    h = load_policy_dict(d)
    assert len(h.policies_for()) == 1
    assert h.policies_for()[0].name == "org_policy"


def test_load_policy_dict_team_policies_parsed():
    d = {
        "team": {
            "eng": [{"name": "eu_only", "allowed_regions": ["eu", "global"]}]
        }
    }
    h = load_policy_dict(d)
    eng_policies = h.policies_for(team="eng")
    assert len(eng_policies) == 1
    assert eng_policies[0].name == "eu_only"
    assert eng_policies[0].allowed_regions == ["eu", "global"]


def test_load_policy_dict_job_policies_parsed():
    d = {
        "job": {
            "job_001": [{"name": "cost_cap", "max_cost_per_task_usd": 0.50}]
        }
    }
    h = load_policy_dict(d)
    policies = h.policies_for(job_id="job_001")
    assert len(policies) == 1
    assert policies[0].max_cost_per_task_usd == pytest.approx(0.50)


def test_load_policy_dict_node_policies_parsed():
    d = {
        "node": {
            "task_001": [{"name": "high_latency_ok", "max_latency_ms": 10000.0}]
        }
    }
    h = load_policy_dict(d)
    policies = h.policies_for(task_id="task_001")
    assert len(policies) == 1
    assert policies[0].max_latency_ms == pytest.approx(10000.0)


def test_load_policy_dict_all_levels_merged():
    d = {
        "global": [{"name": "gdpr"}],
        "team": {"eng": [{"name": "eu_only"}]},
        "job": {"j1": [{"name": "cost_cap", "max_cost_per_task_usd": 1.0}]},
        "node": {"t1": [{"name": "high_lat"}]},
    }
    h = load_policy_dict(d)
    # policies_for includes org + team + job + node
    all_p = h.policies_for(team="eng", job_id="j1", task_id="t1")
    names = [p.name for p in all_p]
    assert "gdpr" in names
    assert "eu_only" in names
    assert "cost_cap" in names
    assert "high_lat" in names


def test_load_policy_dict_different_team_is_isolated():
    d = {
        "team": {
            "eng": [{"name": "eng_policy"}],
            "data": [{"name": "data_policy"}],
        }
    }
    h = load_policy_dict(d)
    eng_policies = h.policies_for(team="eng")
    data_policies = h.policies_for(team="data")
    assert all(p.name == "eng_policy" for p in eng_policies)
    assert all(p.name == "data_policy" for p in data_policies)


# ─────────────────────────────────────────────────────────────────────────────
# _parse_policy — field parsing
# ─────────────────────────────────────────────────────────────────────────────

def test_enforcement_mode_string_parsed():
    d = {"global": [{"name": "test", "enforcement_mode": "soft"}]}
    h = load_policy_dict(d)
    p = h.policies_for()[0]
    assert p.enforcement_mode == EnforcementMode.SOFT


def test_enforcement_mode_hard_parsed():
    d = {"global": [{"name": "test", "enforcement_mode": "hard"}]}
    h = load_policy_dict(d)
    p = h.policies_for()[0]
    assert p.enforcement_mode == EnforcementMode.HARD


def test_enforcement_mode_monitor_parsed():
    d = {"global": [{"name": "test", "enforcement_mode": "monitor"}]}
    h = load_policy_dict(d)
    p = h.policies_for()[0]
    assert p.enforcement_mode == EnforcementMode.MONITOR


def test_unknown_enforcement_mode_defaults_to_none():
    """Unknown string → enforcement_mode=None (HARD default), with a warning logged."""
    d = {"global": [{"name": "test", "enforcement_mode": "foobar"}]}
    h = load_policy_dict(d)
    p = h.policies_for()[0]
    assert p.enforcement_mode is None


def test_rate_limit_parsed():
    d = {
        "global": [{
            "name": "rate_test",
            "rate_limit": {"calls_per_minute": 60, "cost_usd_per_hour": 5.0}
        }]
    }
    h = load_policy_dict(d)
    p = h.policies_for()[0]
    assert p.rate_limit is not None
    assert p.rate_limit.calls_per_minute == 60
    assert p.rate_limit.cost_usd_per_hour == pytest.approx(5.0)


def test_blocked_models_parsed():
    """Valid model strings should be converted to Model enums."""
    from orchestrator.models import Model
    d = {
        "global": [{
            "name": "no_kimi",
            "blocked_models": ["kimi-k2.5"]  # Model.KIMI_K2_5.value
        }]
    }
    h = load_policy_dict(d)
    p = h.policies_for()[0]
    assert p.blocked_models is not None
    assert Model.KIMI_K2_5 in p.blocked_models


def test_unknown_blocked_model_is_skipped():
    """Unknown model strings should be skipped (not crash)."""
    d = {
        "global": [{
            "name": "test",
            "blocked_models": ["does-not-exist-model"]
        }]
    }
    h = load_policy_dict(d)
    p = h.policies_for()[0]
    # Parsed models list should be empty (None) since all models were unknown
    assert p.blocked_models is None


# ─────────────────────────────────────────────────────────────────────────────
# load_policy_file — JSON
# ─────────────────────────────────────────────────────────────────────────────

def test_load_json_policy_file(tmp_path):
    data = {"global": [{"name": "gdpr", "allow_training_on_output": False}]}
    path = tmp_path / "policies.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    h = load_policy_file(path)
    policies = h.policies_for()
    assert len(policies) == 1
    assert policies[0].name == "gdpr"


def test_load_json_policy_file_string_path(tmp_path):
    data = {"global": [{"name": "p1"}]}
    path = tmp_path / "p.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    h = load_policy_file(str(path))
    assert len(h.policies_for()) == 1


def test_load_policy_file_unsupported_extension(tmp_path):
    path = tmp_path / "policies.toml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported policy file extension"):
        load_policy_file(path)


def test_load_policy_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        load_policy_file("/nonexistent/path/policies.json")


def test_load_policy_file_non_dict_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a YAML/JSON object"):
        load_policy_file(path)


# ─────────────────────────────────────────────────────────────────────────────
# load_policy_file — YAML (soft dependency)
# ─────────────────────────────────────────────────────────────────────────────

def test_load_yaml_file_skip_if_no_pyyaml(tmp_path, monkeypatch):
    """If pyyaml is not installed, loading .yml must raise ImportError."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("mocked: pyyaml not installed")
        return real_import(name, *args, **kwargs)

    path = tmp_path / "policies.yml"
    path.write_text("global: []", encoding="utf-8")

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ImportError, match="pyyaml"):
        load_policy_file(path)


def test_load_yaml_file_if_pyyaml_available(tmp_path):
    """Only run if pyyaml is actually installed."""
    pytest.importorskip("yaml")
    data = "global:\n  - name: yaml_policy\n    allow_training_on_output: false\n"
    path = tmp_path / "policies.yaml"
    path.write_text(data, encoding="utf-8")
    h = load_policy_file(path)
    policies = h.policies_for()
    assert len(policies) == 1
    assert policies[0].name == "yaml_policy"
    assert policies[0].allow_training_on_output is False


# ─────────────────────────────────────────────────────────────────────────────
# AnalysisReport
# ─────────────────────────────────────────────────────────────────────────────

def test_analysis_report_is_clean_no_errors_no_warnings():
    r = AnalysisReport(errors=[], warnings=[], info=["some info"])
    assert r.is_clean() is True


def test_analysis_report_not_clean_with_errors():
    r = AnalysisReport(errors=["problem"], warnings=[], info=[])
    assert r.is_clean() is False


def test_analysis_report_not_clean_with_warnings():
    r = AnalysisReport(errors=[], warnings=["caution"], info=[])
    assert r.is_clean() is False


# ─────────────────────────────────────────────────────────────────────────────
# PolicyAnalyzer — contradiction detection
# ─────────────────────────────────────────────────────────────────────────────

def test_overlap_allowed_blocked_providers_is_error():
    p = Policy(
        name="test",
        allowed_providers=["openai", "anthropic"],
        blocked_providers=["openai"],
    )
    report = PolicyAnalyzer.analyze([p])
    assert len(report.errors) >= 1
    assert any("openai" in e for e in report.errors)


def test_no_overlap_allowed_blocked_is_clean():
    p = Policy(
        name="test",
        allowed_providers=["openai"],
        blocked_providers=["anthropic"],
    )
    report = PolicyAnalyzer.analyze([p])
    assert not any("overlap" in e.lower() for e in report.errors)


def test_empty_allowed_regions_is_error():
    p = Policy(name="test", allowed_regions=[])
    report = PolicyAnalyzer.analyze([p])
    assert len(report.errors) >= 1
    assert any("impossible" in e.lower() or "blocks all" in e.lower() for e in report.errors)


def test_non_empty_allowed_regions_is_not_error():
    p = Policy(name="test", allowed_regions=["eu"])
    report = PolicyAnalyzer.analyze([p])
    # No error about allowed_regions
    assert not any("blocks all" in e.lower() for e in report.errors)


def test_disjoint_allowed_providers_across_two_policies_is_warning():
    p1 = Policy(name="only_openai", allowed_providers=["openai"])
    p2 = Policy(name="only_google", allowed_providers=["google"])
    report = PolicyAnalyzer.analyze([p1, p2])
    assert len(report.warnings) >= 1


def test_overlapping_allowed_providers_across_two_policies_no_warning():
    p1 = Policy(name="p1", allowed_providers=["openai", "google"])
    p2 = Policy(name="p2", allowed_providers=["openai"])
    report = PolicyAnalyzer.analyze([p1, p2])
    # Should not warn since they share "openai"
    assert not any("no common" in w.lower() for w in report.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# PolicyAnalyzer — coverage info
# ─────────────────────────────────────────────────────────────────────────────

def test_no_cost_cap_produces_info(  ):
    p = Policy(name="p1")
    report = PolicyAnalyzer.analyze([p])
    assert any("cost cap" in i.lower() for i in report.info)


def test_has_cost_cap_no_cost_info():
    p = Policy(name="p1", max_cost_per_task_usd=0.5)
    report = PolicyAnalyzer.analyze([p])
    assert not any("cost cap" in i.lower() for i in report.info)


def test_no_latency_sla_produces_info():
    p = Policy(name="p1")
    report = PolicyAnalyzer.analyze([p])
    assert any("latency" in i.lower() for i in report.info)


def test_has_latency_sla_no_latency_info():
    p = Policy(name="p1", max_latency_ms=3000.0)
    report = PolicyAnalyzer.analyze([p])
    assert not any("latency sla" in i.lower() for i in report.info)


def test_empty_policy_list_is_clean():
    report = PolicyAnalyzer.analyze([])
    assert len(report.errors) == 0
    assert len(report.warnings) == 0
    # Info messages for missing cost cap and latency SLA
    assert len(report.info) == 2


def test_multiple_errors_accumulated():
    p1 = Policy(name="bad1", allowed_providers=["a"], blocked_providers=["a"])
    p2 = Policy(name="bad2", allowed_regions=[])
    report = PolicyAnalyzer.analyze([p1, p2])
    assert len(report.errors) >= 2
