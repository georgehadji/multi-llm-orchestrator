"""
Unit tests for the batch website factory.

The factory turns a manifest of site specs into N generated, gated, and
optionally git-committed sites. Design constraints these tests lock down:

- One bad site must NOT abort the batch. In a 50-site run, site 7 failing is
  a row in the report, not the end of the run.
- Every outcome carries a verdict. A site that was never judged must not be
  reported as shippable.
- Delivery is per-site git repos, initialised and committed locally.
- Manifests accept YAML or JSON; `defaults` merge into each site entry.
"""

from __future__ import annotations

import json
import subprocess

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def factory_mod():
    from orchestrator.generators import website_factory

    return website_factory


def _fake_result(success=True, gate=True, score=0.9, cost=0.12, failures=()):
    from orchestrator.design_system import QualityCheck, QualityReport
    from orchestrator.generators.website_generator import WebsiteBuildResult

    checks = [QualityCheck(name="c", passed=not failures, score=score, details="")]
    result = WebsiteBuildResult(output_dir="", success=success)
    result.quality_report = QualityReport(checks=checks, score=score)
    result.quality_gate_passed = gate
    result.gate_failures = list(failures)
    result.total_cost = cost
    return result


@pytest.mark.unit
class TestManifestLoading:
    def test_loads_json_manifest_with_defaults_merged(self, factory_mod, tmp_path):
        manifest = tmp_path / "sites.json"
        manifest.write_text(
            json.dumps(
                {
                    "defaults": {"framework": "html", "min_quality": 0.85},
                    "sites": [
                        {"slug": "a", "description": "Site A"},
                        {"slug": "b", "description": "Site B", "min_quality": 0.5},
                    ],
                }
            ),
            encoding="utf-8",
        )
        specs = factory_mod.load_manifest(manifest)
        assert [s.slug for s in specs] == ["a", "b"]
        assert specs[0].framework == "html"
        assert specs[0].min_quality == pytest.approx(0.85)
        assert specs[1].min_quality == pytest.approx(0.5), "per-site value must win over defaults"

    def test_loads_yaml_manifest(self, factory_mod, tmp_path):
        pytest.importorskip("yaml")
        manifest = tmp_path / "sites.yaml"
        manifest.write_text(
            "defaults:\n  framework: html\nsites:\n"
            "  - slug: dental\n    description: A dental clinic\n",
            encoding="utf-8",
        )
        specs = factory_mod.load_manifest(manifest)
        assert specs[0].slug == "dental"
        assert specs[0].framework == "html"

    def test_missing_slug_is_a_clear_error(self, factory_mod, tmp_path):
        manifest = tmp_path / "sites.json"
        manifest.write_text(json.dumps({"sites": [{"description": "no slug"}]}), encoding="utf-8")
        with pytest.raises(ValueError, match="slug"):
            factory_mod.load_manifest(manifest)

    def test_missing_description_is_a_clear_error(self, factory_mod, tmp_path):
        manifest = tmp_path / "sites.json"
        manifest.write_text(json.dumps({"sites": [{"slug": "x"}]}), encoding="utf-8")
        with pytest.raises(ValueError, match="description"):
            factory_mod.load_manifest(manifest)

    def test_slug_must_be_path_safe(self, factory_mod, tmp_path):
        """A slug becomes a directory name — reject traversal outright."""
        manifest = tmp_path / "sites.json"
        manifest.write_text(
            json.dumps({"sites": [{"slug": "../escape", "description": "d"}]}), encoding="utf-8"
        )
        with pytest.raises(ValueError):
            factory_mod.load_manifest(manifest)

    def test_empty_sites_list_is_an_error(self, factory_mod, tmp_path):
        manifest = tmp_path / "sites.json"
        manifest.write_text(json.dumps({"sites": []}), encoding="utf-8")
        with pytest.raises(ValueError):
            factory_mod.load_manifest(manifest)


@pytest.mark.unit
class TestBatchRun:
    @pytest.mark.asyncio
    async def test_runs_every_site_and_reports_each(self, factory_mod, tmp_path):
        specs = [factory_mod.SiteSpec(slug=f"s{i}", description=f"Site {i}") for i in range(3)]

        async def build(spec, output_dir):
            (output_dir / "index.html").write_text("<html></html>", encoding="utf-8")
            return _fake_result()

        factory = factory_mod.WebsiteFactory(build_site=build, git_delivery=False)
        outcomes = await factory.run(specs, output_root=tmp_path)
        assert [o.slug for o in outcomes] == ["s0", "s1", "s2"]
        assert all(o.quality_gate_passed for o in outcomes)

    @pytest.mark.asyncio
    async def test_one_failing_site_does_not_abort_the_batch(self, factory_mod, tmp_path):
        specs = [factory_mod.SiteSpec(slug=f"s{i}", description="d") for i in range(3)]

        async def build(spec, output_dir):
            if spec.slug == "s1":
                raise RuntimeError("model exploded")
            return _fake_result()

        factory = factory_mod.WebsiteFactory(build_site=build, git_delivery=False)
        outcomes = await factory.run(specs, output_root=tmp_path)
        assert len(outcomes) == 3
        bad = [o for o in outcomes if o.slug == "s1"][0]
        assert bad.success is False
        assert "model exploded" in bad.error
        assert all(o.success for o in outcomes if o.slug != "s1")

    @pytest.mark.asyncio
    async def test_crashed_site_is_never_reported_shippable(self, factory_mod, tmp_path):
        specs = [factory_mod.SiteSpec(slug="s", description="d")]

        async def build(spec, output_dir):
            raise RuntimeError("boom")

        factory = factory_mod.WebsiteFactory(build_site=build, git_delivery=False)
        outcomes = await factory.run(specs, output_root=tmp_path)
        assert outcomes[0].quality_gate_passed is not True

    @pytest.mark.asyncio
    async def test_writes_results_manifest(self, factory_mod, tmp_path):
        specs = [factory_mod.SiteSpec(slug="s", description="d")]

        async def build(spec, output_dir):
            return _fake_result(cost=0.25)

        factory = factory_mod.WebsiteFactory(build_site=build, git_delivery=False)
        await factory.run(specs, output_root=tmp_path)
        report = tmp_path / "factory-report.json"
        assert report.exists()
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["sites"][0]["slug"] == "s"
        assert data["total_cost_usd"] == pytest.approx(0.25)

    @pytest.mark.asyncio
    async def test_concurrency_is_bounded(self, factory_mod, tmp_path):
        import asyncio

        specs = [factory_mod.SiteSpec(slug=f"s{i}", description="d") for i in range(6)]
        live = 0
        peak = 0

        async def build(spec, output_dir):
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            live -= 1
            return _fake_result()

        factory = factory_mod.WebsiteFactory(build_site=build, concurrency=2, git_delivery=False)
        await factory.run(specs, output_root=tmp_path)
        assert peak <= 2, f"concurrency limit not honoured (peak={peak})"

    @pytest.mark.asyncio
    async def test_summary_counts_shippable_and_rejected(self, factory_mod, tmp_path):
        specs = [factory_mod.SiteSpec(slug=f"s{i}", description="d") for i in range(3)]

        async def build(spec, output_dir):
            return _fake_result(
                gate=spec.slug != "s1", failures=() if spec.slug != "s1" else ("x",)
            )

        factory = factory_mod.WebsiteFactory(build_site=build, git_delivery=False)
        await factory.run(specs, output_root=tmp_path)
        data = json.loads((tmp_path / "factory-report.json").read_text(encoding="utf-8"))
        assert data["shippable"] == 2
        assert data["rejected"] == 1


@pytest.mark.unit
class TestGitDelivery:
    @staticmethod
    def _git(*args, cwd):
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)

    @pytest.mark.asyncio
    async def test_each_site_becomes_its_own_repo_with_a_commit(self, factory_mod, tmp_path):
        specs = [factory_mod.SiteSpec(slug="dental", description="d")]

        async def build(spec, output_dir):
            (output_dir / "index.html").write_text("<html>hi</html>", encoding="utf-8")
            return _fake_result()

        factory = factory_mod.WebsiteFactory(build_site=build, git_delivery=True)
        outcomes = await factory.run(specs, output_root=tmp_path)

        site_dir = tmp_path / "dental"
        assert (site_dir / ".git").is_dir(), "each site must be its own repository"
        log = self._git("log", "--oneline", cwd=site_dir)
        assert log.returncode == 0 and log.stdout.strip(), "expected a commit"
        assert outcomes[0].committed is True
        assert outcomes[0].commit_sha

    @pytest.mark.asyncio
    async def test_git_delivery_can_be_disabled(self, factory_mod, tmp_path):
        specs = [factory_mod.SiteSpec(slug="plain", description="d")]

        async def build(spec, output_dir):
            (output_dir / "index.html").write_text("<html></html>", encoding="utf-8")
            return _fake_result()

        factory = factory_mod.WebsiteFactory(build_site=build, git_delivery=False)
        outcomes = await factory.run(specs, output_root=tmp_path)
        assert not (tmp_path / "plain" / ".git").exists()
        assert outcomes[0].committed is False

    @pytest.mark.asyncio
    async def test_rejected_site_is_still_committed_for_inspection(self, factory_mod, tmp_path):
        """A rejected build is evidence — keep it in version control to diff against."""
        specs = [factory_mod.SiteSpec(slug="bad", description="d")]

        async def build(spec, output_dir):
            (output_dir / "index.html").write_text("<html></html>", encoding="utf-8")
            return _fake_result(gate=False, score=0.2, failures=("SEO Basics (score 0.10)",))

        factory = factory_mod.WebsiteFactory(build_site=build, git_delivery=True)
        outcomes = await factory.run(specs, output_root=tmp_path)
        assert outcomes[0].quality_gate_passed is False
        assert outcomes[0].committed is True


@pytest.mark.unit
class TestWebsiteBatchCommand:
    @staticmethod
    def _parser():
        import argparse

        from orchestrator.commands import website_batch

        parser = argparse.ArgumentParser()
        website_batch.register(parser.add_subparsers())
        return parser

    def test_registers_subcommand_with_manifest(self):
        args = self._parser().parse_args(["website-batch", "sites.yaml"])
        assert args.manifest == "sites.yaml"

    def test_defaults_are_sane(self):
        args = self._parser().parse_args(["website-batch", "sites.yaml"])
        assert args.concurrency >= 1
        assert args.output_root
        assert args.no_git is False, "git-per-site delivery is the default"

    def test_flags_parse(self):
        args = self._parser().parse_args(
            ["website-batch", "s.yaml", "-o", "out", "-c", "5", "--no-git"]
        )
        assert args.output_root == "out"
        assert args.concurrency == 5
        assert args.no_git is True

    def test_is_auto_discovered_as_a_command(self):
        from orchestrator.commands import discover_command_modules

        assert "website_batch" in discover_command_modules()

    @pytest.mark.parametrize(
        "outcomes,expected",
        [
            ([(True, True)], 0),
            ([(True, True), (True, None)], 0),
            ([(True, True), (True, False)], 2),
            ([(True, True), (False, False)], 1),
            ([(False, False), (True, False)], 1),
        ],
    )
    def test_exit_code_reflects_worst_outcome(self, outcomes, expected):
        """1 beats 2 beats 0: a build failure is worse news than a gate rejection."""
        from orchestrator.commands.website_batch import exit_code_for
        from orchestrator.generators.website_factory import SiteOutcome

        rows = [
            SiteOutcome(slug=f"s{i}", output_dir="", success=s, quality_gate_passed=g)
            for i, (s, g) in enumerate(outcomes)
        ]
        assert exit_code_for(rows) == expected
