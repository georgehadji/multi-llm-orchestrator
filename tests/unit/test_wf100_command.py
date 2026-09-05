"""The `website-audit` command: what it prints and what it exits with."""

from __future__ import annotations

import argparse
import json

import pytest

from orchestrator.commands import website_audit

pytestmark = pytest.mark.unit

_GOOD_PAGE = """<!doctype html>
<html lang="el"><head><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kalamaria Dental — implants and hygiene in Kalamaria</title>
<meta name="description" content="A calm dental practice on Tsimiski street offering implants, hygiene and emergency care for families in Kalamaria.">
<link rel="canonical" href="https://k.example.gr/"></head>
<body><header><nav><a href="/">Home</a></nav></header>
<main><h1>Kalamaria Dental</h1><a href="tel:+302310111222">Call us</a></main>
<footer><a href="tel:+302310111222">Call</a><a href="/privacy.html">Privacy</a></footer>
</body></html>
"""


@pytest.fixture
def site(tmp_path):
    (tmp_path / "index.html").write_text(_GOOD_PAGE, encoding="utf-8")
    (tmp_path / "robots.txt").write_text(
        "User-agent: *\nAllow: /\nSitemap: https://k.example.gr/sitemap.xml\n", encoding="utf-8"
    )
    return tmp_path


def _run(argv):
    parser = argparse.ArgumentParser()
    website_audit.register(parser.add_subparsers())
    args = parser.parse_args(argv)
    with pytest.raises(SystemExit) as exit_info:
        args.func(args)
    return exit_info.value.code


class TestInvocation:
    def test_registers_the_subcommand(self):
        parser = argparse.ArgumentParser()
        website_audit.register(parser.add_subparsers())
        args = parser.parse_args(["website-audit", "somewhere"])
        assert args.target == "somewhere"

    def test_audits_a_directory_and_prints_a_report(self, site, capsys):
        _run(["website-audit", str(site)])
        printed = capsys.readouterr().out
        assert "WF-100 Website Quality Standard" in printed
        assert "OUTSTANDING" in printed

    def test_writes_markdown_to_a_file(self, site, tmp_path):
        out = tmp_path / "nested" / "report.md"
        _run(["website-audit", str(site), "--format", "markdown", "-o", str(out)])
        assert "# Website Quality Report" in out.read_text(encoding="utf-8")

    def test_json_output_is_machine_readable(self, site, tmp_path):
        out = tmp_path / "report.json"
        _run(["website-audit", str(site), "--format", "json", "-o", str(out)])
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["standard"] == "WF-100 v1.0"
        assert len(data["findings"]) == 100

    def test_a_missing_directory_exits_two_with_a_message(self, tmp_path, capsys):
        assert _run(["website-audit", str(tmp_path / "nope")]) == 2
        assert "not a directory" in capsys.readouterr().err

    def test_a_missing_record_file_exits_two(self, site, capsys):
        assert _run(["website-audit", str(site), "--record", "/nope/record.yaml"]) == 2
        assert "no business record" in capsys.readouterr().err


class TestExitCodes:
    def test_a_blocker_exits_one(self, tmp_path, capsys):
        # No viewport meta: phones render the desktop layout. Critical.
        (tmp_path / "index.html").write_text("<html lang='el'><main><h1>x</h1></main></html>")
        assert _run(["website-audit", str(tmp_path)]) == 1

    def test_a_site_that_cannot_reach_the_threshold_exits_one(self, site):
        # This fixture has no failures a human could not fix, but its ceiling is
        # below 90: resolving every outstanding check would still not get there.
        assert _run(["website-audit", str(site), "--fail-on", "threshold"]) == 1

    def test_fail_on_critical_ignores_outstanding_work(self, site):
        assert _run(["website-audit", str(site), "--fail-on", "critical"]) == 0

    def test_fail_on_critical_still_reports_a_blocker(self, tmp_path):
        (tmp_path / "index.html").write_text("<html lang='el'><main><h1>x</h1></main></html>")
        assert _run(["website-audit", str(tmp_path), "--fail-on", "critical"]) == 1

    def test_fail_on_outstanding_is_the_strictest(self, site):
        assert _run(["website-audit", str(site), "--fail-on", "outstanding"]) == 2


class TestRecordIntegration:
    def test_a_client_file_doubles_as_a_business_record(self, site, tmp_path, capsys):
        # The same file that renders a template also verifies the site built
        # from it, so the facts are stated once.
        record = tmp_path / "client.yaml"
        record.write_text(
            "name: Kalamaria Dental\nphone: '+30 2310 111222'\ncity: Kalamaria\n", encoding="utf-8"
        )
        _run(["website-audit", str(site), "--record", str(record)])
        printed = capsys.readouterr().out
        assert "No business record was supplied" not in printed

    def test_without_a_record_the_report_says_so(self, site, capsys):
        _run(["website-audit", str(site)])
        assert "No business record was supplied" in capsys.readouterr().out
