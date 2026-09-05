"""CLI surface for the reusable-template workflow."""

from __future__ import annotations

import argparse

import pytest

pytestmark = pytest.mark.unit


def _parser():
    from orchestrator.commands import website_template

    parser = argparse.ArgumentParser()
    website_template.register(parser.add_subparsers())
    return parser


@pytest.mark.unit
class TestWebsiteTemplateCommand:
    def test_is_auto_discovered(self):
        from orchestrator.commands import discover_command_modules

        assert "website_template" in discover_command_modules()

    def test_list_parses(self):
        args = _parser().parse_args(["website-template", "list"])
        assert args.template_action == "list"

    def test_extract_parses(self):
        args = _parser().parse_args(
            ["website-template", "extract", "site", "-o", "tpl", "--name", "dentist"]
        )
        assert args.site == "site" and args.output == "tpl" and args.name == "dentist"

    def test_apply_parses(self):
        args = _parser().parse_args(
            ["website-template", "apply", "tpl", "-c", "client.yaml", "-o", "out"]
        )
        assert args.template == "tpl" and args.client == "client.yaml"

    def test_an_action_is_required(self):
        with pytest.raises(SystemExit):
            _parser().parse_args(["website-template"])
