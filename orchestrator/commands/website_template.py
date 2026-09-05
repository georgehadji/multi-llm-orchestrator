"""
`website-template` — build a vertical once, ship it per client.

    website-template list
    website-template extract <site> -o templates/websites/dentist --name dentist
    website-template apply  <template> --client clients/kalamaria.yaml -o outputs/kalamaria

`extract` turns a site you already built into a reusable template plus a
starting client file; `apply` renders that template for the next client. Both
are deterministic and free — no model calls — which is the point: the second
dentist should not cost what the first one did.
"""

from __future__ import annotations

import sys
from pathlib import Path


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "website-template",
        help="Create and reuse site templates (build one dentist site, ship many)",
    )
    sub = parser.add_subparsers(dest="template_action", required=True)

    lst = sub.add_parser("list", help="List available site templates")
    lst.add_argument("--root", default="", help="Template directory (default: templates/websites)")

    ext = sub.add_parser("extract", help="Turn a built site into a reusable template")
    ext.add_argument("site", help="Directory of the site to learn from")
    ext.add_argument("--output", "-o", required=True, help="Where to write the template")
    ext.add_argument("--name", default="", help="Template name (default: the site's folder name)")

    app = sub.add_parser("apply", help="Render a template for one client")
    app.add_argument("template", help="Template directory")
    app.add_argument("--client", "-c", required=True, help="Client YAML/JSON data file")
    app.add_argument("--output", "-o", required=True, help="Where to write the site")

    parser.set_defaults(func=execute)


def _load_client(path: Path) -> dict[str, object]:
    import json

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text) or {}
    return json.loads(text)


def execute(args) -> None:
    from ..generators.website_template import (
        apply_template,
        extract_template,
        list_templates,
        load_template,
    )

    action = getattr(args, "template_action", None)

    if action == "list":
        templates = list_templates(args.root or None)
        if not templates:
            print("No site templates found. Create one with `website-template extract`.")
            return
        print(f"{len(templates)} template(s):\n")
        for tpl in templates:
            print(f"  {tpl.name:20s} {len(tpl.sections):2d} sections  {tpl.description}")
            print(f"    sections: {', '.join(tpl.sections)}")
            if tpl.required:
                print(f"    required: {', '.join(tpl.required)}")
        return

    if action == "extract":
        site = Path(args.site)
        if not site.is_dir():
            print(f"[FAIL] not a directory: {site}")
            sys.exit(1)
        try:
            out = extract_template(site, Path(args.output), name=args.name or None)
        except ValueError as exc:
            print(f"[FAIL] {exc}")
            sys.exit(1)
        tpl = load_template(out)
        print(f"[OK] template '{tpl.name}' written to {out}")
        print(f"     sections: {', '.join(tpl.sections)}")
        print(f"     starting client data: {out / 'client.example.yaml'}")
        print(f"\nNext: cp {out / 'client.example.yaml'} client.yaml, edit it, then")
        print(f"  python -m orchestrator website-template apply {out} -c client.yaml -o <site>")
        return

    # apply
    try:
        template = load_template(Path(args.template))
        client = _load_client(Path(args.client))
        out = apply_template(template, client, Path(args.output))
    except (ValueError, OSError) as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)
    print(f"[OK] {template.name} -> {out.resolve()}")
    print(
        f"     {len(template.sections)} sections rendered for "
        f"{client.get('brand_name', 'this client')}"
    )
