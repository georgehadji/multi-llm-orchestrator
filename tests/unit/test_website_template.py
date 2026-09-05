"""
Reusable site templates: build a dentist site once, reuse it for the next dentist.

The factory generates every site from scratch, so site #2 for the same vertical
costs the same as site #1 — in money, latency and inconsistency. A template
captures structure and styling once; per-client data supplies only what actually
differs (brand, copy, contact details, palette).

Deliberately NOT a template engine. Substitution is stdlib `string.Template`
plus one repeat-block convention, because the alternative is taking a Jinja2
dependency to render what is essentially name/value replacement.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def mod():
    from orchestrator.generators import website_template

    return website_template


# ── Rendering ────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRenderSection:
    def test_substitutes_flat_placeholders(self, mod):
        out = mod.render_section("<h1>$brand_name</h1>", {"brand_name": "Aegean Dental"})
        assert out == "<h1>Aegean Dental</h1>"

    def test_leaves_unknown_placeholders_visible_rather_than_crashing(self, mod):
        """A missing value must be obvious in the output, not an exception mid-batch."""
        out = mod.render_section("<p>$phone</p>", {})
        assert "$phone" in out or "TODO" in out

    def test_repeat_block_renders_once_per_item(self, mod):
        html = "<ul>\n<!-- repeat: services -->\n<li>$title — $body</li>\n<!-- /repeat -->\n</ul>"
        out = mod.render_section(
            html,
            {
                "services": [
                    {"title": "Implants", "body": "Titanium roots"},
                    {"title": "Whitening", "body": "In-chair"},
                ]
            },
        )
        assert out.count("<li>") == 2
        assert "Implants — Titanium roots" in out
        assert "Whitening — In-chair" in out
        assert "repeat:" not in out

    def test_empty_repeat_list_removes_the_block(self, mod):
        html = "<ul>\n<!-- repeat: services -->\n<li>$title</li>\n<!-- /repeat -->\n</ul>"
        out = mod.render_section(html, {"services": []})
        assert "<li>" not in out and "repeat" not in out

    def test_repeat_items_can_use_outer_values(self, mod):
        html = "<!-- repeat: items -->\n<p>$brand_name: $title</p>\n<!-- /repeat -->"
        out = mod.render_section(html, {"brand_name": "Aegean", "items": [{"title": "X"}]})
        assert "Aegean: X" in out

    def test_dollar_signs_in_content_survive(self, mod):
        out = mod.render_section("<p>$price</p>", {"price": "$50"})
        assert "$50" in out


# ── Template definition ──────────────────────────────────────────────────────


def _write_template(root):
    tpl = root / "dentist"
    (tpl / "sections").mkdir(parents=True, exist_ok=True)
    (tpl / "template.yaml").write_text(
        "name: dentist\n"
        "description: Dental clinic\n"
        "sections: [hero, services]\n"
        "required: [brand_name, phone]\n"
        "defaults:\n"
        "  tagline: Modern dentistry\n",
        encoding="utf-8",
    )
    (tpl / "sections" / "hero.html").write_text(
        '<section id="hero"><h1>$brand_name</h1><p>$tagline</p></section>', encoding="utf-8"
    )
    (tpl / "sections" / "services.html").write_text(
        '<section id="services">\n<!-- repeat: services -->\n<article><h3>$title</h3></article>\n'
        "<!-- /repeat -->\n</section>",
        encoding="utf-8",
    )
    (tpl / "styles.css").write_text(
        "/* mobile first */\nbody{margin:0}\n@media (min-width:48em){.g{display:grid}}\n",
        encoding="utf-8",
    )
    return tpl


@pytest.mark.unit
class TestLoadTemplate:
    def test_loads_metadata_and_sections(self, mod, tmp_path):
        tpl = mod.load_template(_write_template(tmp_path))
        assert tpl.name == "dentist"
        assert tpl.sections == ["hero", "services"]
        assert tpl.required == ["brand_name", "phone"]
        assert tpl.defaults["tagline"] == "Modern dentistry"

    def test_missing_section_file_is_a_clear_error(self, mod, tmp_path):
        tpl_dir = _write_template(tmp_path)
        (tpl_dir / "sections" / "hero.html").unlink()
        with pytest.raises(ValueError, match="hero"):
            mod.load_template(tpl_dir)

    def test_missing_template_yaml_is_a_clear_error(self, mod, tmp_path):
        tpl_dir = _write_template(tmp_path)
        (tpl_dir / "template.yaml").unlink()
        with pytest.raises((ValueError, FileNotFoundError)):
            mod.load_template(tpl_dir)


# ── Applying a template to a client ──────────────────────────────────────────


@pytest.mark.unit
class TestApplyTemplate:
    def test_produces_a_site(self, mod, tmp_path):
        tpl = mod.load_template(_write_template(tmp_path))
        out = tmp_path / "site"
        mod.apply_template(
            tpl,
            {
                "brand_name": "Aegean Dental",
                "phone": "+30 2310 000000",
                "services": [{"title": "Implants"}],
            },
            out,
        )
        html = (out / "index.html").read_text(encoding="utf-8")
        assert "Aegean Dental" in html
        assert "Implants" in html
        assert (out / "styles.css").exists()

    def test_defaults_fill_unspecified_values(self, mod, tmp_path):
        tpl = mod.load_template(_write_template(tmp_path))
        out = tmp_path / "site2"
        mod.apply_template(tpl, {"brand_name": "B", "phone": "1"}, out)
        assert "Modern dentistry" in (out / "index.html").read_text(encoding="utf-8")

    def test_missing_required_field_is_rejected(self, mod, tmp_path):
        tpl = mod.load_template(_write_template(tmp_path))
        with pytest.raises(ValueError, match="phone"):
            mod.apply_template(tpl, {"brand_name": "B"}, tmp_path / "site3")

    def test_sections_appear_in_declared_order(self, mod, tmp_path):
        tpl = mod.load_template(_write_template(tmp_path))
        out = tmp_path / "site4"
        mod.apply_template(tpl, {"brand_name": "B", "phone": "1"}, out)
        html = (out / "index.html").read_text(encoding="utf-8")
        assert html.index('id="hero"') < html.index('id="services"')

    def test_output_is_a_complete_document(self, mod, tmp_path):
        tpl = mod.load_template(_write_template(tmp_path))
        out = tmp_path / "site5"
        mod.apply_template(tpl, {"brand_name": "B", "phone": "1"}, out)
        html = (out / "index.html").read_text(encoding="utf-8")
        assert html.lstrip().startswith("<!DOCTYPE html>")
        assert "<html lang=" in html
        assert 'name="viewport"' in html

    def test_two_clients_differ_only_in_content(self, mod, tmp_path):
        """The whole point: same structure, different client."""
        tpl = mod.load_template(_write_template(tmp_path))
        a, b = tmp_path / "a", tmp_path / "b"
        mod.apply_template(tpl, {"brand_name": "Clinic A", "phone": "1"}, a)
        mod.apply_template(tpl, {"brand_name": "Clinic B", "phone": "2"}, b)
        assert (a / "styles.css").read_text() == (b / "styles.css").read_text()
        assert "Clinic A" in (a / "index.html").read_text()
        assert "Clinic B" in (b / "index.html").read_text()
        assert "Clinic A" not in (b / "index.html").read_text()


# ── Extracting a template from a site you already built ──────────────────────


@pytest.mark.unit
class TestExtractTemplate:
    @staticmethod
    def _built_site(tmp_path):
        site = tmp_path / "built"
        site.mkdir(parents=True, exist_ok=True)
        (site / "index.html").write_text(
            '<!DOCTYPE html>\n<html lang="en"><head>'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<title>Thessaloniki Dental — Implants</title>"
            '<meta name="description" content="A dental clinic in Thessaloniki.">'
            '<link rel="stylesheet" href="styles.css"></head><body>'
            '<section id="hero"><h1>Thessaloniki Dental</h1></section>'
            '<section id="contact"><p>+30 2310 123456</p><p>hi@example.gr</p></section>'
            "</body></html>",
            encoding="utf-8",
        )
        (site / "styles.css").write_text("body{margin:0}", encoding="utf-8")
        return site

    def test_creates_a_loadable_template(self, mod, tmp_path):
        out = tmp_path / "tpl"
        mod.extract_template(self._built_site(tmp_path), out, name="dental")
        tpl = mod.load_template(out)
        assert tpl.name == "dental"
        assert tpl.sections, "extraction found no sections"

    def test_splits_sections_into_separate_files(self, mod, tmp_path):
        out = tmp_path / "tpl2"
        mod.extract_template(self._built_site(tmp_path), out, name="dental")
        assert (out / "sections" / "hero.html").exists()
        assert (out / "sections" / "contact.html").exists()

    def test_placeholders_replace_obvious_client_content(self, mod, tmp_path):
        out = tmp_path / "tpl3"
        mod.extract_template(self._built_site(tmp_path), out, name="dental")
        hero = (out / "sections" / "hero.html").read_text(encoding="utf-8")
        assert "$brand_name" in hero, f"brand not parameterised: {hero}"
        contact = (out / "sections" / "contact.html").read_text(encoding="utf-8")
        assert "$phone" in contact and "$email" in contact

    def test_writes_starting_client_data(self, mod, tmp_path):
        """Extraction must hand back the values it pulled out, ready to edit."""
        import yaml

        out = tmp_path / "tpl4"
        mod.extract_template(self._built_site(tmp_path), out, name="dental")
        data = yaml.safe_load((out / "client.example.yaml").read_text(encoding="utf-8"))
        assert data["brand_name"] == "Thessaloniki Dental"
        assert data["phone"] == "+30 2310 123456"
        assert data["email"] == "hi@example.gr"

    def test_round_trip_reproduces_the_brand(self, mod, tmp_path):
        """extract -> apply with the extracted data must rebuild a working page."""
        import yaml

        out = tmp_path / "tpl5"
        mod.extract_template(self._built_site(tmp_path), out, name="dental")
        data = yaml.safe_load((out / "client.example.yaml").read_text(encoding="utf-8"))
        site = tmp_path / "rebuilt"
        mod.apply_template(mod.load_template(out), data, site)
        html = (site / "index.html").read_text(encoding="utf-8")
        assert "Thessaloniki Dental" in html
        assert "$brand_name" not in html

    def test_round_trip_then_rebrand(self, mod, tmp_path):
        """The actual user story: reuse a built dentist site for another dentist."""
        import yaml

        out = tmp_path / "tpl6"
        mod.extract_template(self._built_site(tmp_path), out, name="dental")
        data = yaml.safe_load((out / "client.example.yaml").read_text(encoding="utf-8"))
        data.update({"brand_name": "Kalamaria Dental", "phone": "+30 2310 999999"})
        site = tmp_path / "client2"
        mod.apply_template(mod.load_template(out), data, site)
        html = (site / "index.html").read_text(encoding="utf-8")
        assert "Kalamaria Dental" in html and "+30 2310 999999" in html
        assert "Thessaloniki Dental" not in html


@pytest.mark.unit
class TestExtractionIsNotFooled:
    """Extraction runs unattended over sites nobody curated for it.

    Both cases below are real: extracting from thessaloniki-dental produced
    brand_name='THE FUTURE OF<br><span class="gradient-text">DENTISTRY</span>'
    and phone='1606811841689-23' — a cache-busting timestamp that happened to
    match a loose digit pattern.
    """

    @staticmethod
    def _site(tmp_path, body: str, head: str = ""):
        site = tmp_path / "s"
        site.mkdir(parents=True, exist_ok=True)
        (site / "index.html").write_text(
            f'<!DOCTYPE html><html lang="en"><head><title>T</title>{head}</head>'
            f"<body>{body}</body></html>",
            encoding="utf-8",
        )
        (site / "styles.css").write_text("body{margin:0}", encoding="utf-8")
        return site

    def test_brand_name_is_plain_text_not_markup(self, mod, tmp_path):
        import yaml

        site = self._site(
            tmp_path,
            '<section id="hero"><h1>THE FUTURE OF<br><span class="g">DENTISTRY</span></h1></section>',
        )
        out = tmp_path / "t"
        mod.extract_template(site, out, name="x")
        brand = yaml.safe_load((out / "client.example.yaml").read_text())["brand_name"]
        assert "<" not in brand and ">" not in brand, f"markup leaked into brand: {brand!r}"
        assert brand == "THE FUTURE OF DENTISTRY"

    def test_timestamp_is_not_mistaken_for_a_phone_number(self, mod, tmp_path):
        import yaml

        site = self._site(
            tmp_path,
            '<section id="hero"><h1>Clinic</h1>'
            '<img src="hero.webp?v=1606811841689-23" alt="x"></section>',
        )
        out = tmp_path / "t2"
        mod.extract_template(site, out, name="x")
        data = yaml.safe_load((out / "client.example.yaml").read_text())
        assert "1606811841689" not in str(data.get("phone", "")), f"got {data.get('phone')!r}"

    def test_tel_link_is_the_preferred_phone_source(self, mod, tmp_path):
        import yaml

        site = self._site(
            tmp_path,
            '<section id="c"><h1>Clinic</h1>'
            '<img src="a.webp?v=1606811841689-23" alt="x">'
            '<a href="tel:+302310123456">Call</a></section>',
        )
        out = tmp_path / "t3"
        mod.extract_template(site, out, name="x")
        assert yaml.safe_load((out / "client.example.yaml").read_text())["phone"] == "+302310123456"

    def test_plain_formatted_phone_still_found_without_a_tel_link(self, mod, tmp_path):
        import yaml

        site = self._site(
            tmp_path, '<section id="c"><h1>Clinic</h1><p>+30 2310 123 456</p></section>'
        )
        out = tmp_path / "t4"
        mod.extract_template(site, out, name="x")
        assert "2310" in yaml.safe_load((out / "client.example.yaml").read_text())["phone"]


# ── SEO, GEO, social and security are template output, not an afterthought ───

_CLIENT = {
    "brand_name": "Aegean Dental",
    "city": "Thessaloniki",
    "phone": "+30 2310 000000",
    "email": "hello@example.gr",
    "address": "Tsimiski 42",
    "postal_code": "546 23",
    "country": "GR",
    "latitude": 40.6329,
    "longitude": 22.9419,
    "site_url": "https://aegean-dental.gr",
    "accent": "#0f5d4a",
    "services": [{"title": "Implants", "body": "Planned from a 3D scan.", "price": "from 900 EUR"}],
}


@pytest.fixture
def rendered(mod, tmp_path):
    tpl = mod.load_template("templates/websites/dentist")
    out = tmp_path / "site"
    mod.apply_template(tpl, dict(_CLIENT), out)
    return out


@pytest.mark.unit
class TestStructuredDataAndGeo:
    def test_emits_localbusiness_json_ld(self, rendered):
        import json
        import re

        html = (rendered / "index.html").read_text(encoding="utf-8")
        block = re.search(r'<script type="application/ld\+json">(.*?)</script>', html, re.S)
        assert block, "no JSON-LD block — local search needs structured data"
        data = json.loads(block.group(1))
        assert data["@type"] in ("Dentist", "LocalBusiness", "MedicalBusiness")
        assert data["name"] == "Aegean Dental"
        assert data["telephone"] == "+30 2310 000000"

    def test_json_ld_carries_a_postal_address(self, rendered):
        import json
        import re

        html = (rendered / "index.html").read_text(encoding="utf-8")
        data = json.loads(re.search(r'ld\+json">(.*?)</script>', html, re.S).group(1))
        addr = data["address"]
        assert addr["@type"] == "PostalAddress"
        assert addr["streetAddress"] == "Tsimiski 42"
        assert addr["addressLocality"] == "Thessaloniki"
        assert addr["postalCode"] == "546 23"

    def test_json_ld_carries_geo_coordinates(self, rendered):
        import json
        import re

        html = (rendered / "index.html").read_text(encoding="utf-8")
        data = json.loads(re.search(r'ld\+json">(.*?)</script>', html, re.S).group(1))
        assert data["geo"]["@type"] == "GeoCoordinates"
        assert float(data["geo"]["latitude"]) == pytest.approx(40.6329)

    def test_emits_geo_meta_tags(self, rendered):
        html = (rendered / "index.html").read_text(encoding="utf-8")
        for tag in (
            'name="geo.region"',
            'name="geo.placename"',
            'name="geo.position"',
            'name="ICBM"',
        ):
            assert tag in html, f"missing {tag}"

    def test_emits_canonical_and_robots(self, rendered):
        html = (rendered / "index.html").read_text(encoding="utf-8")
        assert 'rel="canonical" href="https://aegean-dental.gr' in html
        assert 'name="robots"' in html

    def test_writes_sitemap_and_robots_txt(self, rendered):
        assert (rendered / "sitemap.xml").exists()
        robots = (rendered / "robots.txt").read_text(encoding="utf-8")
        assert "Sitemap: https://aegean-dental.gr/sitemap.xml" in robots

    def test_geo_is_omitted_cleanly_when_not_supplied(self, mod, tmp_path):
        """A client without coordinates must not emit an empty geo block."""
        import json
        import re

        client = {k: v for k, v in _CLIENT.items() if k not in ("latitude", "longitude")}
        out = tmp_path / "nogeo"
        mod.apply_template(mod.load_template("templates/websites/dentist"), client, out)
        html = (out / "index.html").read_text(encoding="utf-8")
        data = json.loads(re.search(r'ld\+json">(.*?)</script>', html, re.S).group(1))
        assert "geo" not in data
        assert "geo.position" not in html


@pytest.mark.unit
class TestSocialCards:
    def test_emits_open_graph(self, rendered):
        html = (rendered / "index.html").read_text(encoding="utf-8")
        for tag in ("og:title", "og:description", "og:image", "og:url", "og:type", "og:locale"):
            assert f'property="{tag}"' in html, f"missing {tag}"

    def test_emits_twitter_card(self, rendered):
        html = (rendered / "index.html").read_text(encoding="utf-8")
        assert 'name="twitter:card" content="summary_large_image"' in html
        assert 'name="twitter:image"' in html

    def test_generates_a_social_share_image(self, rendered):
        """Every client gets an OG image without an image model or a designer."""
        og = rendered / "public" / "og-image.svg"
        assert og.exists(), "no OG image generated"
        svg = og.read_text(encoding="utf-8")
        assert 'width="1200"' in svg and 'height="630"' in svg
        assert "Aegean Dental" in svg

    def test_og_image_uses_the_client_palette(self, mod, tmp_path):
        out = tmp_path / "purple"
        mod.apply_template(
            mod.load_template("templates/websites/dentist"),
            {**_CLIENT, "accent": "#7a3b8f"},
            out,
        )
        assert "#7a3b8f" in (out / "public" / "og-image.svg").read_text(encoding="utf-8")

    def test_og_image_alt_is_provided(self, rendered):
        html = (rendered / "index.html").read_text(encoding="utf-8")
        assert 'property="og:image:alt"' in html


@pytest.mark.unit
class TestSecurityByDefault:
    def test_ships_security_headers(self, rendered):
        headers = (rendered / "_headers").read_text(encoding="utf-8")
        for directive in (
            "Content-Security-Policy",
            "X-Content-Type-Options: nosniff",
            "Referrer-Policy",
            "X-Frame-Options",
            "Permissions-Policy",
            "Strict-Transport-Security",
        ):
            assert directive in headers, f"missing {directive}"

    def test_csp_does_not_allow_unsafe_inline_scripts(self, rendered):
        headers = (rendered / "_headers").read_text(encoding="utf-8")
        csp = next(line for line in headers.splitlines() if "Content-Security-Policy" in line)
        assert "'unsafe-eval'" not in csp
        assert "script-src" in csp and "'unsafe-inline'" not in csp.split("script-src")[1][:60]

    def test_no_third_party_script_origins(self, rendered):
        """Zero external JS: nothing to compromise, nothing to rate-limit."""
        import re

        html = (rendered / "index.html").read_text(encoding="utf-8")
        external = re.findall(r'<script[^>]+src="(https?://[^"]+)"', html)
        assert not external, f"third-party scripts pull in supply-chain risk: {external}"

    def test_external_links_are_not_exploitable(self, rendered):
        import re

        html = (rendered / "index.html").read_text(encoding="utf-8")
        for match in re.finditer(r'<a\b[^>]*href="https?://[^"]+"[^>]*>', html):
            tag = match.group(0)
            if 'target="_blank"' in tag:
                assert "noopener" in tag, f"target=_blank without noopener: {tag}"

    def test_no_secrets_in_output(self, rendered):
        from orchestrator.generators.website_validator import WebsiteQualityValidator
        import asyncio

        check = (
            asyncio.get_event_loop_policy()
            .new_event_loop()
            .run_until_complete(WebsiteQualityValidator()._check_secret_exposure(rendered))
        )
        assert check.passed

    def test_ships_a_security_txt(self, rendered):
        txt = (rendered / ".well-known" / "security.txt").read_text(encoding="utf-8")
        assert "Contact:" in txt


@pytest.mark.unit
class TestMotionIsAnEnhancement:
    def test_micro_interactions_are_present(self, rendered):
        css = (rendered / "styles.css").read_text(encoding="utf-8")
        assert ":hover" in css and "transition" in css

    def test_three_d_effects_are_present(self, rendered):
        css = (rendered / "styles.css").read_text(encoding="utf-8")
        assert "perspective" in css and "preserve-3d" in css and "rotateX" in css

    def test_all_motion_is_behind_a_reduced_motion_guard(self, rendered):
        css = (rendered / "styles.css").read_text(encoding="utf-8")
        assert "prefers-reduced-motion: no-preference" in css
        assert "prefers-reduced-motion: reduce" in css

    def test_content_is_never_left_hidden_without_javascript(self, rendered):
        """A reveal animation must not be the only thing making text visible."""
        css = (rendered / "styles.css").read_text(encoding="utf-8")
        assert ".no-js .reveal" in css or "@media (prefers-reduced-motion: reduce) { .reveal" in css

    def test_the_enhancement_script_is_deferred_and_local(self, rendered):
        html = (rendered / "index.html").read_text(encoding="utf-8")
        assert '<script src="script.js" defer></script>' in html
        assert (rendered / "script.js").exists()
