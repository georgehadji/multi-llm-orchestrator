"""What the auditor can see, and — the part that matters — what it cannot.

Evidence availability is the mechanism that keeps the score honest. A build
directory cannot tell you whether TLS is configured; if `Evidence.HTTP` is not
in `available`, every check needing it is outstanding by construction rather
than by an author remembering to be careful.
"""

from __future__ import annotations

import pytest

from orchestrator.generators.wf100.evidence import (
    BusinessRecord,
    HttpResponse,
    Page,
    SiteEvidence,
)
from orchestrator.generators.wf100.standard import Evidence

pytestmark = pytest.mark.unit


@pytest.fixture
def site(tmp_path):
    (tmp_path / "index.html").write_text(
        '<!doctype html><html lang="el"><head><title>Κλινική</title>'
        '<link rel="stylesheet" href="styles.css">'
        "<style>body{margin:0}</style></head>"
        "<body><main><h1>Οδοντιατρείο</h1>"
        '<img src="hero.webp" alt="Η κλινική"><a href="about.html">Σχετικά</a></main>'
        '<script src="app.js"></script><script>console.log(1)</script></body></html>',
        encoding="utf-8",
    )
    (tmp_path / "about.html").write_text(
        '<!doctype html><html lang="el"><head><title>Σχετικά</title></head>'
        "<body><h1>Σχετικά</h1></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "styles.css").write_text("h1{color:#0f5d4a}", encoding="utf-8")
    (tmp_path / "app.js").write_text("document.title=document.title", encoding="utf-8")
    (tmp_path / "hero.webp").write_bytes(b"\x00" * 4096)
    (tmp_path / "robots.txt").write_text("User-agent: *\nAllow: /\n", encoding="utf-8")
    return tmp_path


class TestDirectoryMode:
    def test_collects_every_html_page(self, site):
        ev = SiteEvidence.from_directory(site)
        assert {p.path for p in ev.pages} == {"index.html", "about.html"}

    def test_reads_linked_and_inline_css(self, site):
        ev = SiteEvidence.from_directory(site)
        assert "#0f5d4a" in ev.styles
        assert "margin:0" in ev.styles

    def test_reads_linked_and_inline_js(self, site):
        ev = SiteEvidence.from_directory(site)
        assert "document.title" in ev.scripts
        assert "console.log(1)" in ev.scripts

    def test_records_asset_sizes(self, site):
        ev = SiteEvidence.from_directory(site)
        assert ev.assets["hero.webp"] == 4096

    def test_reads_sibling_text_files(self, site):
        ev = SiteEvidence.from_directory(site)
        assert "User-agent" in ev.files["robots.txt"]

    def test_a_directory_offers_static_evidence_only(self, site):
        ev = SiteEvidence.from_directory(site)
        assert ev.available == {
            Evidence.MARKUP,
            Evidence.STYLES,
            Evidence.SCRIPTS,
            Evidence.ASSETS,
        }

    def test_a_directory_can_never_supply_live_evidence(self, site):
        ev = SiteEvidence.from_directory(site)
        for kind in (Evidence.HTTP, Evidence.BROWSER, Evidence.FIELD_DATA, Evidence.HUMAN):
            assert kind not in ev.available

    def test_a_business_record_adds_record_evidence(self, site):
        ev = SiteEvidence.from_directory(site, record=BusinessRecord(name="Κλινική"))
        assert Evidence.RECORD in ev.available

    def test_an_empty_directory_yields_no_markup(self, tmp_path):
        ev = SiteEvidence.from_directory(tmp_path)
        assert Evidence.MARKUP not in ev.available
        assert ev.pages == ()

    def test_a_missing_directory_is_a_clear_error(self, tmp_path):
        with pytest.raises(ValueError, match="not a directory"):
            SiteEvidence.from_directory(tmp_path / "nope")


class TestPage:
    def test_page_exposes_visible_text_without_markup(self, site):
        ev = SiteEvidence.from_directory(site)
        page = ev.page("index.html")
        assert "Οδοντιατρείο" in page.text
        assert "<h1>" not in page.text

    def test_script_and_style_bodies_are_not_visible_text(self, site):
        ev = SiteEvidence.from_directory(site)
        assert "console.log" not in ev.page("index.html").text

    def test_page_lists_its_internal_links(self, site):
        ev = SiteEvidence.from_directory(site)
        assert "about.html" in ev.page("index.html").links


class TestUrlMode:
    def _fetcher(self, pages):
        def fetch(url, method="GET"):
            if url not in pages:
                return HttpResponse(url=url, status=404, headers={}, body=b"")
            body, headers = pages[url]
            return HttpResponse(
                url=url,
                status=200,
                headers=headers,
                body=body.encode("utf-8") if isinstance(body, str) else body,
            )

        return fetch

    def test_url_mode_supplies_live_evidence(self):
        fetch = self._fetcher(
            {
                "https://x.test/": (
                    "<html lang='el'><body><h1>A</h1></body></html>",
                    {"content-type": "text/html", "strict-transport-security": "max-age=63072000"},
                )
            }
        )
        ev = SiteEvidence.from_url("https://x.test/", fetch=fetch)
        assert Evidence.HTTP in ev.available
        assert Evidence.MARKUP in ev.available

    def test_url_mode_still_cannot_supply_field_data_or_judgement(self):
        fetch = self._fetcher({"https://x.test/": ("<html><body>hi</body></html>", {})})
        ev = SiteEvidence.from_url("https://x.test/", fetch=fetch)
        assert Evidence.FIELD_DATA not in ev.available
        assert Evidence.HUMAN not in ev.available
        assert Evidence.BROWSER not in ev.available

    def test_crawl_follows_internal_links(self):
        fetch = self._fetcher(
            {
                "https://x.test/": ('<html><body><a href="/b">b</a></body></html>', {}),
                "https://x.test/b": ("<html><body><h1>B</h1></body></html>", {}),
            }
        )
        ev = SiteEvidence.from_url("https://x.test/", fetch=fetch)
        assert len(ev.pages) == 2

    def test_crawl_does_not_leave_the_origin(self):
        fetch = self._fetcher(
            {
                "https://x.test/": (
                    '<html><body><a href="https://other.test/x">away</a></body></html>',
                    {},
                ),
                "https://other.test/x": ("<html><body>nope</body></html>", {}),
            }
        )
        ev = SiteEvidence.from_url("https://x.test/", fetch=fetch)
        assert len(ev.pages) == 1

    def test_crawl_honours_the_page_cap(self):
        pages = {"https://x.test/": ("".join(f'<a href="/p{i}">x</a>' for i in range(20)), {})}
        for i in range(20):
            pages[f"https://x.test/p{i}"] = (f"<h1>{i}</h1>", {})
        ev = SiteEvidence.from_url("https://x.test/", fetch=self._fetcher(pages), max_pages=5)
        assert len(ev.pages) == 5

    def test_response_headers_are_case_insensitive(self):
        fetch = self._fetcher({"https://x.test/": ("<h1>a</h1>", {"Content-Type": "text/html"})})
        ev = SiteEvidence.from_url("https://x.test/", fetch=fetch)
        assert ev.http.header("https://x.test/", "content-type") == "text/html"

    def test_an_unreachable_entry_url_is_a_clear_error(self):
        fetch = self._fetcher({})
        with pytest.raises(ValueError, match="404"):
            SiteEvidence.from_url("https://x.test/", fetch=fetch)
