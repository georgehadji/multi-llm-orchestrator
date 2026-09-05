# WF-100 — Website Factory Quality Standard v1.0

<!-- Generated from orchestrator/generators/wf100/standard.py. Do not edit by
     hand: regenerate with `python -m orchestrator website-audit --catalogue`. -->

A hundred checks, one point each, applied before a site launches.

> **Launch rule:** score >= 90 **and** zero critical failures.
> An *unverified* critical check blocks launch too — an unread smoke alarm is not
> an absence of fire.

## Scoring

| Status | Points | In the denominator? |
|---|---|---|
| Pass | earned | yes |
| Fail | none | yes |
| **Outstanding** — could not be decided | **none** | **yes** |
| Not applicable — nothing to judge | none | no, with the reason recorded |

Outstanding is the status that matters. A check the auditor could not decide
earns nothing and stays in the denominator: it is a debt against the score,
visible until someone clears it. It is never upgraded to a pass because the
tool ran cleanly.

## Verification levels

| Level | Who decides | Checks |
|---|---|---|
| Level 1 — automated | the auditor, from the build or the response | 69 |
| Level 2 — machine-assisted, needs confirmation | the auditor observes; a person or a live deployment confirms | 18 |
| Level 3 — human review | a person, reading the site and knowing the business | 13 |

## Coverage by this auditor

82 of 100 checks have an implementation. The remaining 18 need a browser, field data from real users, or a person — and are reported outstanding rather than assumed.

## The checks

### A. Architecture & Code — 10 points

| ID | Check | Level | Needs | Automated | Critical |
|---|---|---|---|---|---|
| A1 | Semantic HTML5 structure | automated | HTML | yes |  |
| A2 | Exactly one logical H1 per page | automated | HTML | yes |  |
| A3 | Heading hierarchy without skipped levels | automated | HTML | yes |  |
| A4 | Descriptive, readable URLs | automated | HTML | yes |  |
| A5 | No copy-pasted content blocks across pages | automated | HTML | yes |  |
| A6 | No unused critical dependencies shipped | automated | HTML, JS | yes |  |
| A7 | No console errors on load | assisted | browser | no |  |
| A8 | No broken internal links | automated | HTML, files | yes |  |
| A9 | A working 404 page | automated | HTML, files | yes |  |
| A10 | Production build free of debug artifacts | automated | HTML, JS | yes |  |

### B. Performance — 15 points

| ID | Check | Level | Needs | Automated | Critical |
|---|---|---|---|---|---|
| B1 | LCP at or under 2.5s for real users | assisted | field data | no |  |
| B2 | INP at or under 200ms for real users | assisted | field data | no |  |
| B3 | CLS at or under 0.1 for real users | assisted | field data | no |  |
| B4 | Images within a sane weight budget | automated | files | yes |  |
| B5 | Modern image formats served | automated | HTML, files | yes |  |
| B6 | Responsive image sizing | automated | HTML | yes |  |
| B7 | Below-the-fold media lazy-loaded | automated | HTML | yes |  |
| B8 | Critical assets prioritised | automated | HTML | yes |  |
| B9 | Font payload minimised | automated | CSS, HTML | yes |  |
| B10 | Third-party scripts minimised | automated | HTML | yes |  |
| B11 | CSS and JS minified and small | automated | files | yes |  |
| B12 | Compression enabled on the server | assisted | live response | yes |  |
| B13 | Browser caching configured | assisted | live response | yes |  |
| B14 | CDN in front of static assets where justified | assisted | live response | yes |  |
| B15 | No unnecessary render-blocking resources | automated | HTML | yes |  |

### C. Accessibility — 15 points

| ID | Check | Level | Needs | Automated | Critical |
|---|---|---|---|---|---|
| C1 | WCAG 2.2 AA baseline: no machine-detectable violations | automated | HTML | yes |  |
| C2 | Every function reachable by keyboard | human | a person | no |  |
| C3 | Focus indicator always visible | automated | CSS | yes |  |
| C4 | Logical tab order | automated | HTML | yes |  |
| C5 | Navigation and forms exposed to assistive technology | automated | HTML | yes |  |
| C6 | Every input has an associated label | automated | HTML | yes |  |
| C7 | Informative images carry useful alt text | automated | HTML | yes |  |
| C8 | Decorative images hidden from assistive technology | automated | HTML | yes |  |
| C9 | Sufficient colour contrast | automated | CSS, HTML | yes |  |
| C10 | Information never conveyed by colour alone | human | a person | no |  |
| C11 | Buttons have accessible names | automated | HTML | yes |  |
| C12 | Link text is meaningful out of context | automated | HTML | yes |  |
| C13 | No keyboard traps | human | a person | no |  |
| C14 | Reduced-motion preference respected | automated | CSS | yes |  |
| C15 | Page language declared | automated | HTML | yes |  |

### D. SEO — 15 points

| ID | Check | Level | Needs | Automated | Critical |
|---|---|---|---|---|---|
| D1 | Unique, well-sized title on every page | automated | HTML | yes |  |
| D2 | Unique, well-sized meta description on every page | automated | HTML | yes |  |
| D3 | H1 present, unique and specific | automated | HTML | yes |  |
| D4 | Content matches search intent | human | a person | no |  |
| D5 | Canonical URL declared | automated | HTML | yes |  |
| D6 | XML sitemap present and well-formed | automated | files | yes |  |
| D7 | robots.txt present and sane | automated | files | yes |  |
| D8 | Site is crawlable | automated | files | yes |  |
| D9 | No accidental noindex | automated | HTML | yes | **yes** |
| D10 | Internal linking connects the site | automated | HTML | yes |  |
| D11 | Descriptive anchor text | automated | HTML | yes |  |
| D12 | Open Graph and Twitter cards complete | automated | HTML | yes |  |
| D13 | Schema.org structured data present and valid | automated | HTML | yes |  |
| D14 | Images have alt text and descriptive filenames | automated | HTML | yes |  |
| D15 | Google Search Console verified and reporting | assisted | business record | yes |  |

### E. Local SEO — 10 points

| ID | Check | Level | Needs | Automated | Critical |
|---|---|---|---|---|---|
| E1 | Business name exactly as registered | assisted | HTML, business record | yes |  |
| E2 | Address correct and complete | assisted | HTML, business record | yes |  |
| E3 | Phone number correct and tappable | assisted | HTML, business record | yes |  |
| E4 | Opening hours published and machine-readable | automated | HTML | yes |  |
| E5 | Google Business Profile claimed and linked | assisted | HTML, business record | yes |  |
| E6 | Map to the location embedded | automated | HTML | yes |  |
| E7 | Local service information stated | automated | HTML | yes |  |
| E8 | Location relevance without keyword stuffing | automated | HTML | yes |  |
| E9 | NAP consistent across the whole site | automated | HTML | yes |  |
| E10 | Local structured data complete | automated | HTML | yes |  |

### F. UX & Conversion — 15 points

| ID | Check | Level | Needs | Automated | Critical |
|---|---|---|---|---|---|
| F1 | Value proposition visible above the fold | automated | HTML | yes |  |
| F2 | Primary CTA obvious | automated | CSS, HTML | yes |  |
| F3 | CTA reachable on mobile without hunting | automated | CSS, HTML | yes |  |
| F4 | Tap-to-call works | automated | HTML | yes |  |
| F5 | Booking route is obvious | automated | HTML | yes |  |
| F6 | Contact routes reachable from every page | automated | HTML | yes |  |
| F7 | Forms tested end to end | human | a person | no | **yes** |
| F8 | Form errors are understandable | automated | HTML, JS | yes |  |
| F9 | Successful submission is confirmed | automated | HTML, JS | yes |  |
| F10 | Trust signals present | automated | HTML | yes |  |
| F11 | Professional credentials easy to find | automated | HTML | yes |  |
| F12 | Services structured and scannable | automated | HTML | yes |  |
| F13 | Pricing information addressed | automated | HTML | yes |  |
| F14 | FAQ answers the real questions | automated | HTML | yes |  |
| F15 | No unnecessary friction in the conversion path | automated | HTML | yes |  |

### G. Security & Privacy — 10 points

| ID | Check | Level | Needs | Automated | Critical |
|---|---|---|---|---|---|
| G1 | HTTPS served with a valid certificate | assisted | live response | yes | **yes** |
| G2 | HTTP redirects to HTTPS | assisted | live response | yes |  |
| G3 | Security headers set | assisted | live response | yes |  |
| G4 | Administrative areas protected | automated | HTML, files | yes | **yes** |
| G5 | Strong administrative authentication with 2FA | assisted | business record | yes |  |
| G6 | Third-party code pinned and integrity-checked | automated | HTML | yes |  |
| G7 | Forms protected against spam and abuse | automated | HTML | yes |  |
| G8 | Privacy policy present and linked | automated | HTML | yes |  |
| G9 | Cookie consent where non-essential cookies are used | automated | HTML, JS | yes |  |
| G10 | Analytics fire only after consent | assisted | browser | no | **yes** |

### H. Content & Professional Quality — 10 points

| ID | Check | Level | Needs | Automated | Critical |
|---|---|---|---|---|---|
| H1 | No placeholder text anywhere | automated | HTML | yes | **yes** |
| H2 | No stock photography presented as the real practice | human | a person | no |  |
| H3 | No fabricated testimonials | human | a person | no | **yes** |
| H4 | No fabricated qualifications or credentials | human | a person | no | **yes** |
| H5 | Contact information verified against the client | assisted | HTML, business record | yes | **yes** |
| H6 | All imagery properly licensed | human | a person | no |  |
| H7 | Health and medical claims reviewed by the practitioner | human | a person | no | **yes** |
| H8 | Copy proofread | human | a person | no |  |
| H9 | Content reviewed on a real phone | human | a person | no |  |
| H10 | Client has approved the content | human | a person | no |  |

## Critical failures

These block launch regardless of the score. Some map onto a check; the rest are
detected independently, because a blocker outvoted by ninety-nine passing points
would defeat the purpose of naming it.

| Code | What it means |
|---|---|
| `MIXED_CONTENT` | Insecure subresources on a secure page |
| `EXPOSED_SECRET` | Credentials readable in the published build |
| `EXPOSED_PERSONAL_DATA` | Database or contact exports reachable from the site |
| `NO_CONTACT_ROUTE` | No phone link, email link or form anywhere |
| `ACCIDENTAL_NOINDEX` | Content pages tell search engines not to index them |
| `NO_VIEWPORT` | No viewport meta tag, so phones render the desktop layout |
| `BROKEN_HTTPS` | The site does not answer over HTTPS |

Fabricated testimonials (H3), fabricated credentials (H4) and unreviewed medical
claims (H7) are critical too, and are the checks no tool will ever close: they
ask whether something is *true*.
