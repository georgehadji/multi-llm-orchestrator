# Image Generation Optimization Plan

**Date:** 2026-06-17
**Context:** Website Generator image pipeline — fixes + optimizations
**Architectural constraint:** Respect hexagonal ports & adapters, 4 import-linter contracts, lazy-init patterns

---

## Current State

The website generator (`generators/website_generator.py`) produces 6 image types per site
via two paths: LLM (`infrastructure/image_client.py`) or SVG fallback (`generators/image_generator.py`).
Images are generated sequentially in a `for` loop with no caching, no parallelism,
and three known breakages in the favicon pipeline.

## PHASE 1: Critical Fixes (broken features — ship before anything else)

### Fix 1: Download Recraft SVG URLs from OpenRouter

**Problem:** Recraft V4 Vector models return SVG URLs (not base64). `image_client.py:152-153`
sets `result.image_url` but logs "not downloading" — the favicon/apple-icon
models in `_IMAGE_MODEL_MAP` (`website_generator.py:900-908`) use Recraft Vector
as their primary model for icons, so favicons are silently lost.

**File:** `orchestrator/infrastructure/image_client.py`

**Change:** In `generate()`, after `_parse_response()`, when `result.image_url` is
set but `result.image_data` is None, download the URL:

```python
# Line 152-153 (current):
if result.success and output_path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if result.image_data:
        output_path.write_bytes(result.image_data)
    elif result.image_url:
        logger.info("Image URL received (not downloading): %s", result.image_url[:80])

# Replace with:
if result.success and output_path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if result.image_data:
        output_path.write_bytes(result.image_data)
    elif result.image_url:
        try:
            async with httpx.AsyncClient(timeout=30) as http:
                resp = await http.get(result.image_url)
                if resp.status_code == 200:
                    output_path.write_bytes(resp.content)
                    result.image_data = resp.content
        except Exception as e:
            logger.warning("Failed to download image URL %s: %s", result.image_url[:80], e)
```

**Risk:** LOW — adds one HTTP GET call inside the existing async context.
**Verification:** Run website generation with `--image-model auto` on a site with
`favicon`/`apple-touch-icon` — verify `.png` files exist in `public/images/`.

---

### Fix 2: Add favicon/apple-touch-icon to Next.js `layout.tsx` `<head>`

**Problem:** `_assemble_nextjs_page` (`website_generator.py:1429-1490`) generates
a `layout.tsx` with metadata, OG tags, and security headers — but zero `<link rel="icon">`
or `<link rel="apple-touch-icon">` tags. The HTML assembly path (`_assemble_html_page`,
line 1214-1216) has them, but Next.js sites (the default framework) do not.

**File:** `orchestrator/generators/website_generator.py`

**Change:** Add icon links inside the `<head>` block of the layout.tsx template
(after the JSON-LD script tag, before `</head>`):

```python
# After line ~1488 (the JSON-LD script tag), insert:
"        {/* Favicon + PWA icons */}\n"
"        <link rel=\"icon\" type=\"image/png\" sizes=\"32x32\" href=\"/images/favicon.png\" />\n"
"        <link rel=\"icon\" type=\"image/svg+xml\" href=\"/favicon.svg\" />\n"
"        <link rel=\"apple-touch-icon\" sizes=\"180x180\" href=\"/images/apple-touch-icon.png\" />\n"
```

(These must use `dangerouslySetInnerHTML` or be placed as standard `<link>` tags
inside the `<head>` JSX block — Next.js App Router supports both.)

**Risk:** LOW — purely additive, no code paths removed.
**Verification:** Generate a Next.js site, inspect `<head>` in `layout.tsx`,
confirm the three `<link>` tags are present.

---

### Fix 3: Sync SVG fallback paths with HTML/Next.js assembly references

**Problem:** The SVG fallback (`image_generator.py:59-61`) writes `favicon.svg`
and `apple-touch-icon.svg` to `public/` (the root public dir). But the HTML
assembly at `website_generator.py:1214-1216` references `/images/favicon.png`
and `/images/apple-touch-icon.png`. The PNG files don't exist in the SVG fallback.

**File:** `orchestrator/generators/image_generator.py`

**Change:** Write favicon and apple-touch-icon SVGs to `public/images/` (alongside
other generated images) instead of `public/`:

```python
# Lines 59-61 (current):
_write_svg("favicon.svg", svg_icon(32, 6, primary, site_name[0].upper()), img_dir.parent)
_write_svg(
    "apple-touch-icon.svg", svg_icon(180, 36, bg, site_name[0].upper()), img_dir.parent
)

# Replace with:
_write_svg("favicon.svg", svg_icon(32, 6, primary, site_name[0].upper()), img_dir)
_write_svg(
    "apple-touch-icon.svg", svg_icon(180, 36, bg, site_name[0].upper()), img_dir
)
```

Also add favicon/apple-touch-icon SVG generation to the LLM path's output extension
logic (`_generate_images_llm`): when the selected model is a Recraft SVG model,
write `.svg` instead of `.png`:

```python
# In _generate_images_llm, around line 1086:
ext = ".svg" if "recraft" in img_model and "vector" in img_model else ".png"
output_path = img_dir / f"{img['name']}{ext}"
```

**Risk:** LOW — changes file placement only, no API changes.
**Verification:** Run website generation with `--image-model none`, verify
`public/images/favicon.svg` exists and is referenced correctly.

---

## PHASE 2: High-Impact Optimizations (speed + cost)

### Opt 4: Parallel Image Generation

**Problem:** `_generate_images_llm` (`website_generator.py:1084`) iterates images
sequentially. With 9–13 images at ~5s each, wall-clock time is 45–65 seconds.

**File:** `orchestrator/generators/website_generator.py`

**Change:** Replace the `for` loop with `asyncio.gather` bounded by a semaphore:

```python
# Replace the sequential loop:
success_count = 0
for img in images:
    ...

# With:
semaphore = asyncio.Semaphore(3)  # max 3 concurrent image gen calls

async def _gen_one(img):
    img_type = img["name"].split("-")[0]
    img_type = img_type if img_type in self._IMAGE_MODEL_MAP else "section"
    img_model = model if model != "auto" else self._select_image_model(img_type, "auto")
    ext = ".svg" if "recraft" in img_model and "vector" in img_model else ".png"
    async with semaphore:
        try:
            result = await client.generate(
                prompt=img["prompt"],
                model=img_model,
                width=img["width"],
                height=img["height"],
                output_path=img_dir / f"{img['name']}{ext}",
            )
            return (img, result, img_model)
        except Exception as e:
            return (img, None, img_model)

results = await asyncio.gather(*[_gen_one(img) for img in images])

success_count = 0
models_used = set()
for img, result, img_model in results:
    models_used.add(img_model)
    if result and result.success:
        success_count += 1
```

**Risk:** LOW — semaphore(3) keeps API pressure bounded. Already tested at
OpenRouter concurrency=3 in the section generation pipeline.
**Expected gain:** 45s → 15s wall-clock time (3× faster).
**Verification:** Run with `time` command, compare before/after.

---

### Opt 5: DiskCache for Images

**Problem:** Regenerating the same site (same description, same design colors)
with the same model redownloads every image from scratch. During development,
users iterate on prompts and regenerate 2–5 times.

**File:** `orchestrator/infrastructure/image_client.py`

**Change:** Add a `DiskCache` parameter to `ImageGenClient.__init__` and use it
in `generate()`:

```python
# In __init__:
def __init__(self, api_key=None, base_url=OPENROUTER_BASE, cache=None):
    self._cache = cache  # optional DiskCache

# In generate(), before the API call:
import hashlib

if self._cache:
    cache_key = hashlib.sha256(
        f"{model}:{prompt}:{width}x{height}".encode()
    ).hexdigest()
    cached = await self._cache.get(cache_key, "", 4096, "", 0.0)
    if cached:
        img_bytes = base64.b64decode(cached)
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(img_bytes)
        return ImageGenResult(success=True, image_data=img_bytes)

# ... API call as before ...

# After successful generation, cache the result:
if result.success and result.image_data and self._cache:
    await self._cache.put(
        cache_key, "", 4096,
        base64.b64encode(result.image_data).decode(),
        len(result.image_data), 0, "", 0.0
    )
```

**Architectural note:** The `DiskCache` is already used in `infrastructure/llm_client.py:170`.
Following the same pattern keeps the image client consistent with the LLM client.
The cache is an infrastructure concern — no domain protocol change needed.

**Wire into website generator:** Pass `DiskCache()` to `ImageGenClient` in
`_generate_images`:

```python
from ..cache import DiskCache
client = ImageGenClient(cache=DiskCache())
```

**Risk:** LOW — additive, follows existing `llm_client.py` pattern.
**Expected gain:** 0s for repeat generations of identical images. Cache hit
rate ~80% during iterative development.
**Verification:** Generate the same site twice, confirm images only download once.

---

### Opt 6: Better Width/Height Enforcement

**Problem:** Dimensions are embedded as text (`f"Generate a {width}x{height} image"`)
in `image_client.py:106`. Some models ignore text-based dimension hints.

**Change:** Add a negative prompt to the system message that reinforces aspect ratio:

```python
system_prompt = (
    "Generate a high-quality image based on the user's description. "
    "Do NOT include any text, typography, or words in the image. "
    "Output only the image content. "
    "The image MUST match the requested dimensions exactly — do NOT crop, "
    "letterbox, or change the aspect ratio."
)
```

**Risk:** LOW — prompt engineering only, no code changes.
**Expected gain:** Higher dimension accuracy, fewer off-aspect-ratio images.

---

## PHASE 3: Quality & Robustness

### Opt 7: Per-Image Timeout to Prevent Head-of-Line Blocking

**File:** `orchestrator/generators/website_generator.py`

**Change:** Wrap each `client.generate()` call in `asyncio.wait_for(..., timeout=20)`:

```python
result = await asyncio.wait_for(
    client.generate(prompt=img["prompt"], model=img_model, ...),
    timeout=20.0,
)
```

**Risk:** LOW — 20s is generous; most images complete in 3–8s.
**Verification:** Test with a slow model; confirm timeout triggers and remaining
images still generate.

---

### Opt 8: Better HTTP Error Retry in `ImageGenClient`

**Problem:** `image_client.py:132-136` only retries on HTTP 429. Other 5xx errors
(503 Service Unavailable, 502 Bad Gateway) skip retry and fail immediately.

**Change:** Extend retry condition:

```python
if response.status_code == 429:
    logger.warning("Image gen rate limited, retry %d/%d", attempt + 1, MAX_RETRIES)
    await asyncio.sleep(2**attempt)
    continue

if response.status_code >= 500:
    logger.warning("Image gen server error %d, retry %d/%d",
                   response.status_code, attempt + 1, MAX_RETRIES)
    await asyncio.sleep(2**attempt)
    continue
```

**Risk:** LOW — follows the same pattern as the 429 retry.
**Verification:** Already tested implicitly by the existing retry loop structure.

---

### Opt 9: Lazy Loading Injection in Component Prompts

**Problem:** LLM-generated section components often produce `<img>` tags without
`loading="lazy"`. The validator (`website_validator.py:218`) warns about it but
never enforces it.

**File:** `orchestrator/generators/website_generator.py`

**Change:** Add a rule to the `_build_section_prompt` RULES section:

```
11. All <img> tags MUST include loading="lazy" and decoding="async" attributes.
```

**Risk:** LOW — prompt engineering only.
**Verification:** Generate a site with image-heavy sections, grep output for `loading="lazy"`.

---

### Opt 10: SVG Fallback — Add Logo + Section Thumbnails

**File:** `orchestrator/generators/image_generator.py`

**Change:** Add a `svg_logo()` function and `svg_section_thumb()` function, then
generate `logo.svg` and `section-{name}.svg` for each section:

```python
# Add logo
_write_svg("logo.svg", svg_logo(primary, accent, site_name), img_dir)

# Add section thumbnails
sections = getattr(config, "sections", [])
for section in sections:
    _write_svg(f"section-{section}.svg",
               svg_section_thumb(section, primary, accent, surface, text),
               img_dir)
```

(SVG helper functions follow the same pattern as existing `svg_hero`, `svg_og`, etc.)

**Risk:** LOW — additive only.
**Verification:** Run with `--image-model none`, verify `logo.svg` and
`section-hero.svg` (etc.) exist.

---

### Opt 11: Hook Up `image_optimizer.py` (Minimal — WebP Conversion)

**Problem:** `orchestrator/image_optimizer.py` (682 lines) implements a full
optimization pipeline (WebP, AVIF, JPEG, PNG, SVG optimizers) but is never called
anywhere. All generated images are saved as raw PNG (up to 2MB per hero image).

**Minimal fix:** Don't try to wire the full chain — just convert PNG → WebP for
hero/OG images in `_generate_images_llm` after generation:

```python
# After saving the image, if it's not SVG:
if result.image_data and not output_path.suffix == ".svg":
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(result.image_data))
        webp_path = output_path.with_suffix(".webp")
        img.save(webp_path, "WEBP", quality=85)
        # Update HTML references to use WebP for this image type
    except ImportError:
        pass  # PIL not installed — keep PNG
```

**Alternative (no PIL dependency):** Use the existing `image_optimizer.py`
`WebPOptimizer` class but make it functional (it currently returns input bytes
unchanged). Or add `Pillow>=10.0` to `pyproject.toml` optional deps.

**Risk:** MEDIUM — adds a new dependency or requires fixing the existing optimizer.
**Expected gain:** 60–80% file size reduction for hero/OG images.
**Verification:** Compare file sizes before/after.

---

### Opt 12: `--image-quality` CLI Flag

**File:** `orchestrator/cli.py` + `orchestrator/generators/website_generator.py`

**Change:** Add a `--image-quality` flag with three levels that map to different
model selection chains:

```python
# cli.py:
wp.add_argument("--image-quality", default="balanced",
    choices=["draft", "balanced", "premium"],
    help="Image quality tier (draft=cheap/fast, balanced=best VFM, premium=best quality)")

# website_generator.py — _IMAGE_MODEL_MAP becomes quality-aware:
_IMAGE_MODEL_TIERS = {
    "draft": {
        "favicon": ["sourceful/riverflow-v2-fast", ...],
        "hero-bg": ["black-forest-labs/flux.2-klein-4b", ...],
        ...
    },
    "balanced": { ... (current _IMAGE_MODEL_MAP) ... },
    "premium": {
        "favicon": ["recraft/recraft-v4-pro-vector", ...],
        "hero-bg": ["google/gemini-3-pro-image-preview", ...],
        ...
    },
}
```

**Risk:** LOW — pure data structure change, no logic changes.
**Verification:** Run with each quality tier, verify different models selected.

---

## Implementation Order (dependency-aware)

| Order | Fix | Phase | Blocks | Risk |
|-------|-----|-------|--------|------|
| 1 | Fix 1: SVG URL download | 1 | Nothing | LOW |
| 2 | Fix 2: Next.js favicon `<head>` | 1 | Nothing | LOW |
| 3 | Fix 3: SVG fallback path sync | 1 | Nothing | LOW |
| 4 | Opt 5: DiskCache for images | 2 | Nothing | LOW |
| 5 | Opt 4: Parallel image gen | 2 | Opt 5 (same file region) | LOW |
| 6 | Opt 7: Per-image timeout | 2 | Opt 4 (depends on gather) | LOW |
| 7 | Opt 6: Width/height enforcement | 2 | Nothing | LOW |
| 8 | Opt 8: HTTP 5xx retry | 3 | Nothing | LOW |
| 9 | Opt 9: Lazy loading in prompts | 3 | Nothing | LOW |
| 10 | Opt 10: SVG fallback additions | 3 | Fix 3 | LOW |
| 11 | Opt 11: WebP conversion | 3 | Nothing | MEDIUM |
| 12 | Opt 12: `--image-quality` flag | 3 | Nothing | LOW |

## Files Changed

| File | Phases | Changes |
|------|--------|---------|
| `orchestrator/infrastructure/image_client.py` | 1, 2, 3 | SVG download, DiskCache, 5xx retry, dimension prompts |
| `orchestrator/generators/website_generator.py` | 1, 2, 3 | Next.js favicon, parallel gen, timeout, lazy loading prompt, quality tiers |
| `orchestrator/generators/image_generator.py` | 1, 3 | SVG path sync, logo + section thumbnails |
| `orchestrator/cli.py` | 3 | `--image-quality` flag |
| `pyproject.toml` | 3 | Optional: `Pillow>=10.0` for WebP conversion |

## Architectural Compliance

| Contract | Impact |
|----------|--------|
| Contract 1 (domain-purity) | No domain changes needed — image generation is infrastructure |
| Contract 2 (app-no-infra) | No new infra imports from application layer |
| Contract 3 (app-no-engine) | No engine.py imports |
| Contract 4 (engine-core-no-infra) | No engine_core changes |
| Contract 5 (root-no-infra) | No new root-module infrastructure imports — website_generator is in generators/ |

All changes are either infrastructure (`image_client.py`) or feature modules
(`website_generator.py`, `image_generator.py`, `cli.py`). No import-linter
contracts are violated — none of these modules participate in the enforced
boundaries.
